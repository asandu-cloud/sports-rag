"""A4.2: per-market Platt calibration on one reference side per line (D27)."""
import math

import numpy as np
import pytest

from data_platform import calibration as cal
from data_platform.scorecard import _calibration_error
from Scripts.rag_ingest.core.prediction_guardrails import FIXED_DISPERSIONS
from Scripts.rag_ingest.prob_models import asian_total_equivalent_probability, asian_total_settlement_profile


def sel(market, side, point, p, y, *, w=1.0, fixture="1", stage="pre_match", kickoff="2026-10-10T15:00:00Z",
        mean=None, selected=False, market_prob=None):
    return {"fixture_id": fixture, "stage": stage, "kickoff": kickoff, "market": market, "side": side, "point": point,
            "model_prob": p, "y": y, "w": w, "mean": mean, "home": "Lyon", "away": "Lens", "selected": selected,
            "market_prob": market_prob}


@pytest.mark.parametrize("selection,expected", [
    (sel("goals", "over", 2.5, 0.5, 1), False),       # market-anchored (D21)
    (sel("goals", "over", 2.25, 0.5, 1), True),
    (sel("goals", "under", 3.5, 0.5, 1), True),
    (sel("moneyline", "home", None, 0.5, 1), False),  # 1X2 is market-anchored
    (sel("corners", "over", 9.5, 0.5, 1), True),
])
def test_scope_leaves_out_market_anchored_lines(selection, expected):
    assert cal.in_scope(selection) is expected


def test_every_side_is_turned_to_over_home_or_yes():
    assert cal.reference_side(sel("corners", "under", 9.5, 0.3, 1), 0.3) == (9.5, pytest.approx(0.7), 0.0)
    assert cal.reference_side(sel("btts", "No", None, 0.4, 0), 0.4) == (None, pytest.approx(0.6), 1.0)
    # Lens +0.5 is Lyon -0.5 on the other side.
    assert cal.reference_side(sel("spreads", "Lens", 0.5, 0.45, 1), 0.45) == (-0.5, pytest.approx(0.55), 0.0)
    assert cal.reference_side(sel("spreads", "Lyon", -0.5, 0.55, 0), 0.55) == (-0.5, 0.55, 0)
    assert cal.reference_side(sel("spreads", "Nice", -0.5, 0.55, 0), 0.55) is None


def test_complementary_sides_become_one_row_and_keep_the_pick_flag():
    rows = cal.canonical([sel("spreads", "Lyon", -0.5, 0.55, 0, market_prob=0.52),
                          sel("spreads", "Lens", 0.5, 0.45, 1, selected=True)], live_dispersion=False)
    assert len(rows) == 1
    assert rows[0]["p"] == 0.55 and rows[0]["y"] == 0 and rows[0]["selected"] and rows[0]["market_p"] == 0.52


def test_market_probability_is_turned_with_the_side():
    rows = cal.canonical([sel("corners", "under", 9.5, 0.4, 1, market_prob=0.45)], live_dispersion=False)
    assert rows[0]["market_p"] == pytest.approx(0.55) and rows[0]["p"] == pytest.approx(0.6)


def test_corners_are_priced_with_the_stage_one_fixed_dispersion():
    alpha, mean = FIXED_DISPERSIONS["corners"], 10.4
    expected = asian_total_equivalent_probability(asian_total_settlement_profile(mean, 9.75, "over", mean + alpha * mean ** 2))
    assert cal.live_probability(sel("corners", "Over", 9.75, 0.9, 1, mean=mean)) == pytest.approx(expected)
    assert cal.live_probability(sel("cards", "over", 3.5, 0.42, 1, mean=4.0)) == 0.42  # cards keep their floors


def simulate(n, intercept, slope, seed=1):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.9, n)
    truth = 1 / (1 + np.exp(-(intercept + slope * np.log(p / (1 - p)))))
    return [{"p": float(a), "y": float(b), "w": 1.0} for a, b in zip(p, rng.random(n) < truth)]


def test_platt_recovers_a_known_miscalibration():
    model = cal.Platt().fit(simulate(20000, 0.3, 0.7))
    assert model.intercept == pytest.approx(0.3, abs=0.05) and model.slope == pytest.approx(0.7, abs=0.05)


def test_a_fit_on_both_sides_cannot_see_a_directional_bias():
    one_side = simulate(20000, 0.3, 1.0)
    both = one_side + [{"p": 1 - r["p"], "y": 1 - r["y"], "w": r["w"]} for r in one_side]
    assert cal.Platt().fit(one_side).intercept == pytest.approx(0.3, abs=0.05)
    assert cal.Platt().fit(both).intercept == pytest.approx(0.0, abs=1e-6)


def test_rolling_origin_fits_only_on_earlier_days():
    seen = []

    class Recorder:
        def fit(self, rows):
            seen.append(sorted({r["kickoff"][:10] for r in rows}))
            return self

        def predict(self, p):
            return np.asarray(p)

    rows = [{"fixture_id": f"{d}-{i}", "kickoff": f"2026-10-{d:02d}T15:00:00Z", "p": 0.5, "y": 1.0, "w": 1.0}
            for d in (9, 10, 11) for i in range(3)]
    tested, _ = cal.rolling_origin(rows, Recorder, min_train_fixtures=3)
    assert seen == [["2026-10-09"], ["2026-10-09", "2026-10-10"]]
    assert sorted({r["kickoff"][:10] for r in tested}) == ["2026-10-10", "2026-10-11"]


def test_metrics_follow_the_scorecard_definitions():
    y, w, p = [1.0, 0.0, 1.0], [1.0, 0.5, 1.0], [0.7, 0.4, 0.2]
    m = cal.metrics(y, w, p)
    loss = (-math.log(0.7) - 0.5 * math.log(0.6) - math.log(0.2)) / 2.5
    assert m["log_loss"] == pytest.approx(loss)
    assert m["calibration_error"] == pytest.approx(_calibration_error(list(zip(y, w, p))))


def test_interval_resamples_whole_fixtures():
    rows = [{"fixture_id": "a", "w": 1.0}, {"fixture_id": "a", "w": 1.0}, {"fixture_id": "b", "w": 2.0}]
    low, high = cal.cluster_interval(rows, [0.3, 0.3, 0.3], [0.1, 0.1, 0.1], resamples=200)
    assert low == pytest.approx(0.2) and high == pytest.approx(0.2)
