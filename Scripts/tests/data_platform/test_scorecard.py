"""Weekly scorecard: Spix vs the market on settled published picks (rework A1 step 5)."""
import math

import pytest

from data_platform import scorecard as sc


def row(rid, *, key=None, fixture="f1", market="goals", confidence="high", outcome="hit", odds=2.0,
        p=0.6, market_p=0.5, close_p=0.52, clv=0.05, kickoff="2026-10-10T15:00:00Z", role="initial",
        minutes=60.0):
    return {
        "id": rid, "recommendation_id": rid, "selection_key": key or f"s{rid}", "fixture_id": fixture,
        "market": market, "confidence": confidence, "league": "EPL", "outcome": outcome, "odds": odds,
        "model_prob": p, "clv": clv, "kickoff": kickoff, "publication_role": role,
        "publication_reference": {"status": "available", "fair_probability": market_p} if market_p else
        {"status": "unavailable"},
        "closing_capture": {"status": "final", "minutes_before_kickoff": minutes,
                            "consensus": {"fair_probability": close_p}} if close_p else None,
    }


def test_each_selection_counts_once_at_its_first_publication():
    rows = [row(3, key="a", odds=1.9), row(1, key="a", odds=2.1, role="amendment"), row(2, key="b")]
    picks = sc.cohort(rows, "selections")
    assert sorted(p["recommendation_id"] for p in picks) == [1, 2]
    assert sc.cohort(rows, "initial") == [rows[0], rows[2]]


def test_asian_half_results_count_with_half_weight_and_pushes_not_at_all():
    assert sc.outcome_weight("hit") == (1.0, 1.0)
    assert sc.outcome_weight("half_hit") == (1.0, 0.5)
    assert sc.outcome_weight("half_miss") == (0.0, 0.5)
    assert sc.outcome_weight("push") == (None, 0.0)
    assert sc.outcome_weight(None) == (None, 0.0)


def test_core_metrics_compare_spix_with_the_market():
    rows = [row(1, outcome="hit", p=0.8, market_p=0.6), row(2, outcome="miss", p=0.8, market_p=0.6)]
    m = sc.metrics([sc.record(r) for r in rows])
    assert (m["published"], m["settled"]) == (2, 2)
    assert m["win_rate"] == pytest.approx(0.5)
    assert m["stated"] == pytest.approx(0.8) and m["market"] == pytest.approx(0.6)
    ll_spix = -(math.log(0.8) + math.log(0.2)) / 2
    ll_market = -(math.log(0.6) + math.log(0.4)) / 2
    assert m["log_loss_gap"] == pytest.approx(ll_spix - ll_market)  # positive: Spix worse than the market
    assert m["roi"] == pytest.approx((2.0 - 2) / 2)


def test_closing_line_value_in_price_and_in_fair_probability():
    m = sc.metrics([sc.record(row(1, clv=0.10, market_p=0.5, close_p=0.55)),
                    sc.record(row(2, clv=-0.02, market_p=0.5, close_p=0.45))])
    assert m["price_clv"] == pytest.approx(0.04) and m["price_clv_positive"] == pytest.approx(0.5)
    assert m["fair_clv"] == pytest.approx(((0.55 / 0.5 - 1) + (0.45 / 0.5 - 1)) / 2)
    assert m["closing_minutes_median"] == pytest.approx(60)


def test_unsettled_and_unpriced_picks_are_counted_but_not_scored():
    m = sc.metrics([sc.record(row(1, outcome=None, market_p=None, close_p=None)),
                    sc.record(row(2, market_p=None, close_p=None, clv=None))])
    assert (m["published"], m["settled"], m["with_market"]) == (2, 1, 0)
    assert m["log_loss_gap"] is None and m["fair_clv"] is None


def test_calibration_recovers_a_well_calibrated_slope():
    rows, rid = [], 0
    for p in (0.2, 0.4, 0.6, 0.8):
        for i in range(100):
            rid += 1
            rows.append(row(rid, p=p, outcome="hit" if i < p * 100 else "miss"))
    m = sc.metrics([sc.record(r) for r in rows])
    assert m["calibration_slope"] == pytest.approx(1.0, abs=0.05)
    assert m["calibration_intercept"] == pytest.approx(0.0, abs=0.05)
    assert m["calibration_error"] == pytest.approx(0.0, abs=0.01)


def test_overconfidence_shows_as_a_slope_below_one():
    rows, rid = [], 0
    for p, actual in ((0.3, 0.45), (0.7, 0.55)):
        for i in range(200):
            rid += 1
            rows.append(row(rid, p=p, outcome="hit" if i < actual * 200 else "miss"))
    m = sc.metrics([sc.record(r) for r in rows])
    assert m["calibration_slope"] < 0.5


def test_intervals_resample_fixtures_until_there_are_enough_matchweeks():
    rows = [row(i, fixture=f"f{i % 20}", outcome="hit" if i % 3 else "miss") for i in range(1, 121)]
    card = sc.build([dict(r) for r in rows], as_of="2026-10-19", resamples=200, seed=1)
    overall = card["cohorts"]["selections"]["overall"]
    assert card["interval_method"] == "fixture_bootstrap"
    low, high = overall["intervals"]["win_rate_minus_stated"]
    assert low <= overall["win_rate"] - overall["stated"] <= high


def test_small_slices_are_reported_without_scores():
    rows = [row(i, market="btts") for i in range(1, 6)] + [row(i, market="goals") for i in range(6, 60)]
    card = sc.build(rows, as_of="2026-10-19", resamples=50, seed=1)
    btts = card["cohorts"]["selections"]["by_market"]["btts"]
    assert btts["published"] == 5 and btts["insufficient_sample"] is True and "win_rate" not in btts
    assert "win_rate" in card["cohorts"]["selections"]["by_market"]["goals"]


def test_markdown_names_the_cohorts_and_the_headline_numbers():
    card = sc.build([row(i, fixture=f"f{i % 10}") for i in range(1, 60)], as_of="2026-10-19", resamples=50, seed=1)
    text = sc.markdown(card)
    assert "Spix scorecard" in text and "Each selection once" in text and "Official cohort" in text
    assert "Log-loss gap" in text and "Fair CLV" in text


def test_markdown_tables_keep_their_columns():
    rows = [row(i, fixture=f"f{i % 10}", confidence="medium") for i in range(1, 60)]
    text = sc.markdown(sc.build(rows, as_of="2026-10-19", resamples=50, seed=1))
    table_rows = [line for line in text.splitlines() if line.startswith("| goals · medium")]
    assert table_rows and table_rows[0].count("|") == len(sc.COLUMNS) + 2
    assert "60 min before kickoff" in text


def test_scorecard_runs_every_monday_morning_as_its_own_job(tmp_path):
    import plistlib
    from Scripts.ops import scorecard_launchd as schedule
    from Scripts.ops.match_read_launchd import build_paths
    paths = build_paths(root=tmp_path, plist_path=tmp_path / "job.plist")
    config = plistlib.loads(schedule.render(paths).encode())
    assert config["Label"] == "com.bettingrag.scorecard"
    assert config["StartCalendarInterval"] == {"Weekday": 1, "Hour": 8, "Minute": 0}
    assert config["ProgramArguments"][-2:] == ["-m", "Scripts.ops.scorecard"]
    assert schedule.operate("install", paths, dry_run=True)["configuration"]["Label"] == schedule.LABEL
    assert not paths.plist_path.exists()


def test_picks_are_split_by_the_prediction_changing_switches():
    from data_platform.scorecard import engine_stage
    assert engine_stage(None) == "defaults"  # before 17 Oct: no record, defaults
    base = {"version": "engine-switches.v1", "DOCUMENT_STORE": "sqlite", "SPIX_COUNT_DISPERSION": "floors",
            "SPIX_DEVIG_METHOD": "proportional"}
    assert engine_stage({"engine_switches": base}) == "defaults"  # the store does not change predictions
    staged = {**base, "SPIX_COUNT_DISPERSION": "fixed", "SPIX_DEVIG_METHOD": "power", "SPIX_MARKET_ANCHOR": "strict"}
    assert engine_stage({"engine_switches": staged}) == (
        "SPIX_COUNT_DISPERSION=fixed + SPIX_DEVIG_METHOD=power + SPIX_MARKET_ANCHOR=strict")
