"""A5.1: Spix's disagreement with the market, against outcomes and the closing price."""
import json

import numpy as np
import pytest

from data_platform import disagreement as dis
from data_platform.devig import power


def quote(market_key, side, point, odds, book, team=None):
    return {"market_key": market_key, "side": side, "team": team, "point": point, "odds": odds, "bookmaker": book,
            "period": "regulation_time"}


def candidate(market_key, side, point, p, *, team=None, selected=False):
    return {"option": {"market_key": market_key, "side": side, "team": team, "point": point, "_model_prob": p,
                       "bookmaker": "A"}, "selector_selected": selected}


def read(results, fid="7"):
    return {"fixture_api_id": fid, "stage": "pre_match", "evaluated_at": "2026-10-10T05:00:00Z",
            "kickoff": "2026-10-10T15:00:00Z", "read_json": json.dumps({"canonical_results": results})}


def result(group, inputs, candidates):
    return {"market": {"group": group}, "fixture": {"home_team": "Lyon", "away_team": "Lens"},
            "context": {"decision_audit": {"input_options": inputs, "candidates": candidates}}}


TOTALS = [quote("totals", "over", 2.5, 1.90, "A"), quote("totals", "under", 2.5, 1.95, "A"),
          quote("totals", "over", 2.5, 2.10, "B"), quote("totals", "under", 2.5, 1.75, "B"),
          quote("totals", "over", 2.5, 2.00, "C"), quote("totals", "under", 2.5, 1.85, "C")]


def market_over():
    return float(np.median([power([o, u])[0] for o, u in ((1.90, 1.95), (2.10, 1.75), (2.00, 1.85))]))


def test_the_line_is_taken_on_the_side_spix_prefers_with_the_power_consensus():
    lines = dis.read_lines(read([result("goals", TOTALS, [candidate("totals", "over", 2.5, 0.40, selected=True)])]))
    assert len(lines) == 1
    line = lines[0]
    # Spix says 40% over, the market about 49%: Spix prefers the under, priced from the complements.
    assert line["side"] == "under" and line["p_spix"] == pytest.approx(0.60)
    assert line["p_ref"] == pytest.approx(1 - market_over())
    assert line["best_odds"] == 1.95 and line["book_count"] == 3 and line["selected"] is False


def test_handicap_sides_are_teams_and_the_other_side_takes_the_opposite_line():
    inputs = [quote("spreads", None, -0.5, 1.85, "A", team="Lyon"), quote("spreads", None, 0.5, 2.00, "A", team="Lens")]
    lines = dis.read_lines(read([result("spreads", inputs, [candidate("spreads", None, -0.5, 0.40, team="Lyon")])]))
    assert [(l["side"], l["point"]) for l in lines] == [("lens", 0.5)]
    assert lines[0]["p_spix"] == pytest.approx(0.60)


def test_1x2_keeps_all_three_outcomes_from_the_projection():
    inputs = [quote("h2h", "home", None, 2.0, "A"), quote("h2h", "draw", None, 3.4, "A"), quote("h2h", "away", None, 3.9, "A")]
    res = result("moneyline", inputs, [{"option": {"market_key": "h2h", "side": "home", "_model_prob": None},
                                        "selector_selected": True}])
    res["projection"] = {"components": {"home_probability": 0.5, "draw_probability": 0.3, "away_probability": 0.2}}
    lines = {l["side"]: l for l in dis.read_lines(read([res]))}
    assert sorted(lines) == ["away", "draw", "home"] and lines["home"]["selected"]
    assert lines["draw"]["p_spix"] == 0.3
    assert lines["home"]["p_ref"] == pytest.approx(power([2.0, 3.4, 3.9])[0])


def test_a_line_without_a_two_sided_price_is_left_out():
    inputs = [quote("totals", "over", 3.5, 3.0, "A")]
    assert dis.read_lines(read([result("goals", inputs, [candidate("totals", "over", 3.5, 0.3)])])) == []


def test_closing_probability_uses_the_snapshot_books():
    payload = {"response": [{"bookmakers": [
        {"name": "A", "bets": [{"id": 5, "values": [{"value": "Over 2.5", "odd": "1.80"}, {"value": "Under 2.5", "odd": "2.05"}]}]},
        {"name": "B", "bets": [{"id": 8, "values": [{"value": "Yes", "odd": "1.80"}, {"value": "No", "odd": "2.00"}]}]}]}]}
    line = {"market": "goals", "market_key": "totals", "side": "under", "point": 2.5, "home": "Lyon", "away": "Lens"}
    assert dis.closing_probability(payload, line) == pytest.approx(power([1.80, 2.05])[1])


def simulate(n, c, seed=3):
    rng = np.random.default_rng(seed)
    p_ref = rng.uniform(0.2, 0.8, n)
    p_spix = np.clip(p_ref + rng.normal(0, 0.12, n), 0.02, 0.98)
    logit = lambda p: np.log(p / (1 - p))
    truth = 1 / (1 + np.exp(-(logit(p_ref) + c * (logit(p_spix) - logit(p_ref)))))
    return [{"fixture_id": str(i), "p_ref": float(a), "p_spix": float(b), "y": float(y), "w": 1.0}
            for i, (a, b, y) in enumerate(zip(p_ref, p_spix, rng.random(n) < truth))]


def test_the_outcome_weight_on_spix_is_recovered():
    assert dis.fit_outcome_weight(simulate(40000, 0.3)) == pytest.approx(0.3, abs=0.06)
    assert dis.fit_outcome_weight(simulate(40000, 0.0)) == pytest.approx(0.0, abs=0.06)


def test_the_closing_move_is_the_share_of_spix_deviation_the_market_closes():
    rows = simulate(500, 0.0)
    logit = lambda p: np.log(p / (1 - p))
    for r in rows:
        r["p_close"] = float(1 / (1 + np.exp(-(logit(r["p_ref"]) + 0.25 * (logit(r["p_spix"]) - logit(r["p_ref"]))))))
    assert dis.fit_closing_move(rows) == pytest.approx(0.25)


def test_both_fits_do_not_depend_on_which_side_is_kept():
    rows = simulate(4000, 0.2)
    for r in rows:
        r["p_close"] = min(0.99, r["p_ref"] + 0.01)
    flipped = [{**r, "p_ref": 1 - r["p_ref"], "p_spix": 1 - r["p_spix"], "p_close": 1 - r["p_close"], "y": 1 - r["y"]}
               for r in rows]
    assert dis.fit_outcome_weight(rows) == pytest.approx(dis.fit_outcome_weight(flipped))
    assert dis.fit_closing_move(rows) == pytest.approx(dis.fit_closing_move(flipped))


def test_bootstrap_resamples_whole_fixtures():
    rows = [{"fixture_id": "a", "v": 1.0}, {"fixture_id": "a", "v": 1.0}, {"fixture_id": "b", "v": 1.0}]
    assert dis.fixture_bootstrap(rows, lambda s: sum(r["v"] for r in s) / len(s), resamples=50) == [1.0, 1.0]
