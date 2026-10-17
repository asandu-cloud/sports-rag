"""D29 gate: M1 replayed from each audited read's stored prices, with the engine's own code."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rag_ingest"))

from data_platform import m1_gate, market_goals  # noqa: E402
from core import market_goal_model  # noqa: E402

HOME, AWAY = 1.5, 1.2


def stored_read():
    """A read whose decision audit holds zero-margin prices generated from known goal means."""
    h, d, a = market_goals.result_probabilities(HOME, AWAY)
    over = market_goals.over_probability(HOME, AWAY, 2.5)
    ah = market_goals.handicap_probability(HOME, AWAY, -0.25)
    fixture = {"home_team": "Lyon", "away_team": "Lens"}
    def res(group, options):
        return {"market": {"group": group}, "fixture": fixture,
                "context": {"decision_audit": {"input_options": options, "candidates": []}}}
    return {"canonical_results": [
        res("moneyline", [{"side": s, "odds": 1 / p, "bookmaker": "A"} for s, p in (("home", h), ("draw", d), ("away", a))]),
        res("goals", [{"side": s, "point": 2.5, "odds": 1 / p, "bookmaker": "A"} for s, p in (("over", over), ("under", 1 - over))]),
        res("spreads", [{"team": t, "point": pt, "odds": 1 / p, "bookmaker": "A"}
                        for t, pt, p in (("Lyon", -0.25, ah), ("Lens", 0.25, 1 - ah))])]}


def test_m1_is_solved_again_from_the_stored_prices():
    fit = m1_gate.read_fit(stored_read())
    assert fit["home"] == pytest.approx(HOME, rel=1e-4) and fit["away"] == pytest.approx(AWAY, rel=1e-4)


def line(market, side, point):
    return {"market": market, "side": side, "point": point, "home": "Lyon", "away": "Lens"}


def test_each_market_is_priced_as_stage_4_would_price_it():
    fit = m1_gate.read_fit(stored_read())
    h, a = fit["home"], fit["away"]
    assert m1_gate.m1_probability(fit, line("goals", "under", 3.5)) == pytest.approx(1 - market_goals.over_probability(h, a, 3.5))
    assert m1_gate.m1_probability(fit, line("btts", "no", None)) == pytest.approx(1 - market_goal_model.btts_probability(fit))
    p_home = market_goals.handicap_probability(h, a, -1.0)
    assert m1_gate.m1_probability(fit, line("spreads", "lens", 1.0)) == pytest.approx(1 - p_home, abs=1e-9)


def test_the_handicap_m1_was_fitted_to_is_flagged_on_both_sides():
    fit = m1_gate.read_fit(stored_read())
    assert m1_gate._fitted_handicap(fit, line("spreads", "lyon", -0.25))
    assert m1_gate._fitted_handicap(fit, line("spreads", "lens", 0.25))
    assert not m1_gate._fitted_handicap(fit, line("spreads", "lyon", -1.0))
    assert not m1_gate._fitted_handicap(fit, line("goals", "over", 3.5))


def test_log_loss_is_clipped():
    assert m1_gate.log_loss(1.0, 0.0) == pytest.approx(-__import__("math").log(1e-6))
