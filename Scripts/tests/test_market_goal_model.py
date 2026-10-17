"""A3.7 stage 4: every goal market priced from the market's own goal model (M1, D29)."""
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from core import line_selection, market_goal_model as mgm, market_service  # noqa: E402
from data_platform import market_goals  # noqa: E402

HOME, AWAY = 1.6, 1.1


def fair_odds(p):
    return 1 / p


def market_options(home=HOME, away=AWAY, books=("A", "B", "C")):
    """Zero-margin prices generated from known goal means."""
    h, d, a = market_goals.result_probabilities(home, away)
    over = market_goals.over_probability(home, away, 2.5)
    ah = market_goals.handicap_probability(home, away, -0.5)
    h2h = [{"side": s, "odds": fair_odds(p), "bookmaker": b, "market_key": "h2h"}
           for b in books for s, p in (("home", h), ("draw", d), ("away", a))]
    totals = [{"side": s, "point": 2.5, "odds": fair_odds(p), "bookmaker": b, "market_key": "totals"}
              for b in books for s, p in (("over", over), ("under", 1 - over))]
    totals += [{"side": s, "point": 3.5, "odds": o, "bookmaker": "A", "market_key": "totals"} for s, o in (("over", 3.0), ("under", 1.4))]
    spreads = [{"team": t, "point": pt, "odds": fair_odds(p), "bookmaker": b, "market_key": "spreads"}
               for b in books for t, pt, p in (("Lyon", -0.5, ah), ("Lens", 0.5, 1 - ah))]
    spreads += [{"team": "Lyon", "point": -1.5, "odds": 3.2, "bookmaker": "A", "market_key": "spreads"},
                {"team": "Lens", "point": 1.5, "odds": 1.35, "bookmaker": "A", "market_key": "spreads"}]
    return h2h, totals, spreads


def test_m1_recovers_the_goal_means_behind_the_prices():
    fit = mgm.solve(*market_options(), "Lyon", "Lens")
    assert fit["home"] == pytest.approx(HOME, rel=1e-4) and fit["away"] == pytest.approx(AWAY, rel=1e-4)
    assert fit["inputs"] == 5 and fit["targets"]["handicaps"] == {"-0.5": pytest.approx(
        market_goals.handicap_probability(HOME, AWAY, -0.5))}
    assert sum(mgm.total_probabilities(fit).values()) == pytest.approx(1.0)
    assert mgm.btts_probability(fit) == pytest.approx(
        sum(p for (h, a), p in mgm.score_probabilities(fit).items() if h and a))


def test_the_main_handicap_is_the_line_most_books_price_on_both_sides():
    _, _, spreads = market_options()
    target = mgm.targets([], [], spreads, "Lyon", "Lens")
    assert list(target["handicaps"]) == [-0.5]


def test_prices_that_cannot_identify_both_teams_give_no_model():
    _, totals, _ = market_options()
    assert mgm.solve([], totals, [], "Lyon", "Lens") is None


def test_totals_are_priced_from_the_model_except_the_anchored_2_5():
    h2h, totals, spreads = market_options()
    fit = mgm.solve(h2h, totals, spreads, "Lyon", "Lens")
    counts = mgm.total_probabilities(fit)
    spix = line_selection.select_best_total_recommendation([dict(o) for o in totals], 2.2)
    result = line_selection.select_best_total_recommendation([dict(o) for o in totals], 2.2, anchor_points=(2.5,),
                                                             market_count_probabilities=counts)
    spix_by = {(r["side"], r["point"], r["bookmaker"]): r["_model_prob"] for r in spix["all_lines"]}
    for row in result["all_lines"]:
        assert row["_spix_prob"] == pytest.approx(spix_by[(row["side"], row["point"], row["bookmaker"])])
        if row["point"] == 3.5:
            over = market_goals.over_probability(fit["home"], fit["away"], 3.5)
            assert row["_model_prob"] == pytest.approx(over if row["side"] == "over" else 1 - over)
            assert row["_probability_source"] == "market_goal_model"
        else:
            assert row["_probability_source"] == "market_consensus"


def test_handicaps_use_the_model_and_its_expected_margin():
    h2h, totals, spreads = market_options()
    fit = mgm.solve(h2h, totals, spreads, "Lyon", "Lens")
    result = line_selection.select_best_spread_recommendation(
        [dict(o) for o in spreads], -0.4, "Lyon", away_team="Lens", score_probs={},
        market_score_probs=mgm.score_probabilities(fit), market_diff=fit["home"] - fit["away"])
    for row in result["all_lines"]:
        home_line = row["point"] if row["team"] == "Lyon" else -row["point"]
        p_home = market_goals.handicap_probability(fit["home"], fit["away"], home_line)
        assert row["_model_prob"] == pytest.approx(p_home if row["team"] == "Lyon" else 1 - p_home, abs=1e-9)
        assert row["_probability_source"] == "market_goal_model" and "_spix_prob" in row
        sign = 1 if row["team"] == "Lyon" else -1
        assert row["_goal_edge"] == pytest.approx(sign * (fit["home"] - fit["away"]) + row["point"])


def test_btts_keeps_spixs_probability_beside_the_models():
    options = [{"side": s, "odds": o, "bookmaker": "A", "market_key": "btts"} for s, o in (("Yes", 1.8), ("No", 2.0))]
    result = line_selection.select_best_btts_recommendation(options, 0.52, spix_p_yes=0.61)
    rows = {r["side"].lower(): r for r in result["all_sides"]}
    assert rows["yes"]["_model_prob"] == 0.52 and rows["yes"]["_spix_prob"] == 0.61
    assert rows["no"]["_spix_prob"] == pytest.approx(0.39) and rows["no"]["_probability_source"] == "market_goal_model"


def evaluate(monkeypatch, market, scope):
    from Scripts.tests.test_market_service import _event
    if scope:
        monkeypatch.setenv("SPIX_MARKET_ANCHOR", scope)
    else:
        monkeypatch.delenv("SPIX_MARKET_ANCHOR", raising=False)
    audit = {"temporal_status": "fixture_rows_strictly_before_target_date",
             "current_season_matches": 12, "effective_sample_size": 12}
    from prob_models import dixon_coles_scoreline_matrix
    matrix = dixon_coles_scoreline_matrix(1.7, 1.4)
    scores = {(h, a): p for h, row in enumerate(matrix) for a, p in enumerate(row)}
    with mock.patch.object(market_service, "get_prediction_profile_context", return_value=({"goals_for_pm": 1.6}, audit)), \
            mock.patch.object(market_service, "projected_total_goals", return_value=(3.1, 3.0, 3.2)), \
            mock.patch.object(market_service, "projected_correct_score_probs", return_value=(scores, 1.7, 1.4)), \
            mock.patch.object(market_service, "projected_btts_prob", return_value=(0.6, 1.7, 1.4, None, None)), \
            mock.patch.object(market_service, "projected_goal_difference", return_value=(0.3, 0.3, 0.3)), \
            mock.patch.object(market_service, "projected_moneyline_probs", return_value=(0.5, 0.25, 0.25, 1.7, 1.4)):
        return market_service.evaluate_market(_event(), "EPL", market)


@pytest.mark.parametrize("market", ["goals", "btts", "spreads"])
def test_the_service_records_the_goal_model_only_at_stage_4(monkeypatch, market):
    assert "market_anchor" not in evaluate(monkeypatch, market, None).context
    strict = evaluate(monkeypatch, market, "strict").context.get("market_anchor")
    assert strict is None or "goal_model" not in strict
    record = evaluate(monkeypatch, market, "goals").context["market_anchor"]
    assert record["scope"] == "goals" and "goal_model" in record
    if record["goal_model"] is not None:
        assert record["goal_model"]["version"] == "market-goal-model.v1"
