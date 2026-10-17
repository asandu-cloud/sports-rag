"""A3.7 stage 3: 1X2 and goals O/U 2.5 publish the market's probability, Spix's weight at zero (D21, D26)."""
import statistics
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from core import line_selection, market_service  # noqa: E402
from prob_models import remove_vig_two_way  # noqa: E402

BOOKS = {"A": (1.90, 1.95), "B": (2.10, 1.75), "C": (2.00, 1.85)}


def totals_options(extra_lines=True):
    options = [{"side": side, "point": 2.5, "odds": odds, "bookmaker": book, "market_key": "totals"}
               for book, prices in BOOKS.items() for side, odds in zip(("over", "under"), prices)]
    if extra_lines:
        options += [{"side": "over", "point": 3.5, "odds": 3.1, "bookmaker": "A", "market_key": "totals"},
                    {"side": "under", "point": 3.5, "odds": 1.38, "bookmaker": "A", "market_key": "totals"}]
    return options


def market_over():
    return statistics.median(remove_vig_two_way(o, u)[0] for o, u in BOOKS.values())


def test_anchored_line_is_priced_at_the_market_and_keeps_spixs_probability(monkeypatch):
    monkeypatch.delenv("SPIX_DEVIG_METHOD", raising=False)
    plain = line_selection.select_best_total_recommendation(totals_options(), 3.4)
    anchored = line_selection.select_best_total_recommendation(totals_options(), 3.4, anchor_points=(2.5,))
    spix = {(r["side"], r["point"], r["bookmaker"]): r["_model_prob"] for r in plain["all_lines"]}
    for row in anchored["all_lines"]:
        key = (row["side"], row["point"], row["bookmaker"])
        if row["point"] == 2.5:
            expected = market_over() if row["side"] == "over" else 1 - market_over()
            assert row["_model_prob"] == pytest.approx(expected)
            assert row["_spix_prob"] == spix[key] and row["_probability_source"] == "market_consensus"
            assert row["_probability_basis"] == "asian_equivalent_non_push"
            assert row["_ev"] == pytest.approx(expected * row["odds"] - 1)
        else:  # other lines keep Spix's probability (strict scope)
            assert row["_model_prob"] == spix[key] and "_spix_prob" not in row
    record = anchored["market_anchor"]
    assert record["scope"] == "strict" and record["method"] == "proportional_devig_median"
    assert record["lines"]["2.5"]["books"] == 3 and record["lines"]["2.5"]["market_over"] == pytest.approx(market_over())
    assert "market_anchor" not in plain


def test_only_half_lines_can_be_anchored():
    result = line_selection.select_best_total_recommendation(totals_options(), 3.4, anchor_points=(3.25,))
    assert result["market_anchor"]["lines"] == {}
    assert all("_spix_prob" not in r for r in result["all_lines"])


def test_a_pick_at_the_anchor_needs_a_price_better_than_the_market():
    # Every book equals the consensus: no value edge, so the anchored line cannot be recommended.
    options = [{"side": s, "point": 2.5, "odds": o, "bookmaker": b, "market_key": "totals"}
               for b in ("A", "B", "C") for s, o in (("over", 1.95), ("under", 1.95))]
    result = line_selection.select_best_total_recommendation(options, 4.5, anchor_points=(2.5,))
    assert result["bet_recommendation"] is None


def moneyline_options():
    prices = {"A": (1.90, 3.60, 4.20), "B": (2.00, 3.40, 4.00), "C": (1.95, 3.50, 4.40)}
    return [{"side": side, "odds": odds, "bookmaker": book, "market_key": "h2h"}
            for book, row in prices.items() for side, odds in zip(("home", "draw", "away"), row)]


def test_1x2_is_priced_at_the_rescaled_market_median():
    plain = line_selection.choose_best_moneyline_side(0.60, 0.22, 0.18, moneyline_options(), "Lyon", "Lens")
    anchored = line_selection.choose_best_moneyline_side(0.60, 0.22, 0.18, moneyline_options(), "Lyon", "Lens", anchor=True)
    market = anchored["market_anchor"]["market"]
    assert sum(market.values()) == pytest.approx(1.0) and anchored["market_anchor"]["books"] == 3
    for entry in anchored["all_sides"]:
        assert entry["model_prob"] == pytest.approx(market[entry["side"]])
        assert entry["spix_prob"] == {"home": 0.60, "draw": 0.22, "away": 0.18}[entry["side"]]
        assert entry["probability_source"] == "market_consensus"
    # Spix's 60% home made the home side look like value; at the market's number it is not.
    assert plain["bet_recommendation"] is not None and anchored["bet_recommendation"] is None
    assert "market_anchor" not in plain


class _Anchored:
    @staticmethod
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
                mock.patch.object(market_service, "projected_moneyline_probs", return_value=(0.6, 0.22, 0.18, 1.7, 1.4)):
            return market_service.evaluate_market(_event(), "EPL", market)


def test_the_service_anchors_goals_2_5_and_1x2_only_when_switched_on(monkeypatch):
    for market in ("goals", "moneyline"):
        off = _Anchored.evaluate(monkeypatch, market, None)
        on = _Anchored.evaluate(monkeypatch, market, "strict")
        assert "market_anchor" not in off.context
        assert on.context["market_anchor"]["scope"] == "strict"
        assert on.context["engine_switches"]["SPIX_MARKET_ANCHOR"] == "strict"
    on = _Anchored.evaluate(monkeypatch, "goals", "strict")
    assert on.context["market_anchor"]["lines"]["2.5"]["books"] == 1
