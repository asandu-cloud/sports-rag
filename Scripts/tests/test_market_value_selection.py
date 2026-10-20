"""Stage 4 bundle (D33 (b), D35): select market-priced lines by value against the market, at allowed bookmakers."""
import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from core import bookmaker_sets, line_selection  # noqa: E402
from prob_models import remove_vig_two_way  # noqa: E402
from data_platform.services.match_read_compiler import _Candidate, _market_value_key, _ranking_key  # noqa: E402


@pytest.fixture
def market_value(monkeypatch):
    monkeypatch.setenv("SPIX_SELECTION_POLICY", "market_value")
    monkeypatch.setenv("SPIX_BOOKMAKER_SET", "romania")
    monkeypatch.delenv("SPIX_DEVIG_METHOD", raising=False)


def test_romania_allows_betano_and_superbet_only(monkeypatch):
    monkeypatch.setenv("SPIX_BOOKMAKER_SET", "romania")
    assert bookmaker_sets.allowed("Betano") and bookmaker_sets.allowed(" superbet ")
    assert not bookmaker_sets.allowed("Pinnacle") and not bookmaker_sets.allowed("Bet365")
    monkeypatch.delenv("SPIX_BOOKMAKER_SET")
    assert bookmaker_sets.allowed("Pinnacle")


def corner_options():
    # Pinnacle and Bet365 set the consensus; Betano is generous on the under 10.5.
    rows = {"Pinnacle": (2.02, 1.86), "Bet365": (2.00, 1.83), "Betano": (1.80, 2.05)}
    return [{"side": side, "point": 10.5, "odds": odds, "bookmaker": book, "market_key": "totals_corners_over_under"}
            for book, prices in rows.items() for side, odds in zip(("over", "under"), prices)]


def test_corners_at_the_line_consensus_picked_by_value_at_an_allowed_book(market_value):
    result = line_selection.select_best_total_recommendation(corner_options(), 12.0, 20.0, anchor_points="all")
    pick = result["bet_recommendation"]
    under = 1 - statistics.median(remove_vig_two_way(o, u)[0] for o, u in ((2.02, 1.86), (2.00, 1.83), (1.80, 2.05)))
    assert pick["bookmaker"] == "Betano" and pick["side"] == "under"
    assert pick["_model_prob"] == pytest.approx(under) and pick["_probability_source"] == "market_consensus"
    assert pick["_ev"] == pytest.approx(under * 2.05 - 1) and pick["_ev"] >= 0.02
    # Spix's projection of 12 corners would never pick the under; the market's number does.
    assert pick["_spix_prob"] < 0.5


def test_no_pick_when_the_value_is_only_at_a_book_members_cannot_use(market_value):
    options = [o for o in corner_options() if o["bookmaker"] != "Betano"]
    options += [{"side": "under", "point": 10.5, "odds": 2.05, "bookmaker": "1xBet", "market_key": "totals_corners_over_under"}]
    result = line_selection.select_best_total_recommendation(options, 12.0, 20.0, anchor_points="all")
    assert result["bet_recommendation"] is None


def test_legacy_policy_is_unchanged_without_a_set(monkeypatch):
    monkeypatch.delenv("SPIX_SELECTION_POLICY", raising=False)
    monkeypatch.delenv("SPIX_BOOKMAKER_SET", raising=False)
    plain = line_selection.select_best_total_recommendation(corner_options(), 12.0, 20.0)
    again = line_selection.select_best_total_recommendation(corner_options(), 12.0, 20.0)
    assert plain["bet_recommendation"] == again["bet_recommendation"]
    assert all("_probability_source" not in r for r in plain["all_lines"])


def test_btts_market_value_ignores_spixs_preferred_side(market_value):
    options = [{"side": "Yes", "odds": 1.70, "bookmaker": "Pinnacle", "market_key": "btts"},
               {"side": "No", "odds": 2.20, "bookmaker": "Pinnacle", "market_key": "btts"},
               {"side": "No", "odds": 2.40, "bookmaker": "Betano", "market_key": "btts"}]
    result = line_selection.select_best_btts_recommendation(options, 0.55, spix_p_yes=0.75)
    pick = result["bet_recommendation"]
    assert pick["bookmaker"] == "Betano" and pick["side"] == "No"
    assert pick["_ev"] == pytest.approx(0.45 * 2.40 - 1)


def test_1x2_market_value_takes_the_best_allowed_price(market_value):
    prices = {"Pinnacle": (2.00, 3.50, 4.00), "Bet365": (1.95, 3.40, 4.10), "Betano": (2.15, 3.30, 3.80)}
    options = [{"side": s, "odds": o, "bookmaker": b, "market_key": "h2h"}
               for b, row in prices.items() for s, o in zip(("home", "draw", "away"), row)]
    result = line_selection.choose_best_moneyline_side(0.30, 0.30, 0.40, options, "Lyon", "Lens", anchor=True)
    pick = result["bet_recommendation"]
    market_home = result["market_anchor"]["market"]["home"]
    assert pick["side"] == "home" and pick["bookmaker"] == "Betano"
    assert pick["ev"] == pytest.approx(market_home * 2.15 - 1)


def candidate(market, ev, priced, confidence="low", edge=0.0, index=0):
    return _Candidate(result_index=index, result={}, market_group=market, market_key=market, side="over", line=None,
                      odds=2.0, confidence=confidence, value_edge=edge, expected_value=ev, model_probability=0.5,
                      tags=frozenset(), market_priced=priced)


def test_market_priced_picks_rank_first_by_value_against_the_market():
    corners_spix = candidate("cards", 0.40, False, confidence="high", edge=0.2, index=0)
    goals = candidate("goals", 0.03, True, index=1)
    corners = candidate("corners", 0.05, True, index=2)
    assert [c.market_group for c in sorted([corners_spix, goals, corners], key=_market_value_key)] == ["corners", "goals", "cards"]
    assert sorted([corners_spix, goals, corners], key=_ranking_key)[0].market_group == "cards"  # legacy order unchanged


def score_matrix(home_rate=1.6, away_rate=1.0, top=10):
    from math import exp, factorial
    raw = {(h, a): exp(-home_rate) * home_rate ** h / factorial(h) * exp(-away_rate) * away_rate ** a / factorial(a)
           for h in range(top + 1) for a in range(top + 1)}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def handicap_options(extra=()):
    rows = [("Pinnacle", -0.5, 1.95, 1.95), ("Bet365", -0.5, 1.93, 1.97), ("Betano", -0.5, 2.10, 1.80),
            ("Betano", -1.5, 4.50, 1.22), *extra]
    out = []
    for book, home_line, home_odds, away_odds in rows:
        out += [{"team": "Lyon", "point": home_line, "odds": home_odds, "bookmaker": book, "market_key": "spreads"},
                {"team": "Lens", "point": -home_line, "odds": away_odds, "bookmaker": book, "market_key": "spreads"}]
    return out


@pytest.fixture
def market_value_all_books(monkeypatch):
    monkeypatch.setenv("SPIX_SELECTION_POLICY", "market_value")
    monkeypatch.delenv("SPIX_BOOKMAKER_SET", raising=False)
    monkeypatch.delenv("SPIX_DEVIG_METHOD", raising=False)


def spread_pick(options):
    matrix = score_matrix()
    result = line_selection.select_best_spread_recommendation(
        options, 0.6, "Lyon", "Lens", "Ligue1", score_probs=matrix, market_score_probs=matrix, market_diff=0.6)
    return result["bet_recommendation"], result["all_lines"]


def test_handicap_value_is_taken_only_on_the_main_line(market_value_all_books):
    pick, lines = spread_pick(handicap_options())
    alternate = next(o for o in lines if o["bookmaker"] == "Betano" and o["point"] == -1.5)
    assert alternate["_ev"] > pick["_ev"] and not alternate["_eligible"] and alternate["_main_line"] is False
    assert (pick["bookmaker"], pick["team"], pick["point"]) == ("Betano", "Lyon", -0.5)


def test_sbo_and_dafabet_prices_are_never_taken(market_value_all_books):
    pick, _ = spread_pick(handicap_options(extra=[("SBO", -0.5, 2.25, 1.70), ("Dafabet", -0.5, 2.20, 1.72)]))
    assert pick["bookmaker"] == "Betano"
    assert not bookmaker_sets.quotable("SBO") and bookmaker_sets.allowed("SBO")  # still in the consensus


def test_main_line_rule_needs_the_market_value_policy(monkeypatch):
    monkeypatch.delenv("SPIX_SELECTION_POLICY", raising=False)
    _, lines = spread_pick(handicap_options())
    assert all("_main_line" not in o for o in lines)
