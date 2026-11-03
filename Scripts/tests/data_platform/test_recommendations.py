"""Recommendation layer (D52, D53): every option of a read's priced markets, Spix's suggestion per market, Most likely."""
import pytest

from data_platform import fair_prices as fp
from data_platform import recommendations as rec
from Scripts.tests.data_platform.test_fair_prices import goals_result, line as fp_line, moneyline_result, result, \
    spreads_result


def mk(market="goals", side="over", line=2.5, chance=0.5, price=1.5):
    return {"market": market, "side": side, "line": line, "chance": chance, "typical_price": price}


home_win_60 = mk("moneyline", "home", None, 0.60, 1.62)
under_1_5_goals_58 = mk("goals", "under", 1.5, 0.58, 1.66)
btts_yes_57 = mk("btts", "yes", None, 0.57, 1.70)
corners_over_55 = mk("corners", "over", 9.5, 0.55, 1.75)


def test_only_half_lines_priced_by_two_books_on_both_sides_are_options():
    # GOALS fixture of test_fair_prices: 2.5 (books A, B, C) and 3.5 (A, B) qualify; 1.5 (D only) does not
    opts = rec.options([goals_result()])["goals"]
    assert [(o["side"], o["line"]) for o in opts] == [("over", 2.5), ("under", 2.5), ("over", 3.5), ("under", 3.5)]


def test_whole_and_quarter_lines_are_not_options():
    options = [{"market_key": "totals", "side": side, "point": point, "odds": odds, "bookmaker": book}
               for book in "AB" for point in (2.0, 2.25) for side, odds in (("over", 1.9), ("under", 1.9))]
    assert rec.options([result("goals", options)]) == {"goals": []}


def test_typical_price_is_the_median_of_the_books_prices_and_chance_is_the_fair_card_chance():
    over = next(o for o in rec.options([goals_result()])["goals"] if o["side"] == "over" and o["line"] == 2.5)
    assert over["typical_price"] == 2.00  # median of 1.90, 2.05, 2.00
    assert over["chance"] == pytest.approx(fp_line(fp.fair_card([goals_result()]), "goals")["outcomes"][0]["fair_probability"])
    assert fp.typical_price(fp._books(goals_result(), "goals"), "goals", "over", 1.5) is None  # one book only


def test_labels_are_plain_words():
    opts = rec.options([goals_result(), moneyline_result()[0], spreads_result()])
    labels = {o["label"] for m in opts.values() for o in m}
    assert {"Over 2.5 goals", "Under 3.5 goals", "Lyon to win", "Draw", "Lens to win", "Lyon −0.5", "Lens +0.5"} <= labels


def test_handicap_options_are_home_then_away_each_with_its_own_line():
    opts = rec.options([spreads_result()])["spreads"]
    assert [(o["side"], o["line"], o["typical_price"]) for o in opts] == [("home", -0.5, 1.98), ("away", 0.5, 1.88)]


def test_a_long_shot_is_marked():
    options = [{"market_key": "totals", "side": side, "point": 3.5, "odds": odds, "bookmaker": book}
               for book in "AB" for side, odds in (("over", 4.5), ("under", 1.2))]
    over, under = rec.options([result("goals", options)])["goals"]
    assert over["chance"] < rec.LONG_SHOT and over["long_shot"]
    assert not under["long_shot"]


def test_a_read_with_no_prices_gives_empty_markets():
    assert rec.options([result("goals", [])]) == {"goals": []}
    assert rec.options([]) == {}


def test_wins_settles_each_goal_market_on_a_scoreline():
    assert rec.wins({"market": "goals", "side": "over", "line": 2.5}, 2, 1)
    assert not rec.wins({"market": "goals", "side": "under", "line": 2.5}, 2, 1)
    assert not rec.wins({"market": "btts", "side": "yes", "line": None}, 3, 0)
    assert rec.wins({"market": "btts", "side": "no", "line": None}, 3, 0)
    assert rec.wins({"market": "moneyline", "side": "draw", "line": None}, 1, 1)
    assert rec.wins({"market": "spreads", "side": "home", "line": -1.5}, 2, 0)
    assert not rec.wins({"market": "spreads", "side": "away", "line": 1.5}, 2, 0)
    with pytest.raises(ValueError):
        rec.wins({"market": "corners", "side": "over", "line": 9.5}, 1, 0)


def test_can_win_together_checks_scorelines_and_ignores_corners():
    draw = {"market": "moneyline", "side": "draw", "line": None}
    assert not rec.can_win_together(draw, {"market": "spreads", "side": "home", "line": -0.5})
    assert rec.can_win_together(draw, {"market": "goals", "side": "over", "line": 2.5})  # 2-2
    assert rec.can_win_together(draw, {"market": "corners", "side": "over", "line": 9.5})


def test_suggestion_needs_half_chance_and_a_price_of_1_30_and_prefers_the_higher_chance():
    opts = [mk(chance=0.92, price=1.05), mk(chance=0.71, price=1.33), mk(chance=0.55, price=1.70)]
    assert rec.suggestion(opts)["chance"] == 0.71
    assert rec.suggestion([mk(chance=0.45, price=2.1)]) is None
    assert rec.suggestion([]) is None
    assert rec.suggestion([mk(chance=0.6, price=1.4), mk(side="under", chance=0.6, price=1.5)])["typical_price"] == 1.5


def test_most_likely_keeps_three_compatible_markets_by_chance_with_price_breaking_ties():
    picks = rec.most_likely([home_win_60, under_1_5_goals_58, btts_yes_57, corners_over_55])
    # under 1.5 and BTTS yes cannot both win (needs 1-1 = 2 goals), so BTTS drops and corners takes its place
    assert [p["market"] for p in picks] == ["moneyline", "goals", "corners"]
    assert rec.most_likely([None, corners_over_55]) == [corners_over_55]
    tie = rec.most_likely([mk("btts", "yes", None, 0.6, 1.5), mk("corners", "over", 9.5, 0.6, 1.6)])
    assert [p["market"] for p in tie] == ["corners", "btts"]
