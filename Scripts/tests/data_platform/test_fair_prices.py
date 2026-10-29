"""Fair-price card numbers and Price Check (D36): the market's fair price per main line, from a read's stored prices."""
import statistics

import pytest

from data_platform import fair_prices as fp
from data_platform.devig import power

HOME, AWAY = "Lyon", "Lens"


def option(key, side, point, odds, book, team=None):
    return {"market_key": key, "side": side, "team": team, "point": point, "odds": odds, "bookmaker": book}


def result(group, options, *, projection=None, candidates=(), anchor=None):
    context = {"decision_audit": {"input_options": options, "candidates": list(candidates)}}
    if anchor:
        context["market_anchor"] = anchor
    return {"market": {"group": group, "key": group}, "fixture": {"home_team": HOME, "away_team": AWAY},
            "projection": projection or {"value": 0.5, "components": {}}, "decision": {"status": "no_bet"},
            "context": context}


GOALS = {"A": {2.5: (1.90, 1.95), 3.5: (3.0, 1.38)}, "B": {2.5: (2.05, 1.80), 3.5: (3.1, 1.36)},
         "C": {2.5: (2.00, 1.85)}, "D": {1.5: (1.25, 3.9)}}


def goals_result(candidates=()):
    options = [option("totals", side, point, odds, book) for book, lines in GOALS.items()
               for point, pair in lines.items() for side, odds in zip(("over", "under"), pair)]
    return result("goals", options, candidates=candidates)


def moneyline_result():
    prices = {"A": (2.6, 3.4, 2.7), "B": (2.5, 3.5, 2.8), "C": (2.55, 3.3, 2.75)}
    options = [option("h2h", side, None, odds, book) for book, row in prices.items()
               for side, odds in zip(("home", "draw", "away"), row)]
    return result("moneyline", options, projection={"value": 0.42, "components": {
        "home_probability": 0.42, "draw_probability": 0.29, "away_probability": 0.29}}), prices


def spreads_result():
    options = []
    for book, (home_odds, away_odds) in {"A": (1.95, 1.90), "B": (2.00, 1.85)}.items():
        options += [option("spreads", None, -0.5, home_odds, book, team=HOME),
                    option("spreads", None, 0.5, away_odds, book, team=AWAY)]
    options += [option("spreads", None, -1.5, 3.2, "A", team=HOME), option("spreads", None, 1.5, 1.35, "A", team=AWAY)]
    return result("spreads", options)


def line(card, market):
    return next(entry for entry in card["lines"] if entry["market"] == market)


def test_the_main_goals_line_is_the_one_most_books_price_and_its_fair_price_is_their_median():
    entry = line(fp.fair_card([goals_result()]), "goals")
    assert entry["line"] == 2.5 and entry["book_count"] == 3
    over = entry["outcomes"][0]
    expected = statistics.median(power([o, u])[0] for o, u in (GOALS[b][2.5] for b in "ABC"))
    assert over["side"] == "over" and over["fair_probability"] == pytest.approx(expected)
    assert over["fair_odds"] == pytest.approx(1 / expected)
    assert entry["outcomes"][1]["fair_probability"] == pytest.approx(1 - expected)


def test_the_minimum_odds_are_fair_odds_plus_two_percent_rounded_up():
    assert fp.minimum_odds(1 / 1.85) == 1.89  # D36: "fair 1.85; worth it at 1.89 or better"
    assert fp.minimum_odds(1 / (1.89 / 1.02)) == 1.89  # exactly on a cent stays there


def test_the_1x2_is_rescaled_to_sum_to_one():
    moneyline, prices = moneyline_result()
    entry = line(fp.fair_card([moneyline]), "moneyline")
    assert [o["side"] for o in entry["outcomes"]] == ["home", "draw", "away"]
    assert sum(o["fair_probability"] for o in entry["outcomes"]) == pytest.approx(1.0)
    raw = [statistics.median(power(list(row))[i] for row in prices.values()) for i in range(3)]
    assert entry["outcomes"][0]["fair_probability"] == pytest.approx(raw[0] / sum(raw))


def test_the_main_handicap_is_from_the_home_sides_view():
    entry = line(fp.fair_card([spreads_result()]), "spreads")
    assert entry["line"] == -0.5 and [o["side"] for o in entry["outcomes"]] == ["lyon", "lens"]
    assert [o["line"] for o in entry["outcomes"]] == [-0.5, 0.5]


def test_a_line_one_bookmaker_prices_is_not_on_the_card():
    only_one = result("btts", [option("btts", "yes", None, 1.8, "A"), option("btts", "no", None, 2.0, "A")])
    assert [entry["market"] for entry in fp.fair_card([only_one])["lines"]] == []


def test_spixs_own_take_is_shown_beside_and_labelled():
    candidates = [{"option": {"side": "over", "point": 2.5, "_model_prob": 0.57, "_spix_prob": 0.61}},
                  {"option": {"side": "under", "point": 2.5, "_model_prob": 0.43}}]
    entry = line(fp.fair_card([goals_result(candidates)]), "goals")
    assert [o["spix_probability"] for o in entry["outcomes"]] == [0.61, 0.43]
    moneyline, _ = moneyline_result()
    assert [o["spix_probability"] for o in line(fp.fair_card([moneyline]), "moneyline")["outcomes"]] == [0.42, 0.29, 0.29]


def test_any_priced_line_has_a_fair_price_for_price_check():
    outcome = fp.fair_outcome([goals_result()], "goals", "over", 3.5)
    expected = statistics.median(power(list(GOALS[b][3.5]))[0] for b in "AB")
    assert outcome["fair_probability"] == pytest.approx(expected) and outcome["book_count"] == 2
    assert fp.fair_outcome([goals_result()], "goals", "over", 4.5) is None


@pytest.mark.parametrize("odds, verdict", [(1.89, "good_price"), (1.87, "fair"), (1.82, "fair"),
                                           (1.81, "overpaying")])
def test_price_check_says_good_price_fair_or_overpaying(odds, verdict):
    checked = fp.price_check(1 / 1.85, odds)
    assert checked["verdict"] == verdict and checked["value"] == pytest.approx(odds / 1.85 - 1)
    assert checked["minimum_odds"] == 1.89


def test_the_match_read_card_carries_the_fair_prices():
    from data_platform.services.match_read_cards import _fair_prices_card
    card = _fair_prices_card([goals_result(), moneyline_result()[0]])
    assert [entry["market"] for entry in card["lines"]] == ["moneyline", "goals"]
    assert _fair_prices_card(None)["lines"] == []
