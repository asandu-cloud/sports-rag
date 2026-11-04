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


# --- Task 2: the recommendations.v1 block -------------------------------------------------------------------------

def whole_line_goals_result():
    options = [{"market_key": "totals", "side": side, "point": 5.0, "odds": odds, "bookmaker": book}
               for book, pair in {"A": (2.85, 1.40), "B": (2.70, 1.45)}.items()
               for side, odds in zip(("over", "under"), pair)]
    return result("goals", options)


def btts_result():
    options = [{"market_key": "btts", "side": side, "point": None, "odds": odds, "bookmaker": book}
               for book, pair in {"A": (1.55, 2.40), "B": (1.60, 2.30)}.items()
               for side, odds in zip(("yes", "no"), pair)]
    return result("btts", options)


def corners_result(projected=10.2):
    options = [{"market_key": "totals_corners_over_under", "side": side, "point": 9.5, "odds": odds, "bookmaker": book}
               for book, pair in {"A": (1.70, 2.05), "B": (1.75, 2.00)}.items()
               for side, odds in zip(("over", "under"), pair)]
    return result("corners", options, projection={"value": projected, "components": {}})


def all_markets_results():
    cards = result("cards", [])
    return [moneyline_result()[0], goals_result(), btts_result(), spreads_result(), corners_result(), cards]


def results_with_goal_model(home, away):
    goals = goals_result()
    goals["context"]["market_anchor"] = {"goal_model": {"version": "m1", "home": home, "away": away, "error": 0.0}}
    return [goals, btts_result()]


METRICS = {"projected_total_goals": 3.31, "team_goals": {"home": 1.44, "away": 1.87}, "result_probabilities": None}


def test_value_shows_worth_it_at_the_minimum_odds_even_on_a_whole_line():
    sel = {"position": 1, "market": {"group": "goals"}, "pick": "Over 5", "published_recommendation_id": 9,
           "quote": {"side": "over", "line": 5.0}}
    value = rec.block([whole_line_goals_result()], selections=[sel], metrics={},
                      prices_as_of="2026-11-03T11:13:27")["value"]
    chance = fp.fair_outcome([whole_line_goals_result()], "goals", "over", 5.0)["fair_probability"]
    assert value == [{"position": 1, "market": "goals", "side": "over", "line": 5.0, "label": "Over 5 goals",
                      "pick": "Over 5", "chance": pytest.approx(chance), "worth_it_at": fp.minimum_odds(chance),
                      "published_recommendation_id": 9}]


def test_value_reads_a_stored_selections_quote_and_keeps_at_most_two():
    stored = [{"position": p, "market": {"group": "goals"}, "pick": "Over 2.5", "published_recommendation_id": None,
               "data": {"decision": {"quote": {"side": "over", "line": 2.5}}}} for p in (0, 1, 2)]
    value = rec.block([goals_result()], selections=stored, metrics={}, prices_as_of=None)["value"]
    assert [v["position"] for v in value] == [0, 1]
    assert value[0]["worth_it_at"] == fp.minimum_odds(fp.fair_outcome([goals_result()], "goals", "over", 2.5)["fair_probability"])


def test_value_leaves_out_picks_from_markets_outside_the_layer():
    # D61 (b): a published cards or shots-on-target pick stays on the card but not in the Value list.
    picks = [{"position": 0, "market": {"group": "cards"}, "pick": "Under 4.5", "quote": {"side": "under", "line": 4.5}},
             {"position": 1, "market": {"group": "sot"}, "pick": "Over 8.5", "quote": {"side": "over", "line": 8.5}},
             {"position": 2, "market": {"group": "goals"}, "pick": "Over 2.5", "quote": {"side": "over", "line": 2.5}}]
    value = rec.block([goals_result()], selections=picks, metrics={}, prices_as_of=None)["value"]
    assert [(v["position"], v["market"]) for v in value] == [(2, "goals")]


def test_a_value_pick_keeps_the_chance_it_was_picked_with_so_worth_it_is_not_above_its_price():
    # D68 (owner, 4 Nov): Lens v Como, Under 3.5 goals, picked at 1.65 with 62.07%; the fair card's 61.5% gave
    # "Worth it at 1.66", above the price Spix found.
    sel = {"position": 1, "market": {"group": "goals"}, "pick": "Over 2.5", "quote_odds": 1.95,
           "model_probability": 0.5231, "quote": {"side": "over", "line": 2.5}}
    value = rec.block([goals_result()], selections=[sel], metrics={}, prices_as_of=None)["value"]
    assert fp.fair_outcome([goals_result()], "goals", "over", 2.5)["fair_probability"] != pytest.approx(0.5231)
    assert value[0]["chance"] == 0.5231 and value[0]["worth_it_at"] == fp.minimum_odds(0.5231) == 1.95
    for unusable in (None, 0, 1.2, "0.52"):
        fallback = rec.block([goals_result()], selections=[dict(sel, model_probability=unusable)], metrics={},
                             prices_as_of=None)["value"]
        assert fallback[0]["chance"] == pytest.approx(
            fp.fair_outcome([goals_result()], "goals", "over", 2.5)["fair_probability"])


def test_a_value_pick_with_no_stored_prices_has_no_worth_it_at():
    sel = {"position": 0, "market": {"group": "goals"}, "pick": "Over 2.5", "quote": {"side": "over", "line": 2.5}}
    value = rec.block([result("goals", [])], selections=[sel], metrics={}, prices_as_of=None)["value"]
    assert value[0]["chance"] is None and value[0]["worth_it_at"] is None


def test_a_handicap_value_pick_finds_its_chance_from_the_team_in_the_quotes_side():
    # Stored handicap quotes name the team in "side" and never set "team" (all 3,125 on 4 Nov), e.g. Valencia's
    # "Real Madrid -1.5"; with no chance of its own, the pick's chance is its Explore option's.
    sel = {"position": 0, "market": {"group": "spreads"}, "pick": "Lens 0.5",
           "data": {"decision": {"quote": {"side": "Lens", "line": 0.5}}}}
    value = rec.block([spreads_result()], selections=[sel], metrics={}, prices_as_of=None)["value"]
    away = next(o for o in rec.options([spreads_result()])["spreads"] if o["side"] == "away" and o["line"] == 0.5)
    assert value[0]["chance"] == pytest.approx(away["chance"])
    assert value[0]["worth_it_at"] == fp.minimum_odds(away["chance"])


def test_each_value_item_names_its_option_in_the_explore_lists_words():
    # recommendations.v1, added 4 Nov: side, line and label as the options have them, so the front end matches a
    # Value pick to its option without reading the engine's wording ("Away", "Lens 0.5").
    results = [spreads_result(), moneyline_result()[0], goals_result()]
    picks = [{"position": 0, "market": {"group": "spreads"}, "pick": "Lens 0.5", "quote": {"side": "Lens", "line": 0.5}},
             {"position": 1, "market": {"group": "moneyline"}, "pick": "Away", "quote": {"side": "away", "line": None}}]
    b = rec.block(results, selections=picks, metrics={}, prices_as_of=None)
    assert [(v["side"], v["line"], v["label"]) for v in b["value"]] == [("away", 0.5, "Lens +0.5"), ("away", None, "Lens to win")]
    explore = {(o["market"], o["side"], o["line"]): o["label"] for m in b["markets"] for o in m["options"]}
    assert all(explore[(v["market"], v["side"], v["line"])] == v["label"] for v in b["value"])
    goals = {"position": 0, "market": {"group": "goals"}, "pick": "Under 3.5", "quote": {"side": "under", "line": 3.5}}
    assert rec.block(results, selections=[goals], metrics={}, prices_as_of=None)["value"][0]["label"] == "Under 3.5 goals"


def test_a_handicap_pick_on_a_team_the_read_does_not_name_keeps_the_engines_words():
    sel = {"position": 0, "market": {"group": "spreads"}, "pick": "Lens B 0.5", "quote": {"side": "Lens B", "line": 0.5}}
    value = rec.block([spreads_result()], selections=[sel], metrics={}, prices_as_of=None)["value"][0]
    assert (value["side"], value["line"], value["label"], value["chance"]) == (None, 0.5, "Lens B 0.5", None)


def test_every_market_is_listed_with_its_suggestion_or_none_and_cards_are_absent():
    b = rec.block(all_markets_results(), selections=[], metrics=METRICS, prices_as_of=None)
    assert b["version"] == "recommendations.v1"
    assert [m["market"] for m in b["markets"]] == list(rec.MARKETS)
    assert "cards" not in {m["market"] for m in b["markets"]}
    by_market = {m["market"]: m for m in b["markets"]}
    assert by_market["btts"]["suggestion"]["label"] == "Both teams to score"
    assert by_market["moneyline"]["suggestion"] is None  # nobody is above half


def test_a_read_with_no_prices_gives_an_empty_block():
    b = rec.block([result("goals", [])], selections=[], metrics={}, prices_as_of=None)
    assert b["most_likely"] == [] and b["value"] == [] and b["score_grid"] is None
    assert all(m["options"] == [] and m["suggestion"] is None for m in b["markets"])
    assert [m["market"] for m in b["markets"]] == list(rec.MARKETS)


def test_most_likely_carries_a_why_from_the_reads_figures():
    b = rec.block(all_markets_results(), selections=[], metrics=METRICS, prices_as_of="2026-11-03T11:13:27")
    why = {o["market"]: o["why"] for o in b["most_likely"]}
    assert why["btts"] == "Spix expects Lyon about 1.4 goals and Lens about 1.9."
    assert why["corners"] == "Spix expects about 10.2 corners."
    assert b["prices_as_of"] == "2026-11-03T11:13:27"
    bare = {o["market"]: o["why"] for o in rec.block(all_markets_results(), selections=[], metrics={},
                                                     prices_as_of=None)["most_likely"]}
    assert bare["btts"] is None and bare["goals"] is None  # a missing figure gives no sentence
    assert bare["corners"] == "Spix expects about 10.2 corners."  # from the corners result itself


def test_why_templates_per_market():
    assert rec.why({"market": "goals"}, METRICS, {}) == "Spix expects about 3.3 goals."
    assert rec.why({"market": "moneyline"}, METRICS, {"home_team": "Lyon", "away_team": "Lens"}) == \
        "Spix expects Lyon about 1.4 goals and Lens about 1.9."
    assert rec.why({"market": "goals"}, {}, {}) is None
    assert rec.why({"market": "corners"}, METRICS, {}, corners=None) is None


def test_score_grid_comes_from_the_stored_goal_model_and_is_null_without_one():
    assert rec.block(results_with_goal_model(1.44, 2.32), selections=[], metrics={},
                     prices_as_of=None)["score_grid"] == {"home": 1.44, "away": 2.32}
    assert rec.block([goals_result()], selections=[], metrics={}, prices_as_of=None)["score_grid"] is None


def test_the_switch_is_off_unless_set_to_on(monkeypatch):
    monkeypatch.delenv("SPIX_RECOMMENDATIONS", raising=False)
    assert not rec.enabled()
    monkeypatch.setenv("SPIX_RECOMMENDATIONS", " On ")
    assert rec.enabled()
    monkeypatch.setenv("SPIX_RECOMMENDATIONS", "off")
    assert not rec.enabled()


def test_most_likely_never_lists_the_same_bet_twice_under_two_markets():
    # Lask Linz v Slovan, 3 Nov: "Lask Linz −0.5" and "Lask Linz to win" win on exactly the same scorelines
    handicap = mk("spreads", "home", -0.5, 0.73, 1.33)
    win = mk("moneyline", "home", None, 0.73, 1.35)
    over = mk("goals", "over", 2.5, 0.73, 1.33)
    assert [p["market"] for p in rec.most_likely([handicap, win, over])] == ["moneyline", "goals"]
    assert rec.same_bet(mk("spreads", "away", 0.5), mk("moneyline", "home", None)) is False
    assert rec.same_bet(handicap, win) and not rec.same_bet(win, over)
