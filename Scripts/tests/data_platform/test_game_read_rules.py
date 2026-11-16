"""Game-read bench, task 4: today's Most likely and Value rules, and the engine's (D77)."""
from __future__ import annotations

from dataclasses import replace
from datetime import date
from itertools import combinations

import pytest

from Scripts.data_platform import recommendations
from Scripts.data_platform.game_read import pool as pools
from Scripts.data_platform.game_read import rules
from Scripts.data_platform.game_read.history import Match, Prices


def opt(market, side, line, chance, price):
    return {"market": market, "side": side, "line": line, "chance": chance, "typical_price": price, "real": False}


POOL = [
    opt("moneyline", "home", None, 0.58, 1.65), opt("moneyline", "draw", None, 0.24, 3.90),
    opt("moneyline", "away", None, 0.18, 5.20),
    opt("goals", "over", 1.5, 0.78, 1.30), opt("goals", "under", 1.5, 0.22, 4.10),
    opt("goals", "over", 3.5, 0.29, 3.20), opt("goals", "under", 3.5, 0.71, 1.38),
    opt("btts", "yes", None, 0.52, 1.80), opt("btts", "no", None, 0.48, 1.95),
    opt("spreads", "home", -0.5, 0.58, 1.66), opt("spreads", "away", 0.5, 0.42, 2.25),
    opt("spreads", "home", 1.5, 0.90, 1.08), opt("spreads", "away", -1.5, 0.10, 8.00),
]


def key(option):
    return pools.option_key(option)


def market_engine(options):
    return {key(o): o["chance"] for o in options}


def test_todays_rule_is_the_live_rule():
    markets = {}
    for option in POOL:
        markets.setdefault(option["market"], []).append(option)
    live = recommendations.most_likely([recommendations.suggestion(options) for options in markets.values()])
    assert rules.todays_most_likely(POOL) == live
    # handicap −0.5 (58%, 1.66) outranks the home win (58%, 1.65) on price, and the two are the same bet
    assert [key(o) for o in live] == [("goals", "over", 1.5), ("spreads", "home", -0.5), ("btts", "yes", None)]


def test_with_the_market_as_engine_the_lists_are_todays():
    assert rules.engine_most_likely(POOL, market_engine(POOL), strength=rules.by_edge) == rules.todays_most_likely(POOL)


def test_a_backed_option_beats_a_likelier_one_in_its_market():
    engine = {**market_engine(POOL), ("goals", "under", 3.5): 0.75}
    picks = rules.engine_most_likely(POOL, engine, strength=rules.by_edge)
    goals = [o for o in picks if o["market"] == "goals"]
    assert [key(o) for o in goals] == [("goals", "under", 3.5)]
    assert key(picks[0]) == ("goals", "under", 3.5)  # backed suggestions lead the list


CONFLICT = [
    opt("goals", "over", 1.5, 0.48, 2.00), opt("goals", "under", 1.5, 0.52, 1.85),
    opt("btts", "yes", None, 0.60, 1.55), opt("btts", "no", None, 0.40, 2.40),
    opt("moneyline", "home", None, 0.55, 1.80), opt("moneyline", "draw", None, 0.25, 3.60),
    opt("moneyline", "away", None, 0.20, 4.50),
    opt("spreads", "home", -0.5, 0.55, 1.82), opt("spreads", "away", 0.5, 0.45, 2.15),
    opt("spreads", "home", 0.5, 0.75, 1.32), opt("spreads", "away", -0.5, 0.25, 3.90),
]


def test_the_list_never_holds_more_than_three_or_options_that_cannot_win_together():
    engine = {key(o): o["chance"] + 0.05 for o in CONFLICT}  # every option backed
    engine[("goals", "under", 1.5)] = 0.70  # backed strongest; both teams to score cannot win with it
    for strength in (rules.by_edge, lambda option, chance: chance):
        picks = rules.engine_most_likely(CONFLICT, engine, strength=strength)
        assert 1 <= len(picks) <= 3
        assert len({o["market"] for o in picks}) == len(picks)
        for a, b in combinations(picks, 2):
            assert recommendations.can_win_together(a, b) and not recommendations.same_bet(a, b)
        assert ("btts", "yes", None) not in {key(o) for o in picks}
    picks = rules.engine_most_likely(CONFLICT, engine, strength=rules.by_edge)
    assert [key(o) for o in picks] == [("goals", "under", 1.5), ("spreads", "home", 0.5), ("moneyline", "home", None)]
    engine[("spreads", "home", -0.5)] = 0.80  # now the handicap −0.5 leads, and the home win is the same bet
    picks = rules.engine_most_likely(CONFLICT, engine, strength=rules.by_edge)
    assert key(picks[0]) == ("spreads", "home", -0.5) and ("moneyline", "home", None) not in {key(o) for o in picks}


def test_unscored_options_count_as_not_backed():
    engine = {("goals", "under", 3.5): 0.75}  # the engine scores one option only
    picks = rules.engine_most_likely(POOL, engine, strength=rules.by_edge)
    assert [key(o) for o in picks] == [("goals", "under", 3.5)] + [key(o) for o in rules.todays_most_likely(POOL)
                                                                   if o["market"] != "goals"][:2]
    assert rules.engine_most_likely(POOL, {("btts", "no", None): 0.70}, strength=rules.by_edge) == \
        rules.todays_most_likely(POOL)  # backed, but under the 50% floor: floors stay as live
    assert rules.engine_most_likely(POOL, {}, strength=rules.by_edge) == rules.todays_most_likely(POOL)


def test_by_blend_ranks_by_the_blended_chance():
    model = {"intercepts": [0.0, 0.0], "b": 1.0, "c": 0.0}  # the market alone
    assert rules.by_blend(model)(opt("goals", "over", 2.5, 0.6, 1.6), 0.9) == pytest.approx(0.6)
    weighted = {"intercepts": [0.0, 0.0], "b": 0.5, "c": 0.5}
    assert 0.6 < rules.by_blend(weighted)(opt("goals", "over", 2.5, 0.6, 1.6), 0.9) < 0.9
    assert rules.by_edge(opt("goals", "over", 2.5, 0.6, 1.6), 0.65) == pytest.approx(0.05)


def match(prices):
    return Match(id=1, competition="EPL", country="England", season=2023, day=date(2023, 9, 2), home="England:A",
                 away="England:B", home_goals=1, away_goals=1, pre=prices, close=prices)


PRE = Prices(result_avg=(2.10, 3.40, 3.60), result_max=(2.25, 3.60, 3.90), handicap_line=-0.5,
             handicap_avg=(2.08, 1.82), handicap_max=(2.15, 1.88), over25_avg=(1.95, 1.90), over25_max=(2.02, 1.98))


@pytest.mark.parametrize("prices", [
    PRE,
    Prices(result_avg=(1.25, 6.00, 11.0), handicap_line=-1.5, handicap_avg=(1.90, 1.95), over25_avg=(1.55, 2.45)),
    Prices(result_avg=(4.80, 3.75, 1.72), handicap_line=0.5, handicap_avg=(2.40, 1.60), over25_avg=(2.30, 1.62)),
])
def test_the_market_makes_no_value_pick_at_the_average_price(prices):
    candidates = rules.value_candidates(match(prices), price="avg")
    assert len(candidates) == 7
    assert all(c["chance"] * c["price"] < 1 for c in candidates)
    assert rules.todays_value(candidates) == []
    assert rules.engine_value(candidates, {key(c): c["chance"] for c in candidates}) == []


def test_value_candidates_take_the_asked_price():
    best = {key(c): c["price"] for c in rules.value_candidates(match(PRE), price="max")}
    assert best[("moneyline", "home", None)] == 2.25 and best[("goals", "over", 2.5)] == 2.02
    assert best[("spreads", "away", 0.5)] == 1.88


def test_value_needs_two_percent_and_keeps_two_a_match():
    def cand(market, side, line, chance, price):
        return {"market": market, "side": side, "line": line, "chance": chance, "price": price}
    candidates = [cand("moneyline", "home", None, 0.50, 2.03),   # 1.5%: under the bar
                  cand("goals", "over", 2.5, 0.54, 2.00),         # 8%
                  cand("goals", "under", 2.5, 0.50, 2.10),        # 5%, cannot win with over 2.5
                  cand("moneyline", "draw", None, 0.30, 3.45),    # 3.5%
                  cand("spreads", "home", -0.5, 0.40, 2.60)]      # 4%
    picks = rules.todays_value(candidates)
    assert [(key(p), round(p["value"], 3)) for p in picks] == [(("goals", "over", 2.5), 0.08),
                                                              (("spreads", "home", -0.5), 0.04)]
    engine = {("moneyline", "draw", None): 0.33, ("goals", "under", 2.5): 0.40}  # the engine's own chances
    picks = rules.engine_value(candidates, engine)
    assert [key(p) for p in picks] == [("moneyline", "draw", None)]  # 13.85%; under 2.5 at 0.40 is no value
    assert picks[0]["engine_chance"] == 0.33


def test_quarter_main_handicaps_give_no_value_candidates():
    quarter = replace(PRE, handicap_line=-0.75, handicap_avg=(2.20, 1.72), handicap_max=(2.30, 1.78))
    candidates = rules.value_candidates(match(quarter), price="avg")
    assert not any(c["market"] == "spreads" for c in candidates) and len(candidates) == 5
    assert rules.value_candidates(match(Prices()), price="max") == []
