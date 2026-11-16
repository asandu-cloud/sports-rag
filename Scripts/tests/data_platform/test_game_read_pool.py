"""Game-read bench, task 3: each match's Most likely pool rebuilt from the market's prices (D77)."""
from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from Scripts.data_platform import devig, market_goals
from Scripts.data_platform.game_read import pool
from Scripts.data_platform.game_read.history import Match, Prices, market_over25

PRE = Prices(result_avg=(2.10, 3.40, 3.60), handicap_line=-0.5, handicap_avg=(2.08, 1.82),
             over25_avg=(1.95, 1.90))
MARGINS = {("goals", 0.5): 1.08, ("goals", 1.5): 1.07, ("goals", 3.5): 1.06, ("goals", 4.5): 1.09,
           ("spreads", -1.5): 1.06, ("btts", None): 1.07}


def match(pre=PRE, close=PRE):
    return Match(id=1, competition="EPL", country="England", season=2023, day=date(2023, 9, 2),
                 home="England:Chelsea", away="England:Leeds", home_goals=2, away_goals=1, pre=pre, close=close)


def by_key(options):
    return {pool.option_key(o): o for o in options}


def test_inverse_power_undoes_the_power_method():
    fair = [0.62, 0.38]
    prices = pool.inverse_power(fair, 1.06)
    assert devig.power(prices) == pytest.approx(fair, abs=1e-9)
    assert sum(1 / price for price in prices) == pytest.approx(1.06, abs=1e-9)


def test_the_pool_has_every_half_line_the_live_site_shows():
    keys = set(by_key(pool.build_pool(match(), margins=MARGINS)))
    goals = {("goals", side, line) for line in (0.5, 1.5, 2.5, 3.5, 4.5) for side in ("over", "under")}
    spreads = {key for line in (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)
               for key in (("spreads", "home", line), ("spreads", "away", -line))}
    others = {("btts", "yes", None), ("btts", "no", None),
              ("moneyline", "home", None), ("moneyline", "draw", None), ("moneyline", "away", None)}
    assert keys == goals | spreads | others  # no corners: football-data has no corner prices


def test_real_lines_keep_their_real_chance_and_price():
    options = by_key(pool.build_pool(match(), margins=MARGINS))
    over = options[("goals", "over", 2.5)]
    assert over["chance"] == pytest.approx(market_over25(PRE)[0]) and over["typical_price"] == 1.95
    assert options[("spreads", "away", 0.5)]["typical_price"] == 1.82
    real = {key for key, option in options.items() if option["real"]}
    assert real == {("moneyline", "home", None), ("moneyline", "draw", None), ("moneyline", "away", None),
                    ("goals", "over", 2.5), ("goals", "under", 2.5),
                    ("spreads", "home", -0.5), ("spreads", "away", 0.5)}
    for option in options.values():
        assert {"market", "side", "line", "chance", "typical_price", "real"} <= set(option)


def test_a_quarter_main_handicap_feeds_m1_but_stays_out_of_the_pool():
    quarter = replace(PRE, handicap_line=-0.75, handicap_avg=(2.20, 1.72))
    options = pool.build_pool(match(pre=quarter), margins=MARGINS)
    assert not any(abs(o["line"] or 0) == 0.75 for o in options)
    assert not any(o["real"] for o in options if o["market"] == "spreads")
    without = replace(quarter, handicap_line=None, handicap_avg=None)
    assert pool.m1_means(quarter) != pytest.approx(pool.m1_means(without), abs=1e-6)


def test_derived_chances_agree_with_m1():
    options = by_key(pool.build_pool(match(), margins=MARGINS))
    home, away = pool.m1_means(PRE)
    assert options[("goals", "over", 1.5)]["chance"] == pytest.approx(market_goals.over_probability(home, away, 1.5))
    assert options[("spreads", "home", -1.5)]["chance"] == pytest.approx(
        market_goals.handicap_probability(home, away, -1.5))
    over, under = options[("goals", "over", 1.5)], options[("goals", "under", 1.5)]
    assert over["chance"] + under["chance"] == pytest.approx(1)
    assert 1 / over["typical_price"] + 1 / under["typical_price"] == pytest.approx(1.07, abs=0.01)  # 2-decimal prices
    assert not over["real"]


def test_a_missing_line_is_derived_and_no_prices_give_an_empty_pool():
    no_totals = replace(PRE, over25_avg=None)
    options = by_key(pool.build_pool(match(pre=no_totals), margins=MARGINS))
    over = options[("goals", "over", 2.5)]
    assert not over["real"] and 0 < over["chance"] < 1  # never a chance of 0 or 1 from a missing price
    assert options[("moneyline", "home", None)]["real"]
    assert pool.build_pool(match(pre=Prices()), margins=MARGINS) == []
    assert pool.m1_means(Prices()) is None


def test_the_close_phase_reads_the_closing_prices():
    close = replace(PRE, over25_avg=(2.20, 1.70))
    options = by_key(pool.build_pool(match(close=close), phase="close", margins=MARGINS))
    assert options[("goals", "over", 2.5)]["typical_price"] == 2.20
