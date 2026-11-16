"""Game-read bench (D77): each match's Most likely pool, rebuilt from the market's prices on football-data's history.

The live list chooses among every half line a read prices (``recommendations.options``). History has prices for
three lines only: the result, over/under 2.5 and the main Asian handicap. Those stay as they are, with the market's
chance (the power method on the average price) and the average price (``real``). Every other half line the live
site shows is derived: its chance from M1, the market's goal means solved from those three prices (a quarter or
whole main handicap still feeds M1), and its price from that chance and the line's margin measured on our saved
reads (``margins.json``), rounded to two decimals as a bookmaker quotes. A line whose real price is missing is
derived like the others, so one missing market never removes the rest.
"""
from __future__ import annotations

from functools import lru_cache
import math
from typing import Literal, Mapping, Optional, Sequence

from Scripts.data_platform import market_goals
from Scripts.data_platform.game_read import margins as margin_table
from Scripts.data_platform.game_read.history import (Match, Prices, market_handicap, market_over25,
                                                     market_result)

GOAL_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)
SPREAD_LINES = (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)  # the home side's line


def inverse_power(fair: Sequence[float], booksum: float) -> list:
    """Prices whose implied chances ``fair_i ** k`` sum to ``booksum`` (0 < k <= 1), so that the power method on
    them gives back ``fair``. A booksum of 1 or less gives the fair prices."""
    def total(k):
        return sum(p ** k for p in fair)

    lo, hi = 1e-9, 1.0
    if total(hi) < booksum:
        for _ in range(200):  # total() falls as k rises: find the k where it meets the booksum
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if total(mid) > booksum else (lo, mid)
        k = (lo + hi) / 2
    else:
        k = 1.0
    return [1 / p ** k for p in fair]


def m1_means(prices: Prices) -> Optional[tuple]:
    """M1's home and away goal means from the market's result, over 2.5 and main handicap chances (any main line,
    quarter and whole included), or None when they cannot identify both teams."""
    targets = {}
    result, over, handicap = market_result(prices), market_over25(prices), market_handicap(prices)
    if result:
        targets["1x2"] = list(result)
    if over:
        targets["totals"] = {2.5: over[0]}
    if handicap and prices.handicap_line is not None:
        targets["handicaps"] = {float(prices.handicap_line): handicap[0]}
    fit = market_goals.solve(targets)
    return None if fit is None else (fit["home"], fit["away"])


def option_key(option: Mapping) -> tuple:
    return option["market"], option["side"], option.get("line")


@lru_cache(maxsize=1)
def _saved_margins() -> dict:
    return margin_table.load()


def _half(line: Optional[float]) -> bool:
    return line is not None and math.isclose(abs(line) % 1, 0.5)


def _option(market, side, line, chance, price, real):
    return {"market": market, "side": side, "line": line, "chance": float(chance), "typical_price": float(price),
            "real": real}


def _derived(market, sides, lines, chances, booksum):
    prices = inverse_power(chances, booksum)
    return [_option(market, side, line, chance, round(price, 2), False)
            for side, line, chance, price in zip(sides, lines, chances, prices)]


def build_pool(match: Match, *, phase: Literal["pre", "close"] = "pre", margins: Optional[Mapping] = None) -> list:
    """The options of one match in the live shape (moneyline, goals, btts, spreads; lines ascending), each with
    ``real``. Empty when M1 cannot be solved."""
    prices = match.pre if phase == "pre" else match.close
    means = m1_means(prices)
    if means is None:
        return []
    table = _saved_margins() if margins is None else margins
    home, away = means
    found = []

    result = market_result(prices)
    if result:
        found += [_option("moneyline", side, None, chance, price, True)
                  for side, chance, price in zip(("home", "draw", "away"), result, prices.result_avg)]

    over = market_over25(prices)
    for line in GOAL_LINES:
        if line == 2.5 and over:
            found += [_option("goals", side, line, chance, price, True)
                      for side, chance, price in zip(("over", "under"), over, prices.over25_avg)]
            continue
        chance = market_goals.over_probability(home, away, line)
        found += _derived("goals", ("over", "under"), (line, line), (chance, 1 - chance),
                          margin_table.booksum(table, "goals", line))

    matrix = market_goals._matrix(home, away)
    yes = float(matrix[1:, 1:].sum() / matrix.sum())
    found += _derived("btts", ("yes", "no"), (None, None), (yes, 1 - yes), margin_table.booksum(table, "btts", None))

    handicap = market_handicap(prices)
    real_line = float(prices.handicap_line) if handicap and _half(prices.handicap_line) else None
    for line in sorted(set(SPREAD_LINES) | ({real_line} if real_line is not None else set())):
        if line == real_line:
            found += [_option("spreads", side, side_line, chance, price, True)
                      for side, side_line, chance, price in zip(("home", "away"), (line, -line), handicap,
                                                                prices.handicap_avg)]
            continue
        chance = market_goals.handicap_probability(home, away, line)
        found += _derived("spreads", ("home", "away"), (line, -line), (chance, 1 - chance),
                          margin_table.booksum(table, "spreads", line))
    return found
