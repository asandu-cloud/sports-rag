"""Game-read bench (D77): the Most likely and Value rules the bench scores, today's and the engine's.

Today's Most likely list is the live one, run unchanged on a rebuilt pool. The engine's keeps the live floors and
shape (one suggestion a market, up to three that can win together and are not the same bet) and changes only the
order: an option the engine backs, its chance above the market's by more than float noise, comes before the rest,
the strongest first by a ranking rule (the edge, or the blended chance); the rest keep the live order. With the
market as the engine nothing is backed, so the list is today's.

Value is scored on the lines history prices: the result, over/under 2.5 and a half-line main handicap, at the
average or the best price. Today's rule uses the market's chance; the engine's uses its own. Both need 2% value and
keep at most two a match, highest value first, able to win together.
"""
from __future__ import annotations

from typing import Callable, Literal, Mapping, Optional

from Scripts.data_platform import blend, recommendations
from Scripts.data_platform.game_read.history import Match, market_handicap, market_over25, market_result
from Scripts.data_platform.game_read.pool import _half, option_key
from Scripts.data_platform.recommendations import MAX_MOST_LIKELY, MAX_VALUE, MIN_CHANCE, MIN_PRICE

VALUE_THRESHOLD = 0.02
BACKED_MARGIN = 1e-9  # a chance above the market's by less is float noise (a matrix's sums against the pool's)
Strength = Callable[[dict, float], float]


def _by_market(pool: list) -> dict:
    markets = {}
    for option in pool:
        markets.setdefault(option["market"], []).append(option)
    return markets


def todays_most_likely(pool: list) -> list:
    """The live rule, unchanged: each market's suggestion, then the list."""
    return recommendations.most_likely([recommendations.suggestion(options) for options in _by_market(pool).values()])


def by_edge(option: Mapping, engine_chance: float) -> float:
    return engine_chance - option["chance"]


def by_blend(model: Mapping) -> Strength:
    """The blended chance of the option winning, from a two-way blend (``blend.fit`` on win or lose)."""
    def strength(option: Mapping, engine_chance: float) -> float:
        market, engine = option["chance"], engine_chance
        return float(blend.predict(model, [[market, 1 - market]], [[engine, 1 - engine]])[0][0])
    return strength


def engine_most_likely(pool: list, engine: Mapping[tuple, float], *, strength: Strength) -> list:
    """The live list with the engine's backed options first. Options the engine cannot score are not backed."""
    def backed(option):
        chance = engine.get(option_key(option))
        return chance is not None and chance > option["chance"] + BACKED_MARGIN

    def order(option):
        if backed(option):
            return (0, -strength(option, engine[option_key(option)]), *recommendations._rank(option))
        return (1, 0.0, *recommendations._rank(option))

    suggestions = []
    for options in _by_market(pool).values():
        eligible = [o for o in options if o["chance"] >= MIN_CHANCE and o["typical_price"] >= MIN_PRICE]
        if eligible:
            suggestions.append(min(eligible, key=order))
    kept = []
    for option in sorted(suggestions, key=order):
        if len(kept) == MAX_MOST_LIKELY:
            break
        if all(k["market"] != option["market"] and recommendations.can_win_together(k, option)
               and not recommendations.same_bet(k, option) for k in kept):
            kept.append(option)
    return kept


def _candidate(market, side, line, chance, price):
    return {"market": market, "side": side, "line": line, "chance": float(chance), "price": float(price)}


def value_candidates(match: Match, *, price: Literal["avg", "max"]) -> list:
    """The lines history prices before kickoff, each with the market's chance and the asked price: the result, over
    and under 2.5, and a half-line main handicap's two sides (a quarter or whole line has pushes and is left out)."""
    prices = match.pre
    found = []
    result, quotes = market_result(prices), getattr(prices, f"result_{price}")
    if result and quotes:
        found += [_candidate("moneyline", side, None, chance, quote)
                  for side, chance, quote in zip(("home", "draw", "away"), result, quotes)]
    over, quotes = market_over25(prices), getattr(prices, f"over25_{price}")
    if over and quotes:
        found += [_candidate("goals", side, 2.5, chance, quote)
                  for side, chance, quote in zip(("over", "under"), over, quotes)]
    handicap, quotes = market_handicap(prices), getattr(prices, f"handicap_{price}")
    if handicap and quotes and _half(prices.handicap_line):
        line = float(prices.handicap_line)
        found += [_candidate("spreads", side, side_line, chance, quote)
                  for side, side_line, chance, quote in zip(("home", "away"), (line, -line), handicap, quotes)]
    return found


def _value_list(scored: list) -> list:
    kept = []
    for pick in sorted(scored, key=lambda p: -p["value"]):
        if len(kept) == MAX_VALUE:
            break
        if all(recommendations.can_win_together(k, pick) and not recommendations.same_bet(k, pick) for k in kept):
            kept.append(pick)
    return kept


def _scored(candidate: Mapping, chance: Optional[float], **extra) -> Optional[dict]:
    if chance is None:
        return None
    value = chance * candidate["price"] - 1
    return {**candidate, **extra, "value": value} if value >= VALUE_THRESHOLD else None


def todays_value(candidates: list) -> list:
    return _value_list([p for c in candidates if (p := _scored(c, c["chance"]))])


def engine_value(candidates: list, engine: Mapping[tuple, float]) -> list:
    picks = []
    for candidate in candidates:
        chance = engine.get(option_key(candidate))
        pick = _scored(candidate, chance, engine_chance=chance)
        if pick:
            picks.append(pick)
    return _value_list(picks)
