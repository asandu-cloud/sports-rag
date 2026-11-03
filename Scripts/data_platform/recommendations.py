"""Recommendation layer (tracker D52, D53): what Spix would take in each market of a match, from a read's prices.

Every option is a line of Result, Goals, Both teams to score, Handicap or Corners that at least
``fair_prices.MIN_BOOKS`` bookmakers price on every side, half lines only, so no bet ends with the stake returned or
split. Its chance is the market's fair chance (power method, median across books, the fair-price card's numbers);
its typical price is the median of the bookmakers' best prices, with no bookmaker named.

Spix's suggestion in a market is its likeliest option at a chance of at least ``MIN_CHANCE`` and a typical price of
at least ``MIN_PRICE``. Most likely is the top ``MAX_MOST_LIKELY`` markets' suggestions that can all win together:
goal-market legs are checked against the scorelines 0-0 to ``MAX_SCORE``-``MAX_SCORE``; corners never conflict.
No engine output changes: the lists are compiled from what the read already stores.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, Iterable, List, Mapping, Optional

from . import fair_prices as fp

VERSION = "recommendations.v1"
MIN_CHANCE = 0.50
MIN_PRICE = 1.30
MAX_MOST_LIKELY = 3
MAX_VALUE = 2
LONG_SHOT = 0.25
MAX_SCORE = 10
MARKETS = ("moneyline", "goals", "btts", "spreads", "corners")
GOAL_MARKETS = MARKETS[:4]


def _half_line(line) -> bool:
    return line is None or (isinstance(line, (int, float)) and line % 1 == 0.5)


def _signed(line: float) -> str:
    return f"{'−' if line < 0 else '+'}{abs(line):g}"


def _label(result: Mapping[str, Any], market: str, side: str, line) -> str:
    fixture = result.get("fixture") or {}
    team = {"home": fixture.get("home_team"), "away": fixture.get("away_team")}
    if market in ("goals", "corners"):
        return f"{side.capitalize()} {line:g} {market}"
    if market == "moneyline":
        return "Draw" if side == "draw" else f"{team[side]} to win"
    if market == "btts":
        return "Both teams to score" if side == "yes" else "Not both teams to score"
    return f"{team[side]} {_signed(line)}"


def _market_options(result: Mapping[str, Any], market: str) -> List[dict]:
    books, found = fp._books(result, market), []
    for line in fp._lines(result, market, books):
        if not _half_line(line):
            continue
        entry = fp._entry(result, market, books, line)
        if entry is None:
            continue
        priced = [(o, fp.typical_price(books, market, o["side"], o["line"])) for o in entry["outcomes"]]
        if any(price is None for _, price in priced):
            continue
        for index, (outcome, price) in enumerate(priced):
            side = ("home", "away")[index] if market == "spreads" else outcome["side"]
            chance = outcome["fair_probability"]
            found.append({"market": market, "side": side, "line": outcome["line"],
                          "label": _label(result, market, side, outcome["line"]), "chance": chance,
                          "typical_price": price, "long_shot": chance < LONG_SHOT})
    return found


def options(canonical_results: Iterable[Mapping[str, Any]]) -> Dict[str, List[dict]]:
    """Every option of each market the read covers, lines ascending, sides in the card's order (handicap: home
    team first, each side with its own handicap)."""
    results = fp._results(canonical_results)
    return {market: _market_options(results[market], market) for market in MARKETS if market in results}


def wins(option: Mapping[str, Any], home: int, away: int) -> bool:
    """Whether a goal-market option wins on the scoreline ``home``-``away``."""
    market, side, line = option.get("market"), option.get("side"), option.get("line")
    if market == "goals":
        return home + away > line if side == "over" else home + away < line
    if market == "btts":
        return (home > 0 and away > 0) == (side == "yes")
    if market == "moneyline":
        return {"home": home > away, "draw": home == away, "away": away > home}[side]
    if market == "spreads":
        return home + line > away if side == "home" else away + line > home
    raise ValueError(f"not a goal market: {market!r}")


def can_win_together(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """True when at least one scoreline wins both; corners never conflict with anything."""
    if "corners" in (a.get("market"), b.get("market")):
        return True
    return any(wins(a, h, w) and wins(b, h, w) for h, w in itertools.product(range(MAX_SCORE + 1), repeat=2))


def _rank(option: Mapping[str, Any]):
    return -option["chance"], -option["typical_price"]


def suggestion(market_options: List[dict]) -> Optional[dict]:
    """Spix's suggestion in one market: the likeliest option that clears both floors, ties to the higher price."""
    eligible = [o for o in market_options or [] if o["chance"] >= MIN_CHANCE and o["typical_price"] >= MIN_PRICE]
    return min(eligible, key=_rank) if eligible else None


def most_likely(suggestions: Iterable[Optional[dict]]) -> List[dict]:
    """Up to ``MAX_MOST_LIKELY`` markets' suggestions, by chance, each able to win with every one kept before it."""
    kept: List[dict] = []
    for option in sorted((s for s in suggestions if s), key=_rank):
        if len(kept) == MAX_MOST_LIKELY:
            break
        if all(k["market"] != option["market"] and can_win_together(k, option) for k in kept):
            kept.append(option)
    return kept
