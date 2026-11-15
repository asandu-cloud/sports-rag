"""Game-read bench (D77): the football-data history the bench tests on.

Every league match in ``Index/football_data.sqlite3`` with its final score and the market's prices for the result,
the main Asian handicap and over/under 2.5, before kickoff (collected Friday or Tuesday afternoon) and at the close.
Only football-data's average and best prices are loaded: its own averages from 2019/20, BetBrain's before (the
handicap and over/under name only two bookmakers in most seasons, so a median of named books would rest on two).
The market's chance is the power method on the average price, as on the live site.

The database is opened read-only. A team's key is ``"<country>:<football-data name>"``, so a team promoted from the
Championship keeps its key in the Premier League.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sqlite3
from typing import Iterable, Literal, Optional, Tuple

from Scripts.data_platform import devig
from Scripts.ops.football_data_import import DIVISIONS, OUT_DB

TUNING_FROM, LOCKED_FROM = 2019, 2023  # season start years: warm-up before, tuning, then locked
MARKET_FROM = 2019  # football-data's own average and maximum start with 2019/20
SOURCES = {"avg": ("BetBrain average", "Market average"), "max": ("BetBrain maximum", "Market maximum")}
AGGREGATES = frozenset(name for pair in SOURCES.values() for name in pair)
SIDES = {"1x2": ("home", "draw", "away"), "ah": ("home", "away"), "ou25": ("over", "under")}
FIELDS = {"1x2": "result", "ah": "handicap", "ou25": "over25"}

Triple = Tuple[float, float, float]
Pair = Tuple[float, float]


@dataclass(frozen=True)
class Prices:
    result_avg: Optional[Triple] = None
    result_max: Optional[Triple] = None
    handicap_line: Optional[float] = None
    handicap_avg: Optional[Pair] = None
    handicap_max: Optional[Pair] = None
    over25_avg: Optional[Pair] = None
    over25_max: Optional[Pair] = None


@dataclass(frozen=True)
class Match:
    id: int
    competition: str
    country: str
    season: int
    day: date
    home: str
    away: str
    home_goals: int
    away_goals: int
    pre: Prices
    close: Prices


def connect(path: Path = OUT_DB) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)


def block(season: int) -> Literal["warmup", "tuning", "locked"]:
    return "warmup" if season < TUNING_FROM else "tuning" if season < LOCKED_FROM else "locked"


def _source(season: int, kind: str) -> str:
    return SOURCES[kind][season >= MARKET_FROM]


def _prices(rows, season: int) -> Prices:
    """``rows``: {(bookmaker, market): {selection: (line, price)}} of one match and phase."""
    found = {}
    for market, sides in SIDES.items():
        lines = {}
        for kind in ("avg", "max"):
            quotes = rows.get((_source(season, kind), market), {})
            if all(side in quotes for side in sides):
                found[f"{FIELDS[market]}_{kind}"] = tuple(quotes[side][1] for side in sides)
                lines[kind] = quotes[sides[0]][0]
        if market == "ah" and lines:
            line = lines.get("avg", lines.get("max"))
            found["handicap_line"] = line
            if lines.get("max") != line:  # a best price on another line is not this line's price
                found.pop("handicap_max", None)
    return Prices(**found)


def load_matches(path: Path = OUT_DB, *, competitions: Optional[Iterable[str]] = None) -> list:
    """Every match with a final score, ordered by day then id."""
    wanted = set(competitions) if competitions is not None else None
    with closing(connect(path)) as conn:
        rows = conn.execute(
            "SELECT id, div, competition, season, match_date, home_name, away_name, fthg, ftag FROM matches"
            " WHERE fthg IS NOT NULL AND ftag IS NOT NULL").fetchall()
        quotes = defaultdict(lambda: defaultdict(dict))
        names = ", ".join("?" * len(AGGREGATES))
        for match_id, market, bookmaker, closing_price, selection, line, price in conn.execute(
                f"SELECT match_id, market, bookmaker, closing, selection, line, price FROM odds"
                f" WHERE bookmaker IN ({names})", sorted(AGGREGATES)):
            quotes[(match_id, bool(closing_price))][(bookmaker, market)][selection] = (line, price)
    matches = []
    for match_id, div, competition, season, day, home, away, home_goals, away_goals in rows:
        if wanted is not None and competition not in wanted:
            continue
        country = DIVISIONS[div][1]
        matches.append(Match(
            id=match_id, competition=competition, country=country, season=season, day=date.fromisoformat(day),
            home=f"{country}:{home}", away=f"{country}:{away}", home_goals=home_goals, away_goals=away_goals,
            pre=_prices(quotes.get((match_id, False), {}), season),
            close=_prices(quotes.get((match_id, True), {}), season)))
    return sorted(matches, key=lambda m: (m.day, m.id))


def _fair(prices):
    return None if prices is None else tuple(devig.power(prices))


def market_result(prices: Prices) -> Optional[Triple]:
    return _fair(prices.result_avg)


def market_handicap(prices: Prices) -> Optional[Pair]:
    return _fair(prices.handicap_avg)


def market_over25(prices: Prices) -> Optional[Pair]:
    return _fair(prices.over25_avg)


def named_result_prices(path: Path = OUT_DB) -> dict:
    """Each match's result prices before kickoff from every named bookmaker (no averages or maximums), leaving out
    prices flagged suspect. Only for checking the average price against the named books (design, bench point 3)."""
    quotes = defaultdict(lambda: defaultdict(dict))
    names = ", ".join("?" * len(AGGREGATES))
    with closing(connect(path)) as conn:
        for match_id, bookmaker, selection, price in conn.execute(
                f"SELECT match_id, bookmaker, selection, price FROM odds WHERE market = '1x2' AND closing = 0"
                f" AND suspect = 0 AND bookmaker NOT IN ({names})", sorted(AGGREGATES)):
            quotes[match_id][bookmaker][selection] = price
    return {match_id: [tuple(sides[s] for s in SIDES["1x2"]) for sides in books.values()
                       if all(s in sides for s in SIDES["1x2"])]
            for match_id, books in quotes.items()}
