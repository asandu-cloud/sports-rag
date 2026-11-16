"""Game-read bench (D77): Opportunities, patterns in the history that no model in the design uses.

The owner, 14 Nov: "always look for possibilities during testing". Each pattern comes with its interval (whole
weeks resampled) and how many matches it rests on. Only the tuning seasons are measured, never the locked ones;
warm-up matches only tell which teams were in a league the season before. A pattern found here changes nothing by
itself: anything worth following becomes a candidate that goes through the bench and the gate.

- Drift: how the market's fair chance moves from pick time to the close, by market, league and weekday. A side
  whose chance rises on average was underpriced at pick time.
- Calibration by price band: whether favourites or longshots win more or less often than their price says.
- Promoted teams: a team new to its league (for the Championship, relegated teams too), its first 10 matches.
"""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Sequence

import numpy as np

from Scripts.data_platform.game_read import scores
from Scripts.data_platform.game_read.history import Match, block, market_result
from Scripts.data_platform.game_read.pool import option_key

BANDS = ((1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, math.inf))
FIRST_MATCHES = 10
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
DRIFT = {("moneyline", "home"): "home win", ("moneyline", "draw"): "draw", ("moneyline", "away"): "away win",
         ("goals", "over"): "over 2.5", ("spreads", "home"): "handicap home"}


def _stat(rows: Sequence, at: int = 0) -> dict:
    """The mean of value ``at`` (the last value counts the rows), with its week interval and match count."""
    low, high = scores.week_interval(rows, lambda t: scores._ratio(t[at], t[-1]))
    total = np.sum([v for _, v in rows], axis=0) if rows else None
    return {"estimate": scores._finite(scores._ratio(total[at], total[-1])) if rows else None,
            "low": low, "high": high, "n": len(rows)}


def _grouped(rows: Sequence) -> dict:
    leagues, days = defaultdict(list), defaultdict(list)
    for match, values in rows:
        leagues[match.competition].append((match, values))
        days[WEEKDAYS[match.day.weekday()]].append((match, values))
    return {"all": _stat(rows), "by_league": {k: _stat(v) for k, v in sorted(leagues.items())},
            "by_weekday": {d: _stat(days[d]) for d in WEEKDAYS if d in days}}


def _drift(matches: Sequence[Match]) -> dict:
    rows = defaultdict(list)
    for match in matches:
        for option in scores._pool(match):
            name = DRIFT.get((option["market"], option["side"]))
            if not option["real"] or name is None:
                continue
            after = scores.closing_chance(match, option)
            if after is not None:
                rows[name].append((match, (after - option["chance"], 1.0)))
    return {name: _grouped(rows[name]) for name in DRIFT.values() if rows[name]}


def _band(price: float) -> str:
    low, high = next(band for band in BANDS if band[0] <= price < band[1])
    return f"{low:g}+" if math.isinf(high) else f"{low:g}–{high:g}"


def _calibration(matches: Sequence[Match]) -> dict:
    """Each result outcome at pick time by its average price: chance said, how often it won, the gap, and the
    return of one unit at the average price."""
    rows = defaultdict(list)
    for match in matches:
        fair, prices = market_result(match.pre), match.pre.result_avg
        if fair is None:
            continue
        happened = 0 if match.home_goals > match.away_goals else 1 if match.home_goals == match.away_goals else 2
        for i, (said, price) in enumerate(zip(fair, prices)):
            won = float(i == happened)
            rows[_band(price)].append((match, (won - said, price * won - 1, said, won, 1.0)))
    out = {}
    for low, _ in BANDS:
        found = rows[_band(low)]
        if found:
            total = np.sum([v for _, v in found], axis=0)
            out[_band(low)] = {**_stat(found, 0), "return": _stat(found, 1),
                               "said": float(total[2] / total[-1]), "won": float(total[3] / total[-1])}
    return out


def _newcomers(matches: Sequence[Match]) -> set:
    """(competition, season, team) for each team new to its league, when the league was in the data the season
    before."""
    members = defaultdict(set)
    for match in matches:
        members[(match.competition, match.season)] |= {match.home, match.away}
    return {(league, season, team) for (league, season), teams in members.items()
            if (league, season - 1) in members for team in teams - members[(league, season - 1)]}


def _promoted(matches: Sequence[Match]) -> dict:
    new = _newcomers(matches)
    seen, rows = defaultdict(int), []
    for match in sorted(matches, key=lambda m: (m.day, m.id)):
        if block(match.season) != "tuning":
            continue
        fair = market_result(match.pre)
        for side, team in ((0, match.home), (2, match.away)):
            key = (match.competition, match.season, team)
            if key not in new or seen[key] >= FIRST_MATCHES or fair is None:
                continue
            seen[key] += 1
            won = float(match.home_goals > match.away_goals if side == 0 else match.away_goals > match.home_goals)
            rows.append((match, (won - fair[side], match.pre.result_avg[side] * won - 1, 1.0)))
    return {**_stat(rows, 0), "return": _stat(rows, 1),
            "teams": sum(1 for league, season, _ in new if block(season) == "tuning")}


def _walk(node, path=()):
    """(path, group) for every measured group: a dict with an estimate, its interval and a match count."""
    if isinstance(node, dict):
        if {"estimate", "low", "high", "n"} <= set(node):
            yield path, node
        for key, child in node.items():
            yield from _walk(child, path + (str(key),))


def _lead(path, group) -> bool:
    """A group whose interval leaves out zero. A return only counts above zero: a loss at the average price is
    the bookmaker's margin, not a lead."""
    if group["low"] is None:
        return False
    if path[-1:] == ("return",):
        return group["low"] > 0
    return group["low"] > 0 or group["high"] < 0


def scan(matches: Sequence[Match]) -> dict:
    """Drift, calibration by price band and promoted teams on the tuning seasons, how many groups were measured,
    and the leads (``_lead``), the largest first. Refuses any locked match."""
    if any(block(match.season) == "locked" for match in matches):
        raise ValueError("Opportunities are found on the tuning seasons only; locked matches were passed")
    tuning = [match for match in matches if block(match.season) == "tuning"]
    found = {"drift": _drift(tuning), "calibration": _calibration(tuning), "promoted": _promoted(matches)}
    groups = list(_walk(found))
    found["groups"] = len(groups)
    found["leads"] = sorted(({"what": " / ".join(path), **{k: g[k] for k in ("estimate", "low", "high", "n")}}
                             for path, g in groups if _lead(path, g)),
                            key=lambda lead: -abs(lead["estimate"]))
    return found
