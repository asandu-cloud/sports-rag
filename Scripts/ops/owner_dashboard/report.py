"""The report's numbers: "said" (the average chance Spix gave) against "hit" (how often the picks came in), over
settled picks only, day by day and week by week (by kickoff, ISO weeks, the Mac's local time), market by market and
league by league.

Value picks weigh as the scorecard does (``scorecard.outcome_weight``): a half result counts half, a push or a
void carries no weight. Most likely options are half lines, so each settles whole. Value picks also carry their
flat-stake return, as the scorecard counts it (``outcomes.outcome_accounting``): one unit on each settled pick at
the price it was published at; a push or a void gives the unit back, a half result settles half of it.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from Scripts.data_platform.outcomes import is_valid_decimal_odds, outcome_accounting
from Scripts.data_platform.scorecard import outcome_weight

MARKETS = {"moneyline": "Result", "goals": "Goals", "btts": "Both teams to score", "spreads": "Handicap",
           "corners": "Corners", "cards": "Cards", "sot": "Shots on target"}
KINDS = {"most_likely": "Most likely", "value": "Value"}
LEAGUES = {"EPL": "Premier League", "LaLiga": "La Liga", "SerieA": "Serie A", "Bundesliga": "Bundesliga",
           "Ligue1": "Ligue 1", "Championship": "Championship", "SuperLig": "Süper Lig", "Eredivisie": "Eredivisie",
           "PrimeiraLiga": "Primeira Liga", "BelgianProLeague": "Jupiler Pro League", "UCL": "Champions League",
           "UEL": "Europa League", "UECL": "Conference League"}  # the live site's names and order
OTHER = "Other"  # a pick whose match has no league stored


def local(kickoff) -> Optional[datetime]:
    """A stored UTC time on the Mac's clock."""
    try:
        return datetime.fromisoformat(str(kickoff).replace("Z", "+00:00")).astimezone()
    except ValueError:
        return None


def _picks(conn, kind: str) -> List[dict]:
    """(kickoff, league, market, said, outcome) for every logged pick of one list, and the odds of a Value pick."""
    if kind == "most_likely":
        sql = ("SELECT m.kickoff, COALESCE(m.league, ?) AS league, p.market, p.chance AS said, p.outcome "
               "FROM most_likely p JOIN matches m USING (fixture_api_id)")
    elif kind == "value":
        sql = "SELECT kickoff, COALESCE(league, ?) AS league, market, said, outcome, odds FROM value_picks"
    else:
        raise ValueError(f"unknown list: {kind!r}")
    return [dict(row) for row in conn.execute(sql, (OTHER,))]


def tally(picks: List[dict]) -> dict:
    """Said, hit and counts over some picks; units staked, profit and ROI over those with odds (Value picks)."""
    weight = said = hit = staked = returned = 0.0
    settled = pending = 0
    for pick in picks:
        if pick["outcome"] is None:
            pending += 1
            continue
        settled += 1
        y, w = outcome_weight(pick["outcome"])
        if w and pick["said"] is not None:
            weight, said, hit = weight + w, said + w * pick["said"], hit + w * y
        accounting = outcome_accounting(pick["outcome"], pick.get("odds"))
        if accounting and is_valid_decimal_odds(pick.get("odds")):
            staked, returned = staked + accounting.stake_units, returned + accounting.returned_units
    return {"said": said / weight if weight else None, "hit": hit / weight if weight else None,
            "settled": weight, "picks": settled, "pending": pending, "staked": staked,
            "profit": returned - staked if staked else None, "roi": (returned - staked) / staked if staked else None}


def _monday(moment: datetime):
    return (moment - timedelta(days=moment.weekday())).date()


def _ordered(names, order: List[str]) -> List[str]:
    return sorted(names, key=lambda name: (order.index(name) if name in order else len(order), name))


def summary(conn, *, kind: str, market: Optional[str] = None, league: Optional[str] = None) -> dict:
    """``days`` (each day with a pick), ``weeks`` (every ISO week from the first pick's to the last's, empty ones
    too), ``markets`` (each market with a pick in ``league``, in a fixed order), ``leagues`` (each league with a
    pick in ``market``, in the live site's order) and ``overall``; ``days``, ``weeks`` and ``overall`` follow both
    when given."""
    picks = [p for p in _picks(conn, kind) if local(p["kickoff"])]
    in_league = [p for p in picks if league is None or p["league"] == league]
    in_market = [p for p in picks if market is None or p["market"] == market]
    markets = [dict(tally([p for p in in_league if p["market"] == name]), market=name)
               for name in _ordered({p["market"] for p in in_league}, list(MARKETS))]
    leagues = [dict(tally([p for p in in_market if p["league"] == name]), league=name)
               for name in _ordered({p["league"] for p in in_market}, list(LEAGUES))]
    chosen = [p for p in in_league if market is None or p["market"] == market]
    by_day: Dict = {}
    for pick in chosen:
        by_day.setdefault(local(pick["kickoff"]).date(), []).append(pick)
    days = [dict(tally(by_day[day]), day=day.isoformat()) for day in sorted(by_day)]
    by_week: Dict = {}
    for pick in chosen:
        by_week.setdefault(_monday(local(pick["kickoff"])), []).append(pick)
    weeks = []
    if by_week:
        week, last = min(by_week), max(by_week)
        while week <= last:
            weeks.append(dict(tally(by_week.get(week, [])), week=week.isoformat()))
            week += timedelta(days=7)
    return {"days": days, "weeks": weeks, "markets": markets, "leagues": leagues, "overall": tally(chosen)}
