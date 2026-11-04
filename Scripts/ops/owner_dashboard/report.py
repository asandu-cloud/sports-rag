"""The report's numbers: "said" (the average chance Spix gave) against "hit" (how often the picks came in), over
settled picks only, week by week (ISO weeks by kickoff, the Mac's local time) and market by market.

Value picks weigh as the scorecard does (``scorecard.outcome_weight``): a half result counts half, a push or a
void carries no weight. Most likely options are half lines, so each settles whole.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from Scripts.data_platform.scorecard import outcome_weight

MARKETS = {"moneyline": "Result", "goals": "Goals", "btts": "Both teams to score", "spreads": "Handicap",
           "corners": "Corners", "cards": "Cards", "sot": "Shots on target"}
KINDS = {"most_likely": "Most likely", "value": "Value"}


def local(kickoff) -> Optional[datetime]:
    """A stored UTC time on the Mac's clock."""
    try:
        return datetime.fromisoformat(str(kickoff).replace("Z", "+00:00")).astimezone()
    except ValueError:
        return None


def _picks(conn, kind: str) -> List[dict]:
    """(kickoff, market, said, outcome) for every logged pick of one list."""
    if kind == "most_likely":
        sql = ("SELECT m.kickoff, p.market, p.chance AS said, p.outcome FROM most_likely p "
               "JOIN matches m USING (fixture_api_id)")
    elif kind == "value":
        sql = "SELECT kickoff, market, said, outcome FROM value_picks"
    else:
        raise ValueError(f"unknown list: {kind!r}")
    return [dict(row) for row in conn.execute(sql)]


def _tally(picks: List[dict]) -> dict:
    weight = said = hit = 0.0
    settled = pending = 0
    for pick in picks:
        if pick["outcome"] is None:
            pending += 1
            continue
        settled += 1
        y, w = outcome_weight(pick["outcome"])
        if w and pick["said"] is not None:
            weight, said, hit = weight + w, said + w * pick["said"], hit + w * y
    return {"said": said / weight if weight else None, "hit": hit / weight if weight else None,
            "settled": weight, "picks": settled, "pending": pending}


def _monday(moment: datetime):
    return (moment - timedelta(days=moment.weekday())).date()


def summary(conn, *, kind: str, market: Optional[str] = None) -> dict:
    """``weeks`` (every ISO week from the first pick's to the last's, empty ones too), ``markets`` (each market
    with a pick, in a fixed order) and ``overall``; ``weeks`` and ``overall`` follow ``market`` when given."""
    picks = [p for p in _picks(conn, kind) if local(p["kickoff"])]
    order = list(MARKETS)
    markets = [dict(_tally([p for p in picks if p["market"] == name]), market=name)
               for name in sorted({p["market"] for p in picks}, key=lambda m: (order.index(m) if m in order
                                                                                else len(order), m))]
    chosen = [p for p in picks if market is None or p["market"] == market]
    by_week: Dict = {}
    for pick in chosen:
        by_week.setdefault(_monday(local(pick["kickoff"])), []).append(pick)
    weeks = []
    if by_week:
        week, last = min(by_week), max(by_week)
        while week <= last:
            weeks.append(dict(_tally(by_week.get(week, [])), week=week.isoformat()))
            week += timedelta(days=7)
    return {"weeks": weeks, "markets": markets, "overall": _tally(chosen)}
