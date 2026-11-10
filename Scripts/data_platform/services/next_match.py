"""The asked team's next match under a Research Assistant answer (tracker D95, owner 10 Nov, option B).

A team's corners or goals answer gets one row from the Match Read of the team's next match, the card the match page
shows: what Spix expects in that market and Spix's pick in it. Nothing is worked out afresh. The figures are the
card's own ("Corners expected" and "Goals expected" on the match page) and the pick is the market's suggestion in the
``recommendations.v1`` block, so the two pages always agree. Spix prices the whole match's corners only, so a corners
row has no figure for one team. Cards and shots on target stay out, as they stay out of the picks (D61).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

MARKETS = ("corners", "goals")  # a stat answered by a market of the same name in the recommendations block
PICK_FIELDS = ("label", "side", "line", "chance", "typical_price")
LONDON = ZoneInfo("Europe/London")

Read = Tuple[Mapping[str, Any], Any]  # the match page's card and the read's canonical results


def _number(value) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _day(moment: datetime) -> str:
    local = moment.astimezone(LONDON)
    return f"{local:%a} {local.day} {local:%b}"


def _corners(canonical_results) -> Optional[float]:
    results = canonical_results if isinstance(canonical_results, Sequence) and not isinstance(
        canonical_results, (str, bytes)) else []
    corners = next((r for r in results if isinstance(r, Mapping) and (r.get("market") or {}).get("group") == "corners"),
                   {})
    return _number((corners.get("projection") or {}).get("value"))


def expected(stat: str, card: Mapping[str, Any], canonical_results, venue: str) -> Optional[dict]:
    """What Spix expects for the asked team ("for"), its opponent ("against") and the match, to one decimal."""
    if stat == "corners":
        match = _corners(canonical_results)
        return None if match is None else {"for": None, "against": None, "match": round(match, 1)}
    goals = (card.get("metrics") or {}).get("team_goals") or {}
    home, away = _number(goals.get("home")), _number(goals.get("away"))
    if home is None or away is None:
        return None
    own, other = (home, away) if venue == "home" else (away, home)
    return {"for": round(own, 1), "against": round(other, 1), "match": round(home + away, 1)}


def pick(stat: str, block: Mapping[str, Any]) -> Optional[dict]:
    market = next((m for m in block.get("markets") or [] if isinstance(m, Mapping) and m.get("market") == stat), {})
    suggestion = market.get("suggestion")
    return {field: suggestion.get(field) for field in PICK_FIELDS} if isinstance(suggestion, Mapping) else None


def entry(fixture: Mapping[str, Any], read: Optional[Read], *, stat: str, now: datetime,
          outlook_hours: int) -> Optional[dict]:
    """The ``next_match`` of one answer: ``read`` is None while the fixture has no Match Read yet. None when the card
    has no recommendations block (the display switch is off)."""
    kickoff = datetime.fromisoformat(fixture["kickoff"])
    base = {"fixture": {key: fixture[key] for key in ("event_id", "league", "home_team", "away_team", "kickoff")},
            "venue": fixture["venue"], "opponent": fixture["opponent"], "market": stat}
    if read is None:
        opens = kickoff - timedelta(hours=outlook_hours)  # the Match Read worker's outlook window
        when = f"comes out from {_day(opens)}" if opens > now else "isn't out yet"
        return {**base, "status": "waiting", "label": None, "expected": None, "pick": None,
                "note": f"Spix's read of {fixture['home_team']} v {fixture['away_team']} ({_day(kickoff)}) {when}.",
                "prices_as_of": None}
    card, canonical_results = read
    block = card.get("recommendations")
    if not isinstance(block, Mapping):
        return None
    chosen = pick(stat, block)
    return {**base, "status": "read", "label": f"Spix expects v {fixture['opponent']}, {_day(kickoff)}",
            "expected": expected(stat, card, canonical_results, fixture["venue"]), "pick": chosen,
            "note": None if chosen else f"No pick from Spix in {stat} for that match.",
            "prices_as_of": block.get("prices_as_of")}


def find(team_id: int, stat: str, *, next_fixture: Callable[[int], Optional[Mapping[str, Any]]],
         load_read: Callable[[str], Optional[Read]], now: datetime, outlook_hours: int) -> Optional[dict]:
    """The team's next match for an answer about ``stat``, or None: a stat with no market, or no fixture ahead."""
    if stat not in MARKETS:
        return None
    fixture = next_fixture(team_id)
    if fixture is None:
        return None
    return entry(fixture, load_read(fixture["event_id"]), stat=stat, now=now, outlook_hours=outlook_hours)
