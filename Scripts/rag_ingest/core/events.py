"""Event fetching and filtering — wraps API-Football odds provider."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

try:
    from core.weights import LEAGUE_TO_ODDS_SPORT, DEFAULT_MARKETS
except ImportError:
    from weights import LEAGUE_TO_ODDS_SPORT, DEFAULT_MARKETS

log = logging.getLogger("core.events")


# ---------------------------------------------------------------------------
# Fetch events via API-Football
# ---------------------------------------------------------------------------

def fetch_events(league: str, markets: Optional[Set[str]] = None,
                 target_date: Optional[date] = None) -> Tuple[List[Dict], List[str]]:
    """Fetch events and odds from API-Football."""
    if target_date is None:
        target_date = date.today()

    try:
        from odds_provider import fetch_events_apifootball
        return fetch_events_apifootball(league, target_date)
    except ImportError:
        return [], ["odds_provider module not available."]
    except Exception as exc:
        log.warning("API-Football odds failed for %s: %s", league, exc)
        return [], [f"API-Football error: {exc}"]


def fetch_events_multi(leagues: List[str], markets: Optional[Set[str]] = None,
                       target_date: Optional[date] = None) -> Tuple[List[Dict], List[str]]:
    """Fetch events from multiple leagues, tagging each with its league."""
    all_events: List[Dict] = []
    all_notes: List[str] = []
    for league in leagues:
        events, notes = fetch_events(league, markets, target_date)
        for ev in events:
            ev["_league"] = league
        all_events.extend(events)
        all_notes.extend(notes)
    return all_events, all_notes


def filter_events_by_exact_date(events: List[Dict], target: date,
                                tz_name: str = "Europe/London") -> List[Dict]:
    """Filter events to those on a specific calendar date in the given timezone."""
    tz = ZoneInfo(tz_name)
    out: List[Dict] = []
    for ev in events:
        ts = ev.get("commence_time")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(tz)
        except Exception:
            continue
        if dt.date() == target:
            out.append(ev)
    return out
