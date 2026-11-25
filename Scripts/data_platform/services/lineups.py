"""Confirmed line-ups for the match page (tracker D135, the frontend's D134).

API-Football has no predicted line-ups; it publishes the confirmed elevens about 20–40 minutes before kickoff. Behind
``SPIX_LINEUPS`` the website asks for them itself, only for matches someone opens: from an hour before kickoff, every
five minutes until both elevens are out, once more at kickoff for late changes, then every half hour until 150 minutes
after kickoff when they never came. Matches wanted at the same time share one ``/fixtures?ids=`` request (20 at most),
and at most ``DAILY_CAP`` requests go out a UTC day. The elevens live in memory only; nothing is written, and no pick
or read depends on them. A request never waits for another request's provider call: it gets what is already known.
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .live_status import BATCH, fetch_statuses

logger = logging.getLogger(__name__)

OPENS = timedelta(minutes=60)  # before kickoff
CLOSES = timedelta(minutes=150)  # after kickoff
REFRESH = timedelta(minutes=5)
LATE_REFRESH = timedelta(minutes=30)
DAILY_CAP = 400
KEEP = timedelta(days=2)


def enabled() -> bool:
    """The display switch: the match page gets line-ups only when ``SPIX_LINEUPS`` is ``on``."""
    return os.environ.get("SPIX_LINEUPS", "").strip().lower() == "on"


def _player(item: Any) -> dict:
    player = (item or {}).get("player") if isinstance(item, Mapping) else None
    player = player if isinstance(player, Mapping) else {}
    api_id = player.get("id")
    return {"api_player_id": api_id if isinstance(api_id, int) and not isinstance(api_id, bool) else None,
            "name": player.get("name"), "number": player.get("number"), "position": player.get("pos"),
            "grid": player.get("grid")}


def _side(lineup: Mapping[str, Any]) -> dict:
    coach = lineup.get("coach") if isinstance(lineup.get("coach"), Mapping) else {}
    return {"formation": lineup.get("formation"), "coach": coach.get("name"),
            "start_xi": [_player(p) for p in lineup.get("startXI") or []],
            "bench": [_player(p) for p in lineup.get("substitutes") or []]}


def elevens(item: Mapping[str, Any]) -> Optional[Dict[str, dict]]:
    """``{"home", "away"}`` from one ``/fixtures`` row once both teams' elevens are out, matched by team id."""
    teams = item.get("teams") if isinstance(item.get("teams"), Mapping) else {}
    by_team = {}
    for lineup in item.get("lineups") or []:
        team = lineup.get("team") if isinstance(lineup, Mapping) and isinstance(lineup.get("team"), Mapping) else {}
        if len(lineup.get("startXI") or []) == 11:
            by_team[team.get("id")] = _side(lineup)
    sides = {s: by_team.get(((teams.get(s) or {}) if isinstance(teams.get(s), Mapping) else {}).get("id"))
             for s in ("home", "away")}
    return sides if all(sides.values()) else None


class Lineups:
    def __init__(self, fetch: Callable[[List[str]], Iterable[Mapping[str, Any]]] = fetch_statuses, *,
                 daily_cap: int = DAILY_CAP, batch: int = BATCH) -> None:
        self._fetch, self._cap, self._batch = fetch, daily_cap, batch
        self._lock = threading.Lock()
        self._wanted: Dict[str, datetime] = {}
        self._known: Dict[str, Dict[str, dict]] = {}
        self._asked: Dict[str, datetime] = {}
        self._answered: Dict[str, datetime] = {}
        self._calls: Dict[Any, int] = {}

    def lookup(self, event_id: str, kickoff: datetime, now: datetime) -> dict:
        """``{"status", "checked_at", "home", "away"}``: confirmed, not out yet, or unavailable (none by kickoff's
        check, or the window closed)."""
        event_id = str(event_id)
        self._wanted[event_id] = kickoff
        if self._lock.acquire(blocking=False):
            try:
                self._ask(now)
            finally:
                self._lock.release()
        return self._public(event_id, kickoff, now)

    def _public(self, event_id: str, kickoff: datetime, now: datetime) -> dict:
        known, answered = self._known.get(event_id), self._answered.get(event_id)
        if known:
            status = "confirmed"
        elif now > kickoff + CLOSES or (answered is not None and answered >= kickoff):
            status = "unavailable"
        else:
            status = "not_out_yet"
        return {"status": status, "checked_at": answered.isoformat() if answered else None,
                "home": (known or {}).get("home"), "away": (known or {}).get("away")}

    def _due(self, event_id: str, kickoff: datetime, now: datetime) -> bool:
        if not kickoff - OPENS <= now <= kickoff + CLOSES:
            return False
        asked = self._asked.get(event_id)
        if asked is None or (now >= kickoff > asked):
            return True  # the first ask, or the one check at kickoff for late changes
        if event_id in self._known:
            return False
        return now - asked >= (REFRESH if now < kickoff else LATE_REFRESH)

    def _ask(self, now: datetime) -> None:
        self._calls = {day: count for day, count in self._calls.items() if day == now.date()}
        for event_id, kickoff in list(self._wanted.items()):
            if now - kickoff > KEEP:
                for store in (self._wanted, self._known, self._asked, self._answered):
                    store.pop(event_id, None)
        due = [e for e, kickoff in list(self._wanted.items()) if self._due(e, kickoff, now)]
        for start in range(0, len(due), self._batch):
            if self._calls.get(now.date(), 0) >= self._cap:
                logger.warning("Line-ups: the daily cap of %s requests is reached", self._cap)
                return
            ids = due[start:start + self._batch]
            self._calls[now.date()] = self._calls.get(now.date(), 0) + 1
            for event_id in ids:
                self._asked[event_id] = now
            try:
                rows = list(self._fetch(ids))
            except Exception as exc:  # the type only: transport messages can carry request details
                logger.warning("Line-ups request failed: %s", type(exc).__name__)
                continue
            for item in rows:
                fixture = item.get("fixture") if isinstance(item, Mapping) else None
                event_id = str((fixture or {}).get("id")) if isinstance(fixture, Mapping) else ""
                if event_id not in ids:
                    continue
                self._answered[event_id] = now
                sides = elevens(item)
                if sides:
                    self._known[event_id] = sides
