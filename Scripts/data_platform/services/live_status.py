"""Live status for started matches on the public board (tracker D70).

The stored schedule keeps "NS" until the next morning's refresh, on purpose: the engine and the refresh select
finished fixtures by that status. So the website asks API-Football itself, behind ``SPIX_LIVE_STATUS``: one
request per 20 started matches, at most every five minutes, never again once a match is over or called off, and
at most ``DAILY_CAP`` requests a UTC day. Statuses live in memory only; nothing is written to the database. A
board request never waits for another request's provider call: it gets what is already known.
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

logger = logging.getLogger(__name__)

REFRESH = timedelta(minutes=5)
BATCH = 20  # API-Football's limit for /fixtures?ids=
DAILY_CAP = 400
KEEP = timedelta(days=2)  # a long-running website forgets matches last asked about two days ago
FINAL = frozenset({"FT", "AET", "PEN", "PST", "CANC", "ABD", "AWD", "WO"})
TIMEOUT = (5, 10)  # connect, read: one attempt, so a slow provider cannot hold a board request for long


def enabled() -> bool:
    """The display switch: started matches get a live status only when ``SPIX_LIVE_STATUS`` is ``on``."""
    return os.environ.get("SPIX_LIVE_STATUS", "").strip().lower() == "on"


def _utc(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def fetch_statuses(ids: List[str]) -> List[Dict[str, Any]]:
    """One ``/fixtures?ids=`` request, no retries. Errors carry the path only, never the key or the URL."""
    from Scripts.football_http import ApiFootballResponseError, validate_envelope
    from ..sync.apifootball import API_BASE, ApiFootballClient

    session = ApiFootballClient(request_pause_s=0).session
    try:
        response = session.get(f"{API_BASE}/fixtures", params={"ids": "-".join(ids)}, timeout=TIMEOUT)
        try:
            if response.status_code != 200:
                raise ApiFootballResponseError(f"API-Football /fixtures: HTTP {response.status_code}")
            return validate_envelope(response.json(), "/fixtures")["response"]
        finally:
            response.close()
    finally:
        session.close()


class LiveStatuses:
    def __init__(self, fetch: Callable[[List[str]], Iterable[Mapping[str, Any]]] = fetch_statuses, *,
                 refresh: timedelta = REFRESH, daily_cap: int = DAILY_CAP, batch: int = BATCH) -> None:
        self._fetch, self._refresh, self._cap, self._batch = fetch, refresh, daily_cap, batch
        self._lock = threading.Lock()
        self._known: Dict[str, Dict[str, Any]] = {}
        self._asked: Dict[str, datetime] = {}
        self._calls: Dict[Any, int] = {}

    def lookup(self, fixtures: Iterable[Mapping[str, Any]], *, now: datetime) -> Dict[str, Optional[Dict[str, Any]]]:
        """``{event_id: {"short", "elapsed"} or None}`` for each fixture that has kicked off by ``now``."""
        started = []
        for item in fixtures:
            kickoff, event_id = _utc(item.get("kickoff")), str(item.get("event_id") or "").strip()
            if kickoff is not None and event_id and kickoff <= now and event_id not in started:
                started.append(event_id)
        if self._lock.acquire(blocking=False):
            try:
                self._ask(started, now)
            finally:
                self._lock.release()
        return {event_id: self._public(event_id) for event_id in started}

    def _public(self, event_id: str) -> Optional[Dict[str, Any]]:
        known = self._known.get(event_id)
        return {"short": known["short"], "elapsed": known["elapsed"]} if known else None

    def _ask(self, started: List[str], now: datetime) -> None:
        due = [event_id for event_id in started
               if (self._known.get(event_id) or {}).get("short") not in FINAL
               and (event_id not in self._asked or now - self._asked[event_id] >= self._refresh)]
        self._calls = {day: count for day, count in self._calls.items() if day == now.date()}
        for event_id in [key for key, at in self._asked.items() if now - at > KEEP]:
            self._asked.pop(event_id)
            self._known.pop(event_id, None)
        for start in range(0, len(due), self._batch):
            if self._calls.get(now.date(), 0) >= self._cap:
                logger.warning("Live status: the daily cap of %s requests is reached", self._cap)
                return
            ids = due[start:start + self._batch]
            self._calls[now.date()] = self._calls.get(now.date(), 0) + 1
            for event_id in ids:
                self._asked[event_id] = now
            try:
                rows = list(self._fetch(ids))
            except Exception as exc:  # the type only: transport messages can carry request details
                logger.warning("Live status request failed: %s", type(exc).__name__)
                continue
            for item in rows:
                fixture = item.get("fixture") if isinstance(item, Mapping) else None
                status = (fixture or {}).get("status") if isinstance(fixture, Mapping) else None
                short = str((status or {}).get("short") or "").strip().upper() if isinstance(status, Mapping) else ""
                if short and str(fixture.get("id")) in ids:
                    elapsed = status.get("elapsed")
                    self._known[str(fixture["id"])] = {
                        "short": short, "elapsed": elapsed if isinstance(elapsed, int) and not isinstance(elapsed, bool) else None}
