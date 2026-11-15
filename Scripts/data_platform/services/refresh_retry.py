"""Hourly retry of the daily refresh: run when the day's refresh is due or the last one did not complete.

The daily refresh job fires at a quarter past every hour (``ops/upstream_refresh_launchd.py``) with
``refresh-season --if-due``. From 07:15 local time a run is due when no run has started since 07:15 today, or
when the last full refresh did not complete (the gate is shut, so no Match Reads or measurement run). On 26 and
27 Oct the 07:15 refresh failed on a sleeping Mac and nothing ran again until it was restarted by hand.

A run is skipped, and not counted, when the Mac is on battery with its lid shut (it would sleep mid-run) or when
API-Football's host does not resolve. After ``MAX_ATTEMPTS`` started runs in one day it stops, so a lasting fault
does not repeat the provider and embedding work all day. The safety gate itself is unchanged.

A late run is held for match time (D109, 15 Nov). While the refresh runs, the Match Read worker stands down, so a
refresh that starts late (the Mac slept through 07:15) can take cards off the board at kickoff: on 7 and 8 Nov it
ran over the midday and afternoon kickoffs (D108). From an hour after 07:15, a due run waits while a match the
worker covers kicks off before the run's likely end plus the worker's final window (the two hours before kickoff
when it checks every 10 minutes and the board drops an older card). It takes the first hourly tick clear of them,
and from 23:00 it starts whatever the kickoffs, so no day is skipped. The normal 07:15 run and a retry after a
failure are never held: with the gate shut the worker makes no cards at all until a refresh completes.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
import json
import logging
from pathlib import Path
import socket
import statistics
import subprocess
from typing import Callable, Optional, Sequence, Tuple

from .refresh_coordination import _paths

logger = logging.getLogger(__name__)

DAILY_AT = time(7, 15)
MAX_ATTEMPTS = 6
PROVIDER_HOST = "v3.football.api-sports.io"
KEEP = 60  # attempts kept in the record

NORMAL_HOUR = timedelta(hours=1)  # the 07:15 run's hour is never held
LAST_START = time(23, 0)  # from here a due run starts whatever the kickoffs
HELD = "held for match time"
RUN_FLOOR, RUN_CAP, RUN_DEFAULT = 60, 180, 120  # minutes
RUN_HISTORY = 14  # completed runs the likely run time reads


def refresh_due(now: datetime, *, gate_shut: bool, attempts: Sequence[datetime],
                daily_at: time = DAILY_AT, max_attempts: int = MAX_ATTEMPTS) -> Tuple[bool, str]:
    """Whether a refresh should start at ``now`` (local, aware), and why."""
    start = now.replace(hour=daily_at.hour, minute=daily_at.minute, second=0, microsecond=0)
    if now < start:
        return False, "before the daily refresh time"
    today = [a for a in attempts if a >= start]
    if len(today) >= max_attempts:
        return False, f"attempt limit reached today ({len(today)} of {max_attempts})"
    if gate_shut:
        return True, "the last full refresh did not complete"
    if today:
        return False, "today's refresh already ran"
    return True, "today's refresh has not run"


def likely_minutes(runs: Sequence[float]) -> float:
    """How long a refresh started now is likely to take: the upper quartile of the last ``RUN_HISTORY`` completed
    runs, between ``RUN_FLOOR`` and ``RUN_CAP``; ``RUN_DEFAULT`` with fewer than 5 on record.

    Completed runs on 2-15 Nov took about 45 minutes with the Mac awake, 97 and 150 after a weekend's matches, and
    hours on a sleeping Mac. The upper quartile covers the weekend runs; the cap keeps a run of sleeping mornings
    from holding the refresh all day.
    """
    recent = list(runs)[-RUN_HISTORY:]
    if len(recent) < 5:
        return RUN_DEFAULT
    return min(RUN_CAP, max(RUN_FLOOR, statistics.quantiles(recent, n=4)[2]))


def refresh_hold(now: datetime, *, gate_shut: bool, runs: Sequence[float],
                 kickoffs: Callable[[datetime, datetime], Sequence[Tuple[str, datetime]]], window_minutes: int,
                 daily_at: time = DAILY_AT) -> Optional[str]:
    """Why a due refresh at ``now`` (local, aware) waits for match time, or None when it may start.

    ``kickoffs(now, until)`` gives ``(league, kickoff)`` for the matches the worker covers; it is read only when a
    hold can apply.
    """
    start = now.replace(hour=daily_at.hour, minute=daily_at.minute, second=0, microsecond=0)
    if gate_shut or now < start + NORMAL_HOUR or now.time() >= LAST_START:
        return None
    minutes = likely_minutes(runs)
    until = now + timedelta(minutes=minutes + window_minutes)
    close = sorted((kickoff, league) for league, kickoff in kickoffs(now, until) if now < kickoff <= until)
    if not close:
        return None
    kickoff, league = close[0]
    return (f"{HELD}: {league} kicks off at {kickoff.astimezone(now.tzinfo):%H:%M}, inside the refresh's likely "
            f"{minutes:.0f} min and the {window_minutes}-minute final window")


def upcoming_kickoffs(now: datetime, until: datetime) -> list:
    """``(league, kickoff)`` of the matches the Match Read worker covers, from the schedule it plans with, read
    inside the data gate (``RefreshBusy`` at once while a refresh writes)."""
    from ..config import load_match_read_worker_settings
    from .match_read_cycle import _load_upcoming_platform_fixtures
    from .refresh_coordination import data_access
    leagues = load_match_read_worker_settings().leagues
    with data_access():
        fixtures = _load_upcoming_platform_fixtures(now=now, until=until, leagues=leagues)
    return [(fixture.league, fixture.kickoff_utc) for fixture in fixtures]


def can_stay_awake(run: Callable = subprocess.run) -> bool:
    """False only when the Mac is on battery with its lid shut; True when either cannot be read."""
    try:
        power = run(["/usr/bin/pmset", "-g", "ps"], capture_output=True, text=True, timeout=10).stdout
        lid = run(["/usr/sbin/ioreg", "-r", "-k", "AppleClamshellState", "-d", "4"],
                  capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return True
    return not ("Battery Power" in power and '"AppleClamshellState" = Yes' in lid)


def provider_reachable(host: str = PROVIDER_HOST) -> bool:
    try:
        socket.getaddrinfo(host, 443)
        return True
    except OSError:
        return False


def _attempts_path() -> Path:
    lock, _ = _paths()
    return Path(str(lock).removesuffix(".lock") + "-attempts.json")


def _state(path: Path) -> dict:
    try:
        state = json.loads(path.read_text())
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state))


def _load(path: Path) -> list:
    try:
        return [datetime.fromisoformat(a) for a in _state(path)["attempts"]]
    except (ValueError, KeyError, TypeError):
        return []


def _runs(path: Path) -> list:
    try:
        return [float(run["minutes"]) for run in _state(path).get("runs") or []]
    except (ValueError, KeyError, TypeError):
        return []


def attempts_today(now: Optional[datetime] = None, *, attempts_path: Optional[Path] = None,
                   daily_at: time = DAILY_AT) -> int:
    """Runs started by ``--if-due`` since today's refresh time; during a run, the number of that run."""
    now = now or datetime.now().astimezone()
    start = now.replace(hour=daily_at.hour, minute=daily_at.minute, second=0, microsecond=0)
    return sum(a >= start for a in _load(attempts_path or _attempts_path())) if now >= start else 0


def check_due(now: Optional[datetime] = None, *, gate_shut: Optional[bool] = None,
              attempts_path: Optional[Path] = None, stay_awake: Callable[[], bool] = can_stay_awake,
              reachable: Callable[[], bool] = provider_reachable, kickoffs: Optional[Callable] = None,
              window_minutes: Optional[int] = None) -> Tuple[bool, str]:
    """Decide for ``refresh-season --if-due`` and record the attempt when it is due."""
    now = now or datetime.now().astimezone()
    gate_shut = _paths()[1].exists() if gate_shut is None else gate_shut
    path = attempts_path or _attempts_path()
    state = _state(path)
    attempts = _load(path)
    due, reason = refresh_due(now, gate_shut=gate_shut, attempts=attempts)
    if due and not stay_awake():
        return False, f"{reason}, but the Mac is on battery with its lid shut"
    if due and not reachable():
        return False, f"{reason}, but {PROVIDER_HOST} does not resolve"
    if due:
        try:
            if window_minutes is None:
                from ..config import load_match_read_worker_settings
                window_minutes = load_match_read_worker_settings().final_window_minutes
            held = refresh_hold(now, gate_shut=gate_shut, runs=_runs(path), kickoffs=kickoffs or upcoming_kickoffs,
                                window_minutes=window_minutes)
        except Exception as exc:  # a hold must never cost the day's refresh
            logger.warning("Kickoffs could not be read (%s), so the refresh is not held", type(exc).__name__)
            held = None
        if held:
            return False, f"{reason}, but {held}"
        state["attempts"] = [a.isoformat() for a in (attempts + [now])[-KEEP:]]
        _save(path, state)
    return due, reason


def record_run(minutes: float, *, ok: bool, now: Optional[datetime] = None,
               attempts_path: Optional[Path] = None) -> None:
    """Keep a completed refresh's run time for ``likely_minutes``; a failed run says nothing about a full run's."""
    if not ok:
        return
    try:
        path = attempts_path or _attempts_path()
        state = _state(path)
        run = {"at": (now or datetime.now().astimezone()).isoformat(), "minutes": round(minutes, 1)}
        state["runs"] = ((state.get("runs") or []) + [run])[-KEEP:]
        _save(path, state)
    except Exception:  # the record is advice for tomorrow's hold, never a reason to fail today's run
        logger.warning("The refresh's run time could not be recorded", exc_info=True)


def withdraw_attempt(*, attempts_path: Optional[Path] = None) -> None:
    """Take back the attempt ``check_due`` just recorded: the run never got the gate, so nothing ran.

    On 1 Nov the 07:16 run timed out waiting for the gate and still counted, which skipped the rest of the day.
    """
    path = attempts_path or _attempts_path()
    state = _state(path)
    state["attempts"] = [a.isoformat() for a in _load(path)[:-1]]
    _save(path, state)
