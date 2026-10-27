"""Hourly retry of the daily refresh: run when the day's refresh is due or the last one did not complete.

The daily refresh job fires at a quarter past every hour (``ops/upstream_refresh_launchd.py``) with
``refresh-season --if-due``. From 07:15 local time a run is due when no run has started since 07:15 today, or
when the last full refresh did not complete (the gate is shut, so no Match Reads or measurement run). On 26 and
27 Oct the 07:15 refresh failed on a sleeping Mac and nothing ran again until it was restarted by hand.

A run is skipped, and not counted, when the Mac is on battery with its lid shut (it would sleep mid-run) or when
API-Football's host does not resolve. After ``MAX_ATTEMPTS`` started runs in one day it stops, so a lasting fault
does not repeat the provider and embedding work all day. The safety gate itself is unchanged.
"""
from __future__ import annotations

from datetime import datetime, time
import json
import logging
from pathlib import Path
import socket
import subprocess
from typing import Callable, Optional, Sequence, Tuple

from .refresh_coordination import _paths

logger = logging.getLogger(__name__)

DAILY_AT = time(7, 15)
MAX_ATTEMPTS = 6
PROVIDER_HOST = "v3.football.api-sports.io"
KEEP = 60  # attempts kept in the record


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


def _load(path: Path) -> list:
    try:
        return [datetime.fromisoformat(a) for a in json.loads(path.read_text())["attempts"]]
    except (OSError, ValueError, KeyError, TypeError):
        return []


def check_due(now: Optional[datetime] = None, *, gate_shut: Optional[bool] = None,
              attempts_path: Optional[Path] = None, stay_awake: Callable[[], bool] = can_stay_awake,
              reachable: Callable[[], bool] = provider_reachable) -> Tuple[bool, str]:
    """Decide for ``refresh-season --if-due`` and record the attempt when it is due."""
    now = now or datetime.now().astimezone()
    gate_shut = _paths()[1].exists() if gate_shut is None else gate_shut
    path = attempts_path or _attempts_path()
    attempts = _load(path)
    due, reason = refresh_due(now, gate_shut=gate_shut, attempts=attempts)
    if due and not stay_awake():
        return False, f"{reason}, but the Mac is on battery with its lid shut"
    if due and not reachable():
        return False, f"{reason}, but {PROVIDER_HOST} does not resolve"
    if due:
        path.write_text(json.dumps({"attempts": [a.isoformat() for a in (attempts + [now])[-KEEP:]]}))
    return due, reason
