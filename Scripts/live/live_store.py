"""Persistence helpers for the live runtime.

Stores raw live state, model snapshots, and emitted alerts so the live stack
can be replayed and audited later without affecting the online workflow.

V2 platform migration: when ``PLATFORM_ENABLED=1`` the store delegates
every write to :class:`data_platform.compat.PlatformLiveStore` against the
canonical PostgreSQL tables. The SQLite path below remains the legacy
fallback so tests + development flows stay zero-config.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

_PROJECT_ROOT = _SCRIPTS_ROOT.parent
_DEFAULT_LIVE_DB = _PROJECT_ROOT / "Index" / "live.db"


def _is_default_path(db_path: Path) -> bool:
    """True only when ``db_path`` refers to the conventional Index/live.db.

    Tests + custom deployments pass explicit paths; those keep the legacy
    SQLite implementation so test harnesses stay predictable.
    """
    try:
        return Path(db_path).resolve() == _DEFAULT_LIVE_DB.resolve()
    except Exception:
        return False


def _platform_live_store(db_path: Path):
    """Return a PlatformLiveStore when PLATFORM_ENABLED=1 AND the default
    path is in use, else None.
    """
    if not _is_default_path(db_path):
        return None
    try:
        from data_platform.compat import PlatformLiveStore
        from data_platform.compat.live_shim import is_platform_enabled
    except Exception:
        return None
    if not is_platform_enabled():
        return None
    try:
        return PlatformLiveStore()
    except Exception:  # pragma: no cover - DB unreachable
        logger.exception("Platform live store unavailable; falling back to SQLite")
        return None


class LiveStore:
    """SQLite-backed storage for live fixtures, snapshots, and alerts.

    When the V2 platform flag is on the store transparently forwards to
    :class:`PlatformLiveStore`. Callers keep their existing import.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._platform = _platform_live_store(self.db_path)
        if self._platform is None:
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS live_fixtures (
                    fixture_id INTEGER PRIMARY KEY,
                    league TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    kickoff_utc TEXT,
                    created_at TEXT NOT NULL,
                    finished_at TEXT
                );

                CREATE TABLE IF NOT EXISTS live_state_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fixture_id INTEGER NOT NULL,
                    captured_at TEXT NOT NULL,
                    status TEXT,
                    elapsed_min REAL,
                    state_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS live_model_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fixture_id INTEGER NOT NULL,
                    snapshot_ts TEXT NOT NULL,
                    elapsed_min REAL,
                    score TEXT,
                    snapshot_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS live_odds_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fixture_id INTEGER NOT NULL,
                    captured_at TEXT NOT NULL,
                    odds_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS live_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fixture_id INTEGER NOT NULL,
                    snapshot_ts TEXT NOT NULL,
                    emitted_at TEXT NOT NULL,
                    alert_type TEXT,
                    severity TEXT,
                    message TEXT,
                    alert_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_live_state_fixture_time
                    ON live_state_snapshots (fixture_id, captured_at);
                CREATE INDEX IF NOT EXISTS idx_live_model_fixture_time
                    ON live_model_snapshots (fixture_id, snapshot_ts);
                CREATE INDEX IF NOT EXISTS idx_live_odds_fixture_time
                    ON live_odds_snapshots (fixture_id, captured_at);
                CREATE INDEX IF NOT EXISTS idx_live_alerts_fixture_time
                    ON live_alerts (fixture_id, snapshot_ts);
                """
            )

    def ensure_fixture(self, fixture_id: int, league: str, home_team: str, away_team: str, kickoff_utc: Optional[str]) -> None:
        if self._platform is not None:
            return self._platform.ensure_fixture(fixture_id, league, home_team, away_team, kickoff_utc)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO live_fixtures (fixture_id, league, home_team, away_team, kickoff_utc, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(fixture_id) DO UPDATE SET
                    league = excluded.league,
                    home_team = excluded.home_team,
                    away_team = excluded.away_team,
                    kickoff_utc = COALESCE(excluded.kickoff_utc, live_fixtures.kickoff_utc)
                """,
                (fixture_id, league, home_team, away_team, kickoff_utc, _utc_now()),
            )

    def record_state(self, fixture_id: int, state_payload: Dict[str, Any], *, status: Optional[str], elapsed_min: Optional[float]) -> None:
        if self._platform is not None:
            return self._platform.record_state(fixture_id, state_payload, status=status, elapsed_min=elapsed_min)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO live_state_snapshots (fixture_id, captured_at, status, elapsed_min, state_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (fixture_id, _utc_now(), status, elapsed_min, json.dumps(state_payload, default=str)),
            )

    def record_snapshot(self, fixture_id: int, snapshot_payload: Dict[str, Any], *, snapshot_ts: str, elapsed_min: Optional[float], score: Optional[str]) -> None:
        if self._platform is not None:
            return self._platform.record_snapshot(
                fixture_id, snapshot_payload,
                snapshot_ts=snapshot_ts, elapsed_min=elapsed_min, score=score,
            )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO live_model_snapshots (fixture_id, snapshot_ts, elapsed_min, score, snapshot_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (fixture_id, snapshot_ts, elapsed_min, score, json.dumps(snapshot_payload, default=str)),
            )

    def record_live_odds(self, fixture_id: int, odds_payload: Dict[str, Any], *, captured_at: Optional[str] = None) -> None:
        if self._platform is not None:
            return self._platform.record_live_odds(fixture_id, odds_payload, captured_at=captured_at)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO live_odds_snapshots (fixture_id, captured_at, odds_json)
                VALUES (?, ?, ?)
                """,
                (fixture_id, captured_at or _utc_now(), json.dumps(odds_payload, default=str)),
            )

    def record_alerts(self, fixture_id: int, snapshot_ts: str, alerts: Iterable[Dict[str, Any]]) -> None:
        if self._platform is not None:
            return self._platform.record_alerts(fixture_id, snapshot_ts, alerts)
        alert_rows = [
            (
                fixture_id,
                snapshot_ts,
                _utc_now(),
                alert.get("type"),
                alert.get("severity"),
                alert.get("message"),
                json.dumps(alert, default=str),
            )
            for alert in alerts
        ]
        if not alert_rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO live_alerts (fixture_id, snapshot_ts, emitted_at, alert_type, severity, message, alert_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                alert_rows,
            )

    def mark_finished(self, fixture_id: int) -> None:
        if self._platform is not None:
            return self._platform.mark_finished(fixture_id)
        with self._connect() as conn:
            conn.execute(
                "UPDATE live_fixtures SET finished_at = COALESCE(finished_at, ?) WHERE fixture_id = ?",
                (_utc_now(), fixture_id),
            )
