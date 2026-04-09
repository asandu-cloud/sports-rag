"""Persistence helpers for the live runtime.

Stores raw live state, model snapshots, and emitted alerts so the live stack
can be replayed and audited later without affecting the online workflow.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LiveStore:
    """SQLite-backed storage for live fixtures, snapshots, and alerts."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
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
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO live_state_snapshots (fixture_id, captured_at, status, elapsed_min, state_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (fixture_id, _utc_now(), status, elapsed_min, json.dumps(state_payload, default=str)),
            )

    def record_snapshot(self, fixture_id: int, snapshot_payload: Dict[str, Any], *, snapshot_ts: str, elapsed_min: Optional[float], score: Optional[str]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO live_model_snapshots (fixture_id, snapshot_ts, elapsed_min, score, snapshot_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (fixture_id, snapshot_ts, elapsed_min, score, json.dumps(snapshot_payload, default=str)),
            )

    def record_live_odds(self, fixture_id: int, odds_payload: Dict[str, Any], *, captured_at: Optional[str] = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO live_odds_snapshots (fixture_id, captured_at, odds_json)
                VALUES (?, ?, ?)
                """,
                (fixture_id, captured_at or _utc_now(), json.dumps(odds_payload, default=str)),
            )

    def record_alerts(self, fixture_id: int, snapshot_ts: str, alerts: Iterable[Dict[str, Any]]) -> None:
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
        with self._connect() as conn:
            conn.execute(
                "UPDATE live_fixtures SET finished_at = COALESCE(finished_at, ?) WHERE fixture_id = ?",
                (_utc_now(), fixture_id),
            )
