"""Shim for the legacy ``Scripts/live/live_store.LiveStore``."""

from __future__ import annotations

import logging
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from .. import config as _config
from ..db import session_scope
from ..models import LiveAlert, LiveFixture, LiveModelSnapshot, LiveOddsSnapshot, LiveStateSnapshot
from ..services.live import LiveService

logger = logging.getLogger(__name__)


def is_platform_enabled() -> bool:
    return _config.SETTINGS.platform_enabled


@lru_cache(maxsize=1)
def get_live_service() -> LiveService:
    return LiveService()


class PlatformLiveStore:
    """Drop-in replacement for ``LiveStore`` backed by Postgres."""

    def __init__(self, service: Optional[LiveService] = None):
        self._service = service or get_live_service()

    def ensure_fixture(self, fixture_id: int, league: str, home_team: str, away_team: str, kickoff_utc: Optional[str]) -> None:
        self._service.ensure_fixture(
            fixture_id=fixture_id, league=league,
            home_team=home_team, away_team=away_team, kickoff_utc=kickoff_utc,
        )

    def record_state(
        self,
        fixture_id: int,
        state_payload: Mapping[str, Any],
        *,
        status: Optional[str],
        elapsed_min: Optional[float],
    ) -> None:
        self._service.record_state(
            fixture_id=fixture_id, state_payload=state_payload,
            status=status, elapsed_min=elapsed_min,
        )

    def record_snapshot(
        self,
        fixture_id: int,
        snapshot_payload: Mapping[str, Any],
        *,
        snapshot_ts: str,
        elapsed_min: Optional[float],
        score: Optional[str],
    ) -> None:
        self._service.record_model_snapshot(
            fixture_id=fixture_id, snapshot_payload=snapshot_payload,
            snapshot_ts=snapshot_ts, elapsed_min=elapsed_min, score=score,
        )

    def record_live_odds(
        self,
        fixture_id: int,
        odds_payload: Mapping[str, Any],
        *,
        captured_at: Optional[str] = None,
    ) -> None:
        self._service.record_live_odds(
            fixture_id=fixture_id, odds_payload=odds_payload, captured_at=captured_at,
        )

    def record_alerts(self, fixture_id: int, snapshot_ts: str, alerts: Iterable[Mapping[str, Any]]) -> None:
        self._service.record_alerts(
            fixture_id=fixture_id, snapshot_ts=snapshot_ts, alerts=alerts,
        )

    def mark_finished(self, fixture_id: int) -> None:
        self._service.mark_finished(fixture_id)


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def migrate_sqlite_live(
    sqlite_path: Optional[Path] = None,
    *,
    dry_run: bool = False,
    keep_rows_per_fixture: Optional[int] = None,
) -> Dict[str, int]:
    """Copy live.db contents into the canonical tables.

    ``keep_rows_per_fixture`` is applied to the high-volume snapshot
    tables so operators can migrate only the most recent history if
    ``live.db`` has grown large.
    """
    sqlite_path = Path(sqlite_path) if sqlite_path else _config.SETTINGS.project_root / "Index" / "live.db"
    if not sqlite_path.exists():
        return {"fixtures": 0, "state_snapshots": 0, "model_snapshots": 0, "odds_snapshots": 0, "alerts": 0}

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        fixtures = _safe_fetch(conn, "SELECT * FROM live_fixtures")
        state_rows = _safe_fetch(conn, "SELECT * FROM live_state_snapshots ORDER BY id DESC")
        model_rows = _safe_fetch(conn, "SELECT * FROM live_model_snapshots ORDER BY id DESC")
        odds_rows = _safe_fetch(conn, "SELECT * FROM live_odds_snapshots ORDER BY id DESC")
        alert_rows = _safe_fetch(conn, "SELECT * FROM live_alerts ORDER BY id DESC")
    finally:
        conn.close()

    if keep_rows_per_fixture is not None:
        state_rows = _truncate_per_fixture(state_rows, keep_rows_per_fixture)
        model_rows = _truncate_per_fixture(model_rows, keep_rows_per_fixture)
        odds_rows = _truncate_per_fixture(odds_rows, keep_rows_per_fixture)
        alert_rows = _truncate_per_fixture(alert_rows, keep_rows_per_fixture)

    counts = {
        "fixtures": len(fixtures),
        "state_snapshots": len(state_rows),
        "model_snapshots": len(model_rows),
        "odds_snapshots": len(odds_rows),
        "alerts": len(alert_rows),
    }
    if dry_run:
        counts["dry_run"] = True
        return counts

    import json as _json
    with session_scope() as session:
        for row in fixtures:
            if row.get("fixture_id") is None:
                continue
            existing = session.get(LiveFixture, row["fixture_id"])
            data = {
                "league": row.get("league") or "",
                "home_team": row.get("home_team") or "",
                "away_team": row.get("away_team") or "",
                "kickoff_utc": _parse_dt(row.get("kickoff_utc")),
                "finished_at": _parse_dt(row.get("finished_at")),
            }
            if existing is None:
                session.add(LiveFixture(fixture_id=row["fixture_id"], **data))
            else:
                for key, value in data.items():
                    setattr(existing, key, value)

        for row in state_rows:
            session.add(LiveStateSnapshot(
                fixture_id=row.get("fixture_id"),
                captured_at=_parse_dt(row.get("captured_at")),
                status=row.get("status"),
                elapsed_min=row.get("elapsed_min"),
                state_json=_safe_json(row.get("state_json")),
            ))
        for row in model_rows:
            session.add(LiveModelSnapshot(
                fixture_id=row.get("fixture_id"),
                snapshot_ts=row.get("snapshot_ts") or "",
                elapsed_min=row.get("elapsed_min"),
                score=row.get("score"),
                snapshot_json=_safe_json(row.get("snapshot_json")),
            ))
        for row in odds_rows:
            session.add(LiveOddsSnapshot(
                fixture_id=row.get("fixture_id"),
                captured_at=_parse_dt(row.get("captured_at")),
                odds_json=_safe_json(row.get("odds_json")),
            ))
        for row in alert_rows:
            session.add(LiveAlert(
                fixture_id=row.get("fixture_id"),
                snapshot_ts=row.get("snapshot_ts") or "",
                emitted_at=_parse_dt(row.get("emitted_at")),
                alert_type=row.get("alert_type"),
                severity=row.get("severity"),
                message=row.get("message"),
                alert_json=_safe_json(row.get("alert_json")),
            ))
    return counts


def _safe_fetch(conn: sqlite3.Connection, sql: str) -> list:
    try:
        return [dict(r) for r in conn.execute(sql)]
    except sqlite3.OperationalError:
        return []


def _truncate_per_fixture(rows, keep: int):
    out = []
    seen: Dict[Any, int] = {}
    for row in rows:
        fid = row.get("fixture_id")
        seen[fid] = seen.get(fid, 0) + 1
        if seen[fid] <= keep:
            out.append(row)
    return out


def _parse_dt(value):
    from datetime import datetime as _dt, timezone as _tz
    if not value:
        return None
    if isinstance(value, _dt):
        return value if value.tzinfo else value.replace(tzinfo=_tz.utc)
    try:
        return _dt.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _safe_json(value):
    import json as _json
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return _json.loads(value)
    except (TypeError, ValueError):
        return {"_raw": str(value)}
