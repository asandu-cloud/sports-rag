"""Shim for the legacy ``ParlaySessionStore``.

``ParlaySessionStore`` in ``Scripts/rag_ingest/core/parlay_service.py``
decides at instantiation whether to fall back to SQLite or route through
the canonical store via ``PlatformParlayStore`` below.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from .. import config as _config
from ..db import session_scope
from ..models import ParlaySession
from ..services.parlays import ParlayService

logger = logging.getLogger(__name__)


def is_platform_enabled() -> bool:
    return _config.SETTINGS.platform_enabled


@lru_cache(maxsize=1)
def get_parlay_service() -> ParlayService:
    return ParlayService()


class PlatformParlayStore:
    """Drop-in replacement for ``ParlaySessionStore`` backed by Postgres.

    Used by the legacy shim when ``PLATFORM_ENABLED=1``. The method
    signatures mirror the SQLite store exactly so callers (discord bot
    parlay_builder cog) remain untouched.
    """

    def __init__(self, service: Optional[ParlayService] = None):
        self._service = service or get_parlay_service()

    def create_session(self, request: Any, result: Any) -> int:
        session_id = self._service.create_session(request=request, result=result)
        return session_id

    def update_message_context(
        self,
        session_id: int,
        *,
        message_id: Optional[int] = None,
        channel_id: Optional[int] = None,
        thread_id: Optional[int] = None,
    ) -> None:
        self._service.update_message_context(
            session_id,
            message_id=message_id,
            channel_id=channel_id,
            thread_id=thread_id,
        )

    def get_session(self, session_id: int) -> Any:
        record_dict = self._service.get_session(session_id)
        if record_dict is None:
            return None
        return _record_from_dict(record_dict)

    def list_persistent_sessions(self, limit: int = 250) -> List[Any]:
        rows = self._service.list_persistent_sessions(limit=limit)
        return [_record_from_dict(r) for r in rows]


def _record_from_dict(row: Dict[str, Any]) -> Any:
    """Materialize a ParlaySessionRecord in the same way the legacy code did."""
    # Lazy import to avoid a circular dep with core.parlay_models
    try:
        from core.parlay_models import ParlaySessionRecord
    except Exception:
        from Scripts.rag_ingest.core.parlay_models import ParlaySessionRecord  # type: ignore
    payload = {
        "id": row["id"],
        "parent_session_id": row.get("parent_session_id"),
        "discord_user_id": row.get("discord_user_id"),
        "guild_id": row.get("guild_id"),
        "channel_id": row.get("channel_id"),
        "thread_id": row.get("thread_id"),
        "message_id": row.get("message_id"),
        "source_command": row.get("source_command"),
        "action_type": row.get("action_type"),
        "request_json": row.get("request_json"),
        "result_json": row.get("result_json"),
        "created_at": row.get("created_at"),
    }
    return ParlaySessionRecord.from_row(payload)


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def migrate_sqlite_parlay_sessions(
    sqlite_path: Optional[Path] = None,
    *,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Copy parlay_sessions + (optional) trial tables from SQLite into Postgres."""
    sqlite_path = Path(sqlite_path) if sqlite_path else _config.SETTINGS.project_root / "Index" / "predictions.db"
    if not sqlite_path.exists():
        return {"parlay_sessions": 0, "trial_users": 0, "trial_parlays": 0}

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        try:
            sessions = [dict(r) for r in conn.execute("SELECT * FROM parlay_sessions")]
        except sqlite3.OperationalError:
            sessions = []
        try:
            trial_users = [dict(r) for r in conn.execute("SELECT * FROM trial_users")]
        except sqlite3.OperationalError:
            trial_users = []
        try:
            trial_parlays = [dict(r) for r in conn.execute("SELECT * FROM trial_parlays")]
        except sqlite3.OperationalError:
            trial_parlays = []
    finally:
        conn.close()

    if dry_run:
        return {
            "parlay_sessions": len(sessions),
            "trial_users": len(trial_users),
            "trial_parlays": len(trial_parlays),
            "dry_run": True,
        }

    counts = {"parlay_sessions": 0, "trial_users": 0, "trial_parlays": 0}
    import json as _json
    from ..models import TrialParlay, TrialUser

    with session_scope() as session:
        for row in sessions:
            try:
                request_json = _json.loads(row.get("request_json") or "null")
            except _json.JSONDecodeError:
                request_json = {"_raw": row.get("request_json")}
            try:
                result_json = _json.loads(row.get("result_json") or "null")
            except _json.JSONDecodeError:
                result_json = {"_raw": row.get("result_json")}
            session.add(ParlaySession(
                parent_session_id=row.get("parent_session_id"),
                discord_user_id=row.get("discord_user_id"),
                guild_id=row.get("guild_id"),
                channel_id=row.get("channel_id"),
                thread_id=row.get("thread_id"),
                message_id=row.get("message_id"),
                source_command=row.get("source_command") or "parlay",
                action_type=row.get("action_type") or "build",
                request_json=request_json,
                result_json=result_json,
            ))
            counts["parlay_sessions"] += 1

        for row in trial_users:
            discord_id = row.get("discord_id")
            if not discord_id:
                continue
            existing = session.scalar(select(TrialUser).where(TrialUser.discord_id == discord_id))
            fields = {
                "discord_username": row.get("discord_username"),
                "activated_at": _parse_dt(row.get("activated_at")),
                "hits": int(row.get("hits") or 0),
                "total_parlays_seen": int(row.get("total_parlays_seen") or 0),
                "total_profit": float(row.get("total_profit") or 0.0),
                "graduated": int(row.get("graduated") or 0),
                "graduated_at": _parse_dt(row.get("graduated_at")),
            }
            if existing is None:
                session.add(TrialUser(discord_id=discord_id, **fields))
            else:
                for key, value in fields.items():
                    setattr(existing, key, value)
            counts["trial_users"] += 1

        for row in trial_parlays:
            try:
                legs = _json.loads(row.get("legs") or "[]")
            except _json.JSONDecodeError:
                legs = []
            session.add(TrialParlay(
                posted_at=_parse_dt(row.get("posted_at")) or _parse_dt_fallback(row.get("matchday_date")),
                matchday_date=_parse_date(row.get("matchday_date")),
                legs=legs,
                combined_odds=float(row.get("combined_odds") or 0),
                outcome=row.get("outcome"),
                resolved_at=_parse_dt(row.get("resolved_at")),
                stake=float(row.get("stake") or 10.0),
                profit=row.get("profit"),
            ))
            counts["trial_parlays"] += 1
    return counts


def _parse_date(value: Any):
    from datetime import date as _date
    if not value:
        return None
    if isinstance(value, _date):
        return value
    try:
        return _date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _parse_dt(value: Any):
    from datetime import datetime as _dt, timezone as _tz
    if not value:
        return None
    if isinstance(value, _dt):
        return value if value.tzinfo else value.replace(tzinfo=_tz.utc)
    try:
        return _dt.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        try:
            return _dt.strptime(str(value), "%Y-%m-%d %H:%M:%S").replace(tzinfo=_tz.utc)
        except (ValueError, TypeError):
            return None


def _parse_dt_fallback(value: Any):
    from datetime import datetime as _dt, timezone as _tz
    d = _parse_date(value)
    if d is None:
        return _dt.now(_tz.utc)
    return _dt(d.year, d.month, d.day, tzinfo=_tz.utc)
