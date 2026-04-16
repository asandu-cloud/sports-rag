"""ParlaySession repository. Mirrors legacy ``ParlaySessionStore`` API."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from ..db import session_scope
from ..models import ParlaySession

logger = logging.getLogger(__name__)


def _row_to_legacy_dict(row: ParlaySession) -> Dict[str, Any]:
    """Map a ParlaySession ORM row to the dict shape legacy code expects."""
    return {
        "id": row.id,
        "parent_session_id": row.parent_session_id,
        "discord_user_id": row.discord_user_id,
        "guild_id": row.guild_id,
        "channel_id": row.channel_id,
        "thread_id": row.thread_id,
        "message_id": row.message_id,
        "source_command": row.source_command,
        "action_type": row.action_type,
        "request_json": row.request_json,
        "result_json": row.result_json,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


class ParlayRepository:
    def __init__(self, session_factory=session_scope):
        self._factory = session_factory

    def create(
        self,
        *,
        parent_session_id: Optional[int],
        discord_user_id: Optional[int],
        guild_id: Optional[int],
        channel_id: Optional[int],
        thread_id: Optional[int],
        source_command: str,
        action_type: str,
        request_json: Any,
        result_json: Any,
    ) -> int:
        request_dict = _coerce_json(request_json)
        result_dict = _coerce_json(result_json)
        with self._factory() as session:
            row = ParlaySession(
                parent_session_id=parent_session_id,
                discord_user_id=discord_user_id,
                guild_id=guild_id,
                channel_id=channel_id,
                thread_id=thread_id,
                message_id=None,
                source_command=source_command,
                action_type=action_type,
                request_json=request_dict,
                result_json=result_dict,
            )
            session.add(row)
            session.flush()
            new_id = int(row.id)
        return new_id

    def update_result(self, session_id: int, *, result_json: Any) -> None:
        payload = _coerce_json(result_json)
        with self._factory() as session:
            row = session.get(ParlaySession, session_id)
            if row is None:
                return
            row.result_json = payload

    def update_message_context(
        self,
        session_id: int,
        *,
        message_id: Optional[int] = None,
        channel_id: Optional[int] = None,
        thread_id: Optional[int] = None,
    ) -> None:
        with self._factory() as session:
            row = session.get(ParlaySession, session_id)
            if row is None:
                return
            if message_id is not None:
                row.message_id = message_id
            if channel_id is not None:
                row.channel_id = channel_id
            if thread_id is not None:
                row.thread_id = thread_id

    def get(self, session_id: int) -> Optional[Dict[str, Any]]:
        with self._factory() as session:
            row = session.get(ParlaySession, session_id)
            return _row_to_legacy_dict(row) if row is not None else None

    def list_persistent(self, *, limit: int = 250) -> List[Dict[str, Any]]:
        with self._factory() as session:
            rows = session.scalars(
                select(ParlaySession)
                .where(ParlaySession.message_id.isnot(None))
                .order_by(desc(ParlaySession.id))
                .limit(limit)
            ).all()
            return [_row_to_legacy_dict(r) for r in rows]


def _coerce_json(value: Any) -> Any:
    """Accept dicts, dataclasses-turned-dicts, or JSON strings; return JSON-serializable."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"_raw": value}
    # Objects with to_dict()
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    # Fallback — best-effort dict conversion
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return {"_repr": repr(value)}
