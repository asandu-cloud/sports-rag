"""ParlayService — orchestrates session storage.

Build logic still lives in ``Scripts/rag_ingest/core/parlay_service.build_parlay``;
the service here just handles persistence so the engine isn't coupled to Postgres.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..repositories.parlays import ParlayRepository

logger = logging.getLogger(__name__)


class ParlayService:
    def __init__(self, repo: Optional[ParlayRepository] = None):
        self._repo = repo or ParlayRepository()

    def create_session(
        self,
        *,
        request: Any,
        result: Any,
    ) -> int:
        request_dict = _dict_from(request)
        result_dict = _dict_from(result)
        # Strip the session_id field off the result dict before first write —
        # the repo assigns it on insert.
        session_id = self._repo.create(
            parent_session_id=_optional_int(request_dict.get("parent_session_id")),
            discord_user_id=_optional_int(request_dict.get("discord_user_id")),
            guild_id=_optional_int(request_dict.get("guild_id")),
            channel_id=_optional_int(request_dict.get("channel_id")),
            thread_id=_optional_int(request_dict.get("thread_id")),
            source_command=str(request_dict.get("source_command") or "parlay"),
            action_type=str(request_dict.get("action_type") or "build"),
            request_json=request_dict,
            result_json=result_dict,
        )
        # Legacy contract: after assigning session_id, persist the updated result.
        if isinstance(result, object) and hasattr(result, "session_id"):
            try:
                setattr(result, "session_id", session_id)
                self._repo.update_result(session_id, result_json=_dict_from(result))
            except Exception:
                logger.debug("could not backfill session_id on result", exc_info=True)
        return session_id

    def update_message_context(
        self,
        session_id: int,
        *,
        message_id: Optional[int] = None,
        channel_id: Optional[int] = None,
        thread_id: Optional[int] = None,
    ) -> None:
        self._repo.update_message_context(
            session_id,
            message_id=message_id,
            channel_id=channel_id,
            thread_id=thread_id,
        )

    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        return self._repo.get(session_id)

    def list_persistent_sessions(self, *, limit: int = 250) -> List[Dict[str, Any]]:
        return self._repo.list_persistent(limit=limit)


def _dict_from(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    import json as _json
    try:
        return _json.loads(_json.dumps(value, default=str))
    except Exception:
        return {"_repr": repr(value)}


def _optional_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
