"""LiveService — wraps the live repository for API + live bot consumers."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

from ..repositories.live import LiveRepository

logger = logging.getLogger(__name__)


class LiveService:
    def __init__(self, repo: Optional[LiveRepository] = None):
        self._repo = repo or LiveRepository()

    def ensure_fixture(self, **kwargs: Any) -> None:
        self._repo.ensure_fixture(**kwargs)

    def record_state(self, **kwargs: Any) -> None:
        self._repo.record_state(**kwargs)

    def record_model_snapshot(self, **kwargs: Any) -> None:
        self._repo.record_model_snapshot(**kwargs)

    def record_live_odds(self, **kwargs: Any) -> None:
        self._repo.record_live_odds(**kwargs)

    def record_alerts(self, **kwargs: Any) -> None:
        self._repo.record_alerts(**kwargs)

    def mark_finished(self, fixture_id: int) -> None:
        self._repo.mark_finished(fixture_id)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def list_active(self) -> List[Dict[str, Any]]:
        return self._repo.get_active_fixtures()

    def latest_snapshot(self, fixture_id: int) -> Optional[Dict[str, Any]]:
        return self._repo.get_latest_model_snapshot(fixture_id)

    def snapshot_history(self, fixture_id: int, *, limit: int = 20) -> List[Dict[str, Any]]:
        return self._repo.get_recent_model_snapshots(fixture_id, limit=limit)

    def recent_alerts(self, fixture_id: int, *, limit: int = 20) -> List[Dict[str, Any]]:
        return self._repo.get_recent_alerts(fixture_id, limit=limit)
