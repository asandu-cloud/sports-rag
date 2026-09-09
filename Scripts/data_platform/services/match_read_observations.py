"""Service boundary for Match Read refresh observations.

Keeping this thin wrapper lets delivery/read surfaces ask one clear question:
"has this immutable card been successfully rechecked recently?"  They do not
need to know the scheduler's table layout or treat an old immutable timestamp
as a reason to create a fake new Match Read version.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ..repositories.match_read_cycle import MatchReadCycleRepository


class MatchReadObservationService:
    def __init__(self, repo: Optional[MatchReadCycleRepository] = None) -> None:
        self._repo = repo or MatchReadCycleRepository()

    def record(self, **kwargs: Any) -> Dict[str, Any]:
        return self._repo.record_observation(**kwargs)

    def latest_for_fixture_stages(
        self,
        fixture_ids: Iterable[str],
        *,
        stages: Sequence[str] = ("pre_match", "confirmed_lineups"),
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        return self._repo.latest_for_fixture_stages(fixture_ids, stages=stages)

    def latest_successful_by_match_read_ids(self, match_read_ids: Iterable[int]) -> Dict[int, datetime]:
        return self._repo.latest_successful_by_match_read_ids(match_read_ids)
