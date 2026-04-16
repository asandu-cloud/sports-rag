"""FixtureService — read-side convenience on top of FixtureRepository."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from ..repositories.fixtures import FixtureRepository


class FixtureService:
    def __init__(self, repo: Optional[FixtureRepository] = None):
        self._repo = repo or FixtureRepository()

    def by_date(self, *, league: Optional[str] = None, target_date: Optional[date] = None) -> List[Dict[str, Any]]:
        return self._repo.list_by_date(league=league, target_date=target_date)

    def by_api_id(self, api_football_id: int) -> Optional[Dict[str, Any]]:
        return self._repo.get_by_api_id(api_football_id)

    def team_stats(self, fixture_id: int) -> List[Dict[str, Any]]:
        return self._repo.team_stats_for_fixture(fixture_id)

    def recent_team_stats(self, *, team_api_id: int, limit: int = 6, before: Optional[datetime] = None) -> List[Dict[str, Any]]:
        return self._repo.recent_team_stats(team_api_id=team_api_id, limit=limit, before=before)
