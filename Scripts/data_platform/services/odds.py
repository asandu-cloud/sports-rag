"""OddsService — persistence wrapper around OddsSnapshotRepository."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Mapping, Optional

from ..repositories.odds import OddsSnapshotRepository


class OddsService:
    def __init__(self, repo: Optional[OddsSnapshotRepository] = None):
        self._repo = repo or OddsSnapshotRepository()

    def record_prematch(
        self,
        *,
        event_id: str,
        league: Optional[str],
        home_team: Optional[str],
        away_team: Optional[str],
        kickoff_utc: Optional[datetime],
        odds: Mapping[str, Any],
        captured_at: Optional[datetime] = None,
    ) -> Optional[int]:
        kickoff_date = kickoff_utc.date().isoformat() if kickoff_utc else None
        return self._repo.record(
            event_id=event_id,
            league=league,
            home_team=home_team,
            away_team=away_team,
            kickoff_utc=kickoff_utc,
            kickoff_date=kickoff_date,
            odds_json=odds,
            kind="prematch",
            captured_at=captured_at,
        )

    def record_closing(self, *, event_id: str, odds: Mapping[str, Any], captured_at: Optional[datetime] = None, **kw) -> Optional[int]:
        return self._repo.record(
            event_id=event_id,
            kind="closing",
            odds_json=odds,
            captured_at=captured_at,
            league=kw.get("league"),
            home_team=kw.get("home_team"),
            away_team=kw.get("away_team"),
            kickoff_utc=kw.get("kickoff_utc"),
            kickoff_date=kw.get("kickoff_date"),
        )

    def latest(self, event_id: str, *, kind: Optional[str] = None) -> Optional[Dict[str, Any]]:
        return self._repo.latest_for_event(event_id, kind=kind)
