"""Odds snapshot repository. Used by PredictionService for CLV capture and
by the web surfaces when historical odds are needed without hitting the API.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from ..db import session_scope
from ..models import OddsSnapshot

logger = logging.getLogger(__name__)


class OddsSnapshotRepository:
    def __init__(self, session_factory=session_scope):
        self._factory = session_factory

    def record(
        self,
        *,
        event_id: str,
        league: Optional[str],
        home_team: Optional[str],
        away_team: Optional[str],
        kickoff_utc: Optional[datetime],
        kickoff_date: Optional[str],
        odds_json: Mapping[str, Any],
        kind: str = "prematch",
        captured_at: Optional[datetime] = None,
    ) -> Optional[int]:
        try:
            captured = captured_at or datetime.now(timezone.utc)
            with self._factory() as session:
                # Dedup: same (event_id, captured_at, kind) is our unique key
                row = OddsSnapshot(
                    event_id=event_id,
                    league=league,
                    home_team=home_team,
                    away_team=away_team,
                    kickoff_utc=kickoff_utc,
                    kickoff_date=kickoff_date,
                    kind=kind,
                    captured_at=captured,
                    odds_json=dict(odds_json),
                )
                session.add(row)
                session.flush()
                return int(row.id)
        except Exception:
            # Unique-constraint collisions (same event + second + kind) are harmless.
            logger.debug("odds snapshot dedup collision event=%s", event_id, exc_info=True)
            return None

    def latest_for_event(self, event_id: str, *, kind: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._factory() as session:
            stmt = select(OddsSnapshot).where(OddsSnapshot.event_id == event_id)
            if kind is not None:
                stmt = stmt.where(OddsSnapshot.kind == kind)
            stmt = stmt.order_by(desc(OddsSnapshot.captured_at)).limit(1)
            row = session.scalar(stmt)
            return _row_to_dict(row) if row else None


def _row_to_dict(row: OddsSnapshot) -> Dict[str, Any]:
    return {
        "id": row.id,
        "event_id": row.event_id,
        "league": row.league,
        "home_team": row.home_team,
        "away_team": row.away_team,
        "kickoff_utc": row.kickoff_utc.isoformat() if row.kickoff_utc else None,
        "kickoff_date": row.kickoff_date,
        "kind": row.kind,
        "captured_at": row.captured_at.isoformat() if row.captured_at else None,
        "odds": row.odds_json,
    }
