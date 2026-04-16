"""Live runtime repository — parity with ``Scripts/live/live_store.LiveStore``."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from ..db import session_scope
from ..models import LiveAlert, LiveFixture, LiveModelSnapshot, LiveOddsSnapshot, LiveStateSnapshot

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


class LiveRepository:
    def __init__(self, session_factory=session_scope):
        self._factory = session_factory

    def ensure_fixture(
        self,
        *,
        fixture_id: int,
        league: str,
        home_team: str,
        away_team: str,
        kickoff_utc: Optional[str] = None,
    ) -> None:
        kickoff_dt = _parse_iso(kickoff_utc)
        with self._factory() as session:
            row = session.get(LiveFixture, fixture_id)
            if row is None:
                session.add(LiveFixture(
                    fixture_id=fixture_id,
                    league=league,
                    home_team=home_team,
                    away_team=away_team,
                    kickoff_utc=kickoff_dt,
                ))
            else:
                row.league = league
                row.home_team = home_team
                row.away_team = away_team
                if kickoff_dt is not None:
                    row.kickoff_utc = kickoff_dt

    def record_state(
        self,
        *,
        fixture_id: int,
        state_payload: Mapping[str, Any],
        status: Optional[str],
        elapsed_min: Optional[float],
    ) -> None:
        with self._factory() as session:
            session.add(LiveStateSnapshot(
                fixture_id=fixture_id,
                captured_at=_utc_now(),
                status=status,
                elapsed_min=elapsed_min,
                state_json=dict(state_payload),
            ))

    def record_model_snapshot(
        self,
        *,
        fixture_id: int,
        snapshot_payload: Mapping[str, Any],
        snapshot_ts: str,
        elapsed_min: Optional[float],
        score: Optional[str],
    ) -> None:
        with self._factory() as session:
            session.add(LiveModelSnapshot(
                fixture_id=fixture_id,
                snapshot_ts=snapshot_ts,
                elapsed_min=elapsed_min,
                score=score,
                snapshot_json=dict(snapshot_payload),
            ))

    def record_live_odds(
        self,
        *,
        fixture_id: int,
        odds_payload: Mapping[str, Any],
        captured_at: Optional[str] = None,
    ) -> None:
        captured = _parse_iso(captured_at) or _utc_now()
        with self._factory() as session:
            session.add(LiveOddsSnapshot(
                fixture_id=fixture_id,
                captured_at=captured,
                odds_json=dict(odds_payload),
            ))

    def record_alerts(
        self,
        *,
        fixture_id: int,
        snapshot_ts: str,
        alerts: Iterable[Mapping[str, Any]],
    ) -> None:
        rows = list(alerts)
        if not rows:
            return
        with self._factory() as session:
            for alert in rows:
                session.add(LiveAlert(
                    fixture_id=fixture_id,
                    snapshot_ts=snapshot_ts,
                    emitted_at=_utc_now(),
                    alert_type=alert.get("type"),
                    severity=alert.get("severity"),
                    message=alert.get("message"),
                    alert_json=dict(alert),
                ))

    def mark_finished(self, fixture_id: int) -> None:
        with self._factory() as session:
            row = session.get(LiveFixture, fixture_id)
            if row is None:
                return
            if row.finished_at is None:
                row.finished_at = _utc_now()

    # ------------------------------------------------------------------
    # Read side
    # ------------------------------------------------------------------
    def get_active_fixtures(self) -> List[Dict[str, Any]]:
        with self._factory() as session:
            rows = session.scalars(
                select(LiveFixture).where(LiveFixture.finished_at.is_(None))
                .order_by(LiveFixture.kickoff_utc.asc().nullslast())
            ).all()
            return [_fixture_to_dict(r) for r in rows]

    def get_latest_model_snapshot(self, fixture_id: int) -> Optional[Dict[str, Any]]:
        with self._factory() as session:
            row = session.scalar(
                select(LiveModelSnapshot).where(LiveModelSnapshot.fixture_id == fixture_id)
                .order_by(desc(LiveModelSnapshot.snapshot_ts), desc(LiveModelSnapshot.id))
                .limit(1)
            )
            return _model_snapshot_to_dict(row) if row else None

    def get_recent_model_snapshots(self, fixture_id: int, *, limit: int = 20) -> List[Dict[str, Any]]:
        with self._factory() as session:
            rows = session.scalars(
                select(LiveModelSnapshot).where(LiveModelSnapshot.fixture_id == fixture_id)
                .order_by(desc(LiveModelSnapshot.snapshot_ts), desc(LiveModelSnapshot.id))
                .limit(limit)
            ).all()
            return list(reversed([_model_snapshot_to_dict(r) for r in rows]))

    def get_recent_alerts(self, fixture_id: int, *, limit: int = 20) -> List[Dict[str, Any]]:
        with self._factory() as session:
            rows = session.scalars(
                select(LiveAlert).where(LiveAlert.fixture_id == fixture_id)
                .order_by(desc(LiveAlert.emitted_at))
                .limit(limit)
            ).all()
            return [_alert_to_dict(r) for r in rows]


def _fixture_to_dict(row: LiveFixture) -> Dict[str, Any]:
    return {
        "fixture_id": row.fixture_id,
        "league": row.league,
        "home_team": row.home_team,
        "away_team": row.away_team,
        "kickoff_utc": row.kickoff_utc.isoformat() if row.kickoff_utc else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
    }


def _model_snapshot_to_dict(row: LiveModelSnapshot) -> Dict[str, Any]:
    return {
        "fixture_id": row.fixture_id,
        "snapshot_ts": row.snapshot_ts,
        "elapsed_min": float(row.elapsed_min) if row.elapsed_min is not None else None,
        "score": row.score,
        "snapshot": row.snapshot_json,
    }


def _alert_to_dict(row: LiveAlert) -> Dict[str, Any]:
    return {
        "fixture_id": row.fixture_id,
        "snapshot_ts": row.snapshot_ts,
        "emitted_at": row.emitted_at.isoformat() if row.emitted_at else None,
        "type": row.alert_type,
        "severity": row.severity,
        "message": row.message,
        "data": row.alert_json,
    }
