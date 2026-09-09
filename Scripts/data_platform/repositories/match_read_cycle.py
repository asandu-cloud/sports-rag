"""Storage primitives for scheduled Match Read cycles.

The repository intentionally exposes small transactional operations rather
than a long-running session: a worker may make API/model calls for several
seconds, and no database transaction should be held open while it does that.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from sqlalchemy import desc, select
from sqlalchemy.exc import IntegrityError

from ..db import session_scope
from ..models import MatchReadObservation, WorkerLease


SUCCESSFUL_OBSERVATION_STATUSES = frozenset({"verified", "unchanged"})


def _as_utc(value: Optional[Any] = None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return _as_utc(value).isoformat()


def _observation_to_dict(row: MatchReadObservation) -> Dict[str, Any]:
    return {
        "id": int(row.id),
        "sync_run_id": int(row.sync_run_id) if row.sync_run_id is not None else None,
        "match_read_id": int(row.match_read_id) if row.match_read_id is not None else None,
        "fixture_api_id": row.fixture_api_id,
        "league": row.league,
        "stage": row.stage,
        "checked_at": _iso(row.checked_at),
        "status": row.status,
        "source": row.source,
        "input_snapshot_id": row.input_snapshot_id,
        "read_key": row.read_key,
        "provider_calls": dict(row.provider_calls_json) if isinstance(row.provider_calls_json, Mapping) else {},
        "detail": dict(row.detail_json) if isinstance(row.detail_json, Mapping) else {},
        "error": row.error_text,
    }


class MatchReadCycleRepository:
    """Append-only observation history and short-lived worker leases."""

    def __init__(self, session_factory=session_scope):
        self._factory = session_factory

    def acquire_lease(
        self,
        *,
        lease_key: str,
        owner_id: str,
        ttl_seconds: int,
        now: Optional[Any] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """Claim a lease only when it is absent, expired, or already ours.

        The unique key plus the integrity-error fallback covers the race where
        two schedulers both see an absent row. PostgreSQL serialises the row
        update normally; SQLite's write lock yields the same safe loser/retry
        behaviour for the local worker.
        """
        acquired_at = _as_utc(now)
        expires_at = acquired_at + timedelta(seconds=max(1, int(ttl_seconds)))
        key = str(lease_key or "").strip()
        owner = str(owner_id or "").strip()
        if not key or not owner:
            raise ValueError("lease_key and owner_id are required.")

        try:
            with self._factory() as session:
                row = session.scalar(select(WorkerLease).where(WorkerLease.lease_key == key))
                if row is not None:
                    expires = _as_utc(row.expires_at)
                    if row.owner_id != owner and expires > acquired_at:
                        return False
                    row.owner_id = owner
                    row.acquired_at = acquired_at
                    row.heartbeat_at = acquired_at
                    row.expires_at = expires_at
                    row.metadata_json = dict(metadata) if metadata else None
                else:
                    session.add(WorkerLease(
                        lease_key=key,
                        owner_id=owner,
                        acquired_at=acquired_at,
                        heartbeat_at=acquired_at,
                        expires_at=expires_at,
                        metadata_json=dict(metadata) if metadata else None,
                    ))
                session.flush()
                return True
        except IntegrityError:
            # A concurrent first insert won. Its lease is authoritative; this
            # one-shot invocation can safely exit and let the next schedule
            # tick run normally.
            return False

    def release_lease(
        self,
        *,
        lease_key: str,
        owner_id: str,
        now: Optional[Any] = None,
    ) -> bool:
        released_at = _as_utc(now)
        with self._factory() as session:
            row = session.scalar(select(WorkerLease).where(WorkerLease.lease_key == str(lease_key)))
            if row is None or row.owner_id != str(owner_id):
                return False
            row.heartbeat_at = released_at
            row.expires_at = released_at
            return True

    def record_observation(
        self,
        *,
        fixture_api_id: str,
        league: str,
        stage: str,
        status: str,
        checked_at: Optional[Any] = None,
        sync_run_id: Optional[int] = None,
        match_read_id: Optional[int] = None,
        source: str = "api_football",
        input_snapshot_id: Optional[str] = None,
        read_key: Optional[str] = None,
        provider_calls: Optional[Mapping[str, Any]] = None,
        detail: Optional[Mapping[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        fixture = str(fixture_api_id or "").strip()
        competition = str(league or "").strip()
        stage_name = str(stage or "").strip().lower()
        state = str(status or "").strip().lower()
        if not fixture or not competition or not stage_name or not state:
            raise ValueError("fixture_api_id, league, stage, and status are required.")
        with self._factory() as session:
            row = MatchReadObservation(
                sync_run_id=int(sync_run_id) if sync_run_id is not None else None,
                match_read_id=int(match_read_id) if match_read_id is not None else None,
                fixture_api_id=fixture,
                league=competition,
                stage=stage_name,
                checked_at=_as_utc(checked_at),
                status=state,
                source=str(source or "api_football").strip() or "api_football",
                input_snapshot_id=str(input_snapshot_id).strip() if input_snapshot_id else None,
                read_key=str(read_key).strip() if read_key else None,
                provider_calls_json=dict(provider_calls) if provider_calls else None,
                detail_json=dict(detail) if detail else None,
                error_text=str(error).strip() if error else None,
            )
            session.add(row)
            session.flush()
            return _observation_to_dict(row)

    def latest_for_fixture_stages(
        self,
        fixture_ids: Iterable[str],
        *,
        stages: Sequence[str] = ("pre_match", "confirmed_lineups"),
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        ids = tuple(sorted({str(value).strip() for value in fixture_ids if str(value).strip()}))
        stage_names = tuple(sorted({str(value).strip().lower() for value in stages if str(value).strip()}))
        if not ids or not stage_names:
            return {}
        with self._factory() as session:
            rows = session.scalars(
                select(MatchReadObservation)
                .where(
                    MatchReadObservation.fixture_api_id.in_(ids),
                    MatchReadObservation.stage.in_(stage_names),
                )
                .order_by(desc(MatchReadObservation.checked_at), desc(MatchReadObservation.id))
            ).all()
        latest: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in rows:
            key = (str(row.fixture_api_id), str(row.stage))
            latest.setdefault(key, _observation_to_dict(row))
        return latest

    def latest_successful_by_match_read_ids(
        self,
        match_read_ids: Iterable[int],
    ) -> Dict[int, datetime]:
        """Return the newest verified worker check for each immutable read."""
        ids = tuple(sorted({int(value) for value in match_read_ids if value is not None}))
        if not ids:
            return {}
        with self._factory() as session:
            rows = session.scalars(
                select(MatchReadObservation)
                .where(
                    MatchReadObservation.match_read_id.in_(ids),
                    MatchReadObservation.status.in_(tuple(SUCCESSFUL_OBSERVATION_STATUSES)),
                )
                .order_by(desc(MatchReadObservation.checked_at), desc(MatchReadObservation.id))
            ).all()
        result: Dict[int, datetime] = {}
        for row in rows:
            if row.match_read_id is not None:
                result.setdefault(int(row.match_read_id), _as_utc(row.checked_at))
        return result

    def active_lease(self, lease_key: str, *, now: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        reference = _as_utc(now)
        with self._factory() as session:
            row = session.scalar(select(WorkerLease).where(WorkerLease.lease_key == str(lease_key)))
            if row is None or _as_utc(row.expires_at) <= reference:
                return None
            return {
                "lease_key": row.lease_key,
                "owner_id": row.owner_id,
                "acquired_at": _iso(row.acquired_at),
                "heartbeat_at": _iso(row.heartbeat_at),
                "expires_at": _iso(row.expires_at),
            }
