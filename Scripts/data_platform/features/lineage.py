"""Lineage helpers — start/finish a feature build and get back an id."""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Iterable, Iterator, Optional

from sqlalchemy.orm import Session

from ..db import session_scope
from ..models import FeatureBuild
from .versions import CURRENT_FEATURE_VERSION, feature_version_for

logger = logging.getLogger(__name__)


def _hash_inputs(obj: Any) -> str:
    body = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


@dataclass
class FeatureBuildContext:
    """Handle passed around feature builders so they can stamp lineage."""

    build_id: int
    feature_version: str
    scope: str
    as_of_date: date
    pipeline: str
    started_at: datetime

    def stamp(self, *, row_count: int = 0, extras: Optional[dict] = None) -> None:
        """Convenience alias for updating row_count + notes post-build."""
        _update_build(self.build_id, row_count=row_count, extras=extras)


def start_feature_build(
    *,
    scope: str,
    as_of_date: date,
    pipeline: str = "default",
    feature_version: Optional[str] = None,
    sync_run_id: Optional[int] = None,
    upstream_payload_ids: Optional[Iterable[int]] = None,
    inputs: Optional[Any] = None,
    notes: Optional[str] = None,
) -> FeatureBuildContext:
    """Open a new feature build row. Idempotent on (scope, as_of_date, version).

    If a completed build already exists for the same key we just return
    its id — callers can choose whether to skip or rebuild.
    """
    version = feature_version or feature_version_for(pipeline)
    started = datetime.now(timezone.utc)
    payload_ids = list(upstream_payload_ids or [])
    inputs_digest = _hash_inputs(inputs) if inputs is not None else None
    with session_scope() as session:
        existing = (
            session.query(FeatureBuild)
            .filter(
                FeatureBuild.scope == scope,
                FeatureBuild.as_of_date == as_of_date,
                FeatureBuild.feature_version == version,
            )
            .order_by(FeatureBuild.id.desc())
            .first()
        )
        if existing is not None and existing.status in {"completed", "running"}:
            build_id = int(existing.id)
            logger.info("Reusing existing feature build id=%s for %s %s v=%s",
                        build_id, scope, as_of_date, version)
            return FeatureBuildContext(
                build_id=build_id,
                feature_version=version,
                scope=scope,
                as_of_date=as_of_date,
                pipeline=pipeline,
                started_at=existing.started_at,
            )
        row = FeatureBuild(
            feature_version=version,
            scope=scope,
            as_of_date=as_of_date,
            pipeline=pipeline,
            started_at=started,
            sync_run_id=sync_run_id,
            upstream_payload_ids=payload_ids,
            inputs_digest=inputs_digest,
            notes=notes,
            status="running",
            row_count=0,
        )
        session.add(row)
        session.flush()
        build_id = int(row.id)
    return FeatureBuildContext(
        build_id=build_id,
        feature_version=version,
        scope=scope,
        as_of_date=as_of_date,
        pipeline=pipeline,
        started_at=started,
    )


def finish_feature_build(build_id: int, *, status: str = "completed", row_count: Optional[int] = None,
                         notes: Optional[str] = None, error: Optional[str] = None) -> None:
    with session_scope() as session:
        row = session.get(FeatureBuild, build_id)
        if row is None:
            return
        row.status = status
        row.finished_at = datetime.now(timezone.utc)
        if row_count is not None:
            row.row_count = row_count
        if notes:
            row.notes = notes
        if error:
            row.notes = (row.notes or "") + f"\n[error] {error}"


def _update_build(build_id: int, *, row_count: Optional[int] = None, extras: Optional[dict] = None) -> None:
    with session_scope() as session:
        row = session.get(FeatureBuild, build_id)
        if row is None:
            return
        if row_count is not None:
            row.row_count = row_count
        if extras:
            existing_notes = row.notes or ""
            row.notes = existing_notes + "\n" + json.dumps(extras, default=str)


@contextmanager
def feature_build(**kwargs) -> Iterator[FeatureBuildContext]:
    """Context manager — start + finish in one go.

    ::

        with feature_build(scope="EPL:2025", as_of_date=date.today()) as ctx:
            # writers pass ctx.build_id + ctx.feature_version into upserts
            ...
    """
    ctx = start_feature_build(**kwargs)
    try:
        yield ctx
    except Exception as exc:
        finish_feature_build(ctx.build_id, status="failed", error=str(exc))
        raise
    else:
        finish_feature_build(ctx.build_id, status="completed")
