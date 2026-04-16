"""Structured metrics for ops dashboards."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from ..db import session_scope
from ..models import KBRefreshQueueItem, SyncRun, SyncWatermark


def sync_job_metrics(*, since_hours: int = 24) -> Dict[str, Any]:
    """Counts of recent sync runs by kind + status."""
    since = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    with session_scope() as session:
        stmt = (
            select(SyncRun.run_kind, SyncRun.status, func.count(SyncRun.id))
            .where(SyncRun.started_at >= since)
            .group_by(SyncRun.run_kind, SyncRun.status)
        )
        rows = session.execute(stmt).all()
        last = session.scalar(select(SyncRun).order_by(desc(SyncRun.started_at)).limit(1))
        latest = {
            "id": last.id, "kind": last.run_kind, "scope": last.scope,
            "status": last.status,
            "started_at": last.started_at.isoformat() if last.started_at else None,
            "finished_at": last.finished_at.isoformat() if last.finished_at else None,
        } if last else None
    buckets: Dict[str, Dict[str, int]] = {}
    for kind, status, count in rows:
        buckets.setdefault(kind or "?", {})[status or "?"] = int(count)
    return {
        "window_hours": since_hours,
        "counts_by_kind": buckets,
        "latest_run": latest,
    }


def queue_metrics() -> Dict[str, Any]:
    """KB queue depth + oldest pending item."""
    with session_scope() as session:
        by_status = session.execute(
            select(KBRefreshQueueItem.status, func.count(KBRefreshQueueItem.id))
            .group_by(KBRefreshQueueItem.status)
        ).all()
        oldest = session.scalar(
            select(KBRefreshQueueItem)
            .where(KBRefreshQueueItem.status == "pending")
            .order_by(KBRefreshQueueItem.enqueued_at.asc())
            .limit(1)
        )
    stats = {status or "?": int(count) for status, count in by_status}
    oldest_at = oldest.enqueued_at.isoformat() if (oldest and oldest.enqueued_at) else None
    return {
        "kb_queue": stats,
        "kb_oldest_pending_at": oldest_at,
    }


def watermark_metrics(*, stale_after_minutes: int = 120) -> Dict[str, Any]:
    """Watermarks that haven't ticked inside ``stale_after_minutes``."""
    now = datetime.now(timezone.utc)
    stale_threshold = now - timedelta(minutes=stale_after_minutes)
    with session_scope() as session:
        rows = session.scalars(select(SyncWatermark)).all()
    total = len(rows)
    stale = []
    for row in rows:
        last = row.last_successful_at
        if last is None:
            stale.append({"scope": row.scope, "last_successful_at": None})
        elif last < stale_threshold:
            stale.append({"scope": row.scope, "last_successful_at": last.isoformat()})
    return {
        "total": total,
        "stale": stale,
        "stale_after_minutes": stale_after_minutes,
    }
