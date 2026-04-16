"""Knowledge-base document + refresh-queue repository.

The KB layer treats Chroma as a downstream consumer: ``kb_documents``
holds the canonical doc text + metadata + content hash; ``kb_refresh_queue``
drives incremental rebuilds. Changing an entity (fixture finishes, team
profile refreshed) is a tiny enqueue. The refresher drains the queue,
recomputes doc text for each affected entity, and only re-embeds docs
whose content hash changed.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import session_scope
from ..models import KBDocument, KBRefreshQueueItem

logger = logging.getLogger(__name__)


def compute_content_hash(text: str, metadata: Mapping[str, Any]) -> str:
    """Stable hash over doc text + relevant metadata."""
    import json as _json
    blob = _json.dumps({"text": text, "meta": dict(metadata or {})}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class KBDocumentRepository:
    def __init__(self, session_factory=session_scope):
        self._factory = session_factory

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------
    def upsert(
        self,
        *,
        doc_id: str,
        entity_type: str,
        entity_id: Optional[str],
        league: Optional[str],
        season: Optional[str],
        doc_type: str,
        text: str,
        metadata: Mapping[str, Any],
        source_run_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Insert or update a doc. Returns summary + ``changed`` flag."""
        content_hash = compute_content_hash(text, metadata)
        with self._factory() as session:
            row = session.scalar(select(KBDocument).where(KBDocument.doc_id == doc_id))
            changed = True
            if row is None:
                row = KBDocument(
                    doc_id=doc_id,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    league=league,
                    season=season,
                    doc_type=doc_type,
                    content_hash=content_hash,
                    text=text,
                    doc_metadata=dict(metadata),
                    last_source_run_id=source_run_id,
                )
                session.add(row)
                session.flush()
            else:
                if row.content_hash == content_hash:
                    changed = False
                row.entity_type = entity_type
                row.entity_id = entity_id
                row.league = league
                row.season = season
                row.doc_type = doc_type
                row.text = text
                row.doc_metadata = dict(metadata)
                row.content_hash = content_hash
                if source_run_id is not None:
                    row.last_source_run_id = source_run_id
            return {
                "doc_id": doc_id,
                "id": row.id,
                "changed": changed,
                "content_hash": content_hash,
            }

    def mark_synced(
        self,
        *,
        doc_ids: Iterable[str],
        synced_hashes: Optional[Mapping[str, str]] = None,
    ) -> int:
        synced_at = datetime.now(timezone.utc)
        ids = list(doc_ids)
        if not ids:
            return 0
        with self._factory() as session:
            rows = session.scalars(
                select(KBDocument).where(KBDocument.doc_id.in_(ids))
            ).all()
            count = 0
            for row in rows:
                row.chroma_synced_at = synced_at
                row.chroma_synced_hash = (synced_hashes or {}).get(row.doc_id, row.content_hash)
                count += 1
            return count

    def list_pending_embed(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Documents whose current content_hash differs from the synced hash."""
        with self._factory() as session:
            stmt = select(KBDocument).where(
                (KBDocument.chroma_synced_hash.is_(None))
                | (KBDocument.chroma_synced_hash != KBDocument.content_hash)
            ).order_by(KBDocument.updated_at.desc())
            if limit is not None:
                stmt = stmt.limit(limit)
            rows = session.scalars(stmt).all()
            return [_doc_to_dict(r) for r in rows]

    def get_by_doc_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._factory() as session:
            row = session.scalar(select(KBDocument).where(KBDocument.doc_id == doc_id))
            return _doc_to_dict(row) if row else None

    def count(self) -> int:
        with self._factory() as session:
            return session.scalar(select(func.count(KBDocument.id))) or 0

    # ------------------------------------------------------------------
    # Refresh queue
    # ------------------------------------------------------------------
    def enqueue(
        self,
        *,
        entity_type: str,
        entity_key: str,
        reason: Optional[str] = None,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> int:
        with self._factory() as session:
            existing = session.scalar(
                select(KBRefreshQueueItem).where(
                    KBRefreshQueueItem.entity_type == entity_type,
                    KBRefreshQueueItem.entity_key == entity_key,
                    KBRefreshQueueItem.status == "pending",
                )
            )
            if existing is not None:
                # Coalesce duplicate enqueues — newer reason wins but id stable.
                if reason:
                    existing.reason = reason
                if meta is not None:
                    merged = dict(existing.meta or {})
                    merged.update(meta)
                    existing.meta = merged
                return int(existing.id)
            row = KBRefreshQueueItem(
                entity_type=entity_type,
                entity_key=entity_key,
                reason=reason,
                status="pending",
                attempts=0,
                enqueued_at=datetime.now(timezone.utc),
                meta=dict(meta) if meta is not None else None,
            )
            session.add(row)
            session.flush()
            return int(row.id)

    def claim_pending(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        """Mark up to ``limit`` pending items as ``in_progress`` and return them."""
        claimed: List[Dict[str, Any]] = []
        started = datetime.now(timezone.utc)
        with self._factory() as session:
            rows = session.scalars(
                select(KBRefreshQueueItem)
                .where(KBRefreshQueueItem.status == "pending")
                .order_by(KBRefreshQueueItem.enqueued_at.asc())
                .limit(limit)
            ).all()
            for row in rows:
                row.status = "in_progress"
                row.started_at = started
                row.attempts = (row.attempts or 0) + 1
                claimed.append({
                    "id": row.id,
                    "entity_type": row.entity_type,
                    "entity_key": row.entity_key,
                    "reason": row.reason,
                    "attempts": row.attempts,
                    "meta": row.meta,
                })
        return claimed

    def finish(self, queue_id: int, *, status: str = "completed", error: Optional[str] = None) -> None:
        with self._factory() as session:
            row = session.get(KBRefreshQueueItem, queue_id)
            if row is None:
                return
            row.status = status
            row.finished_at = datetime.now(timezone.utc)
            if error:
                row.error_text = error

    def queue_stats(self) -> Dict[str, int]:
        with self._factory() as session:
            rows = session.execute(
                select(KBRefreshQueueItem.status, func.count(KBRefreshQueueItem.id))
                .group_by(KBRefreshQueueItem.status)
            ).all()
            return {status: int(count) for status, count in rows}


def _doc_to_dict(row: KBDocument) -> Dict[str, Any]:
    return {
        "doc_id": row.doc_id,
        "entity_type": row.entity_type,
        "entity_id": row.entity_id,
        "league": row.league,
        "season": row.season,
        "doc_type": row.doc_type,
        "content_hash": row.content_hash,
        "text": row.text,
        "metadata": row.doc_metadata or {},
        "chroma_synced_at": row.chroma_synced_at.isoformat() if row.chroma_synced_at else None,
        "chroma_synced_hash": row.chroma_synced_hash,
    }
