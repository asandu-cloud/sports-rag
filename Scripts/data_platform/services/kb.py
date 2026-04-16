"""KBService — document management + incremental refresh to Chroma.

Turns entity-change signals (e.g. a fixture finished) into targeted
Chroma upserts. Content-hash deduping means re-embed calls only happen
for docs whose text actually changed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from ..repositories.kb import KBDocumentRepository

logger = logging.getLogger(__name__)


@dataclass
class RefreshResult:
    dequeued: int = 0
    docs_scanned: int = 0
    docs_changed: int = 0
    docs_embedded: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dequeued": self.dequeued,
            "docs_scanned": self.docs_scanned,
            "docs_changed": self.docs_changed,
            "docs_embedded": self.docs_embedded,
            "error_count": len(self.errors),
            "errors": self.errors[:10],
        }


# A ``DocBuilder`` takes an (entity_type, entity_key) and returns an iterable
# of doc dicts. Each doc dict must have: doc_id, entity_type, entity_id,
# league, season, doc_type, text, metadata.
DocBuilder = Callable[[str, str], Iterable[Dict[str, Any]]]


class KBService:
    def __init__(
        self,
        *,
        repo: Optional[KBDocumentRepository] = None,
        doc_builder: Optional[DocBuilder] = None,
        chroma_upserter: Optional[Callable[[List[Dict[str, Any]]], int]] = None,
    ):
        self._repo = repo or KBDocumentRepository()
        self._doc_builder = doc_builder
        self._chroma_upserter = chroma_upserter

    # ------------------------------------------------------------------
    # Doc management
    # ------------------------------------------------------------------
    def upsert_document(
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
        return self._repo.upsert(
            doc_id=doc_id,
            entity_type=entity_type,
            entity_id=entity_id,
            league=league,
            season=season,
            doc_type=doc_type,
            text=text,
            metadata=metadata,
            source_run_id=source_run_id,
        )

    def enqueue(
        self,
        *,
        entity_type: str,
        entity_key: str,
        reason: Optional[str] = None,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> int:
        return self._repo.enqueue(entity_type=entity_type, entity_key=entity_key, reason=reason, meta=meta)

    def enqueue_many(self, items: Iterable[Mapping[str, Any]]) -> int:
        count = 0
        for entry in items:
            self._repo.enqueue(
                entity_type=str(entry["entity_type"]),
                entity_key=str(entry["entity_key"]),
                reason=entry.get("reason"),
                meta=entry.get("meta"),
            )
            count += 1
        return count

    def queue_stats(self) -> Dict[str, int]:
        return self._repo.queue_stats()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------
    def refresh(
        self,
        *,
        batch_size: int = 50,
        embed_batch_size: int = 256,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Drain the refresh queue and push changed docs to Chroma.

        The sequence is:
          1. Claim ``batch_size`` pending queue items.
          2. For each, call the configured ``doc_builder`` to regenerate
             the candidate docs. Upsert them, recording which ones changed.
          3. Send changed docs to the configured ``chroma_upserter``.
          4. Mark claimed queue items completed (or failed).
        """
        if self._doc_builder is None:
            raise RuntimeError("KBService.refresh requires doc_builder to be set")

        result = RefreshResult()
        claimed = self._repo.claim_pending(limit=batch_size)
        result.dequeued = len(claimed)

        changed_docs: List[Dict[str, Any]] = []
        per_item_changed: Dict[int, List[str]] = {}

        for item in claimed:
            try:
                generated = list(self._doc_builder(item["entity_type"], item["entity_key"]))
                for doc in generated:
                    summary = self._repo.upsert(
                        doc_id=doc["doc_id"],
                        entity_type=doc.get("entity_type", item["entity_type"]),
                        entity_id=doc.get("entity_id"),
                        league=doc.get("league"),
                        season=doc.get("season"),
                        doc_type=doc["doc_type"],
                        text=doc["text"],
                        metadata=doc.get("metadata", {}),
                    )
                    result.docs_scanned += 1
                    if summary["changed"]:
                        result.docs_changed += 1
                        changed_docs.append({
                            "id": doc["doc_id"],
                            "text": doc["text"],
                            "metadata": doc.get("metadata", {}),
                            "content_hash": summary["content_hash"],
                        })
                        per_item_changed.setdefault(item["id"], []).append(doc["doc_id"])
            except Exception as exc:
                result.errors.append({
                    "queue_id": item["id"],
                    "entity_type": item["entity_type"],
                    "entity_key": item["entity_key"],
                    "error": str(exc),
                })
                self._repo.finish(item["id"], status="failed", error=str(exc))
                continue
            self._repo.finish(item["id"], status="completed")

        if changed_docs and not dry_run and self._chroma_upserter is not None:
            # Batch embed through the configured upserter.
            for i in range(0, len(changed_docs), embed_batch_size):
                batch = changed_docs[i:i + embed_batch_size]
                embedded = self._chroma_upserter(batch)
                result.docs_embedded += int(embedded or 0)
                self._repo.mark_synced(
                    doc_ids=[d["id"] for d in batch],
                    synced_hashes={d["id"]: d["content_hash"] for d in batch},
                )
        elif changed_docs and dry_run:
            # In dry-run we still record the synced hash to match what embed
            # would have done — callers should decide when to set dry_run=False.
            pass

        return result.as_dict()

    def pending_embed_count(self) -> int:
        return len(self._repo.list_pending_embed())
