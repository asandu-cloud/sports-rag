"""In-memory backend — drives tests and dry runs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .base import VectorUpsertResult


class InMemoryVectorBackend:
    """Dict-backed store. No embeddings; purely for functional tests."""

    name = "inmemory"

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    def upsert(self, docs: Sequence[Mapping[str, Any]]) -> VectorUpsertResult:
        count = 0
        for doc in docs:
            doc_id = doc["id"]
            self._store[doc_id] = {
                "id": doc_id,
                "text": doc.get("text", ""),
                "metadata": dict(doc.get("metadata") or {}),
            }
            count += 1
        return VectorUpsertResult(upserted=count)

    def delete(self, doc_ids: Iterable[str]) -> int:
        count = 0
        for doc_id in doc_ids:
            if doc_id in self._store:
                del self._store[doc_id]
                count += 1
        return count

    def query(self, text: str, *, limit: int = 5, where: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        def _matches(meta: Dict[str, Any]) -> bool:
            if not where:
                return True
            return all(meta.get(k) == v for k, v in where.items())

        matches: List[Dict[str, Any]] = []
        for doc in self._store.values():
            if not _matches(doc.get("metadata") or {}):
                continue
            matches.append({
                "id": doc["id"],
                "distance": 0.0,
                "metadata": doc.get("metadata") or {},
            })
            if len(matches) >= limit:
                break
        return matches

    # Test helpers
    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._store

    def get(self, doc_id: str) -> Dict[str, Any] | None:
        return self._store.get(doc_id)

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {k: dict(v) for k, v in self._store.items()}
