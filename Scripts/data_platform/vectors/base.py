"""Vector backend protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class VectorUpsertResult:
    upserted: int
    skipped: int = 0


@runtime_checkable
class VectorBackend(Protocol):
    """Minimal interface the KB layer depends on.

    Implementations must be safe to use from synchronous code. Errors
    should bubble up; the KB service decides whether to retry.
    """

    name: str

    def upsert(self, docs: Sequence[Mapping[str, Any]]) -> VectorUpsertResult:
        """Persist a batch of documents.

        Each doc is expected to expose:
          * ``id`` — string identifier (matches ``kb_documents.doc_id``)
          * ``text`` — canonical text to embed
          * ``metadata`` — mapping of scalar fields for retrieval filtering
        """

    def delete(self, doc_ids: Iterable[str]) -> int:
        """Remove docs by id. Returns the number of rows removed."""

    def query(self, text: str, *, limit: int = 5, where: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        """Optional — semantic search. Returns [{id, distance, metadata}]."""
