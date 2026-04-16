"""Chroma-backed VectorBackend — production default.

Imports chromadb + openai lazily so the platform still boots when
neither is installed (e.g. bots-only deployments).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .base import VectorUpsertResult

logger = logging.getLogger(__name__)


EmbedFn = Callable[[List[str], str], List[List[float]]]


def _default_openai_embed(texts: List[str], model: str) -> List[List[float]]:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]


class ChromaVectorBackend:
    name = "chroma"

    def __init__(
        self,
        *,
        collection_name: str = "football_top5",
        chroma_dir: Optional[str] = None,
        embed_model: str = "text-embedding-3-large",
        embed_fn: Optional[EmbedFn] = None,
    ):
        self.collection_name = collection_name
        self.chroma_dir = chroma_dir or os.path.join(os.getcwd(), "Index", "chroma")
        self.embed_model = embed_model
        self.embed_fn = embed_fn or _default_openai_embed
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            try:
                from chroma_backend import get_chroma_client
            except ImportError:
                from Scripts.rag_ingest.chroma_backend import get_chroma_client  # type: ignore
            client = get_chroma_client(self.chroma_dir)
            self._collection = client.get_or_create_collection(self.collection_name)
        return self._collection

    def upsert(self, docs: Sequence[Mapping[str, Any]]) -> VectorUpsertResult:
        if not docs:
            return VectorUpsertResult(upserted=0)
        collection = self._get_collection()
        ids = [d["id"] for d in docs]
        texts = [d.get("text", "") for d in docs]
        metas = [dict(d.get("metadata") or {}) for d in docs]
        vectors = self.embed_fn(texts, self.embed_model)
        collection.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
        return VectorUpsertResult(upserted=len(docs))

    def delete(self, doc_ids: Iterable[str]) -> int:
        ids = list(doc_ids)
        if not ids:
            return 0
        collection = self._get_collection()
        collection.delete(ids=ids)
        return len(ids)

    def query(self, text: str, *, limit: int = 5, where: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        collection = self._get_collection()
        embedding = self.embed_fn([text], self.embed_model)[0]
        result = collection.query(query_embeddings=[embedding], n_results=limit, where=where or {})
        ids = (result.get("ids") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        out: List[Dict[str, Any]] = []
        for i, doc_id in enumerate(ids):
            out.append({
                "id": doc_id,
                "distance": distances[i] if i < len(distances) else None,
                "metadata": metas[i] if i < len(metas) else {},
            })
        return out
