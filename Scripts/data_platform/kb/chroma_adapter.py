"""Chroma adapter for the KB refresher.

Isolated so tests can pass a no-op upserter instead of the real OpenAI +
Chroma stack. Production code can rely on :func:`embed_and_upsert_chroma`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChromaKBAdapter:
    """Minimal interface: a callable that takes a list of docs and returns
    the number of docs embedded. Provided so the KBService stays decoupled
    from OpenAI / chromadb.
    """

    upserter: Callable[[List[Dict[str, Any]]], int]

    def __call__(self, docs: List[Dict[str, Any]]) -> int:
        return self.upserter(docs)


def _default_openai_embed(texts: List[str], model: str) -> List[List[float]]:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]


def embed_and_upsert_chroma(
    docs: List[Dict[str, Any]],
    *,
    collection_name: str = "football_top5",
    chroma_dir: Optional[str] = None,
    embed_model: str = "text-embedding-3-large",
    embed_fn: Optional[Callable[[List[str], str], List[List[float]]]] = None,
) -> int:
    """Embed a batch of docs and upsert them into Chroma.

    Kept deliberately thin — mirrors the logic of
    ``Scripts/rag_ingest/01_embed_and_upsert.py`` but operates on an
    arbitrary doc list, not a season-wide JSON file.
    """
    if not docs:
        return 0
    try:
        from chroma_backend import env_first, get_chroma_client
    except ImportError:
        from Scripts.rag_ingest.chroma_backend import env_first, get_chroma_client  # type: ignore

    if chroma_dir is None:
        chroma_dir = os.path.join(os.getcwd(), "Index", "chroma")
    client = get_chroma_client(chroma_dir)
    collection = client.get_or_create_collection(collection_name)

    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metas = [d.get("metadata", {}) for d in docs]
    vectors = (embed_fn or _default_openai_embed)(texts, embed_model)

    collection.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    return len(docs)
