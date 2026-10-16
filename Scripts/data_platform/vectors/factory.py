"""Pick a VectorBackend implementation at runtime.

Controlled by the ``VECTOR_BACKEND`` env var (``chroma`` | ``inmemory`` | ``sqlite`` | ``sqlite+chroma``).
Defaults to chroma for backward compatibility with V1/V2.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .base import VectorBackend
from .chroma import ChromaVectorBackend
from .inmemory import InMemoryVectorBackend

logger = logging.getLogger(__name__)


def build_backend(kind: Optional[str] = None) -> VectorBackend:
    target = (kind or os.environ.get("VECTOR_BACKEND") or "chroma").strip().lower()
    if target == "inmemory":
        return InMemoryVectorBackend()
    if target in ("sqlite", "sqlite+chroma"):
        # Rework A2a: the engine reads the SQLite document store; Chroma is a best-effort mirror.
        from .document_store import DocumentStoreBackend, MirroredBackend
        store = DocumentStoreBackend()
        return store if target == "sqlite" else MirroredBackend(store, ChromaVectorBackend())
    if target == "chroma":
        try:
            return ChromaVectorBackend()
        except Exception as exc:
            logger.warning("ChromaVectorBackend unavailable (%s); falling back to inmemory.", exc)
            return InMemoryVectorBackend()
    raise ValueError(f"Unknown VECTOR_BACKEND: {target}")
