"""Pick a VectorBackend implementation at runtime.

Controlled by the ``VECTOR_BACKEND`` env var (``chroma`` | ``inmemory``).
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
    if target == "chroma":
        try:
            return ChromaVectorBackend()
        except Exception as exc:
            logger.warning("ChromaVectorBackend unavailable (%s); falling back to inmemory.", exc)
            return InMemoryVectorBackend()
    raise ValueError(f"Unknown VECTOR_BACKEND: {target}")
