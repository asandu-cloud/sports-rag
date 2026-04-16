"""Vector backend abstraction (V3).

The KB refresher talks to a :class:`VectorBackend` protocol. Chroma is
the default implementation, but it is no longer a hardcoded forever
dependency — swap to pgvector, Weaviate, Pinecone, etc. by implementing
the protocol and binding a new default in :func:`build_backend`.
"""

from .base import VectorBackend, VectorUpsertResult  # noqa: F401
from .chroma import ChromaVectorBackend  # noqa: F401
from .inmemory import InMemoryVectorBackend  # noqa: F401
from .factory import build_backend  # noqa: F401
