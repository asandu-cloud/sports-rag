"""Entity-based incremental KB refresh.

Bridges the canonical football tables (teams, fixtures, feature snapshots)
to Chroma. Emits doc entries from entity changes and only re-embeds what
has actually changed. Replaces the whole-season ``01_embed_and_upsert.py``
rebuild loop with ``spixctl kb refresh``.
"""

from .doc_builders import build_docs_for_entity  # noqa: F401
from .chroma_adapter import ChromaKBAdapter, embed_and_upsert_chroma  # noqa: F401
from .refresher import refresh_kb, enqueue_all_entities  # noqa: F401
