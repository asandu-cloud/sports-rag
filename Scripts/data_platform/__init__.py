"""Canonical data platform for Betting RAG (V1).

Introduces a PostgreSQL-backed source of truth for competitions, seasons,
teams, players, fixtures, fixture stats, and feature snapshots. Provides
incremental sync, raw payload archival, and a user/subscription repository
that can replace Index/predictions.db over time.

The existing file-heavy pipeline (pull_season.py, 00_normalize_to_docs.py,
01_embed_and_upsert.py) is left intact. Chroma remains the KB/retrieval
layer but is now downstream of structured data.

Feature-flagged via ``PLATFORM_ENABLED=1`` in the environment so existing
workflows keep working until migration is complete.
"""

__version__ = "0.1.0"
