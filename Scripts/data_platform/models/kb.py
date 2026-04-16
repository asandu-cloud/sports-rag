"""Knowledge-base document tables — canonical metadata for Chroma docs.

The embed pipeline moves from ``rebuild Index/normalized_*.json → upsert
all`` to ``diff kb_documents by content_hash → upsert only changed ids``.
The refresh queue lets entity-change signals (a fixture finished, a team
profile recomputed) drive delta embeds instead of full-season rebuilds.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin
from .json_type import JSONB


class KBDocument(Base, TimestampMixin):
    """One row per Chroma document; tracks content hash + embed sync state."""

    __tablename__ = "kb_documents"
    __table_args__ = (
        UniqueConstraint("doc_id", name="uq_kb_documents_doc_id"),
        Index("ix_kb_documents_entity", "entity_type", "entity_id"),
        Index("ix_kb_documents_league_season", "league", "season", "doc_type"),
        Index("ix_kb_documents_hash", "content_hash"),
        Index("ix_kb_documents_synced", "chroma_synced_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(64), nullable=False)  # MD5 used by Chroma
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)  # 'team'|'player'|'fixture'|'guide'
    entity_id: Mapped[Optional[str]] = mapped_column(String(128))         # e.g. '<team_id>', '<player_id>', '<fixture_api_id>'
    league: Mapped[Optional[str]] = mapped_column(String(32))
    season: Mapped[Optional[str]] = mapped_column(String(16))
    doc_type: Mapped[str] = mapped_column(String(32), nullable=False)     # 'team_profile'|'team_fixture'|'player_profile'|'player_fixture'|'data_dictionary'
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    doc_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False)
    chroma_synced_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    chroma_synced_hash: Mapped[Optional[str]] = mapped_column(String(64))
    last_source_run_id: Mapped[Optional[int]] = mapped_column(Integer)


class KBRefreshQueueItem(Base, TimestampMixin):
    """A pending (entity_type, entity_key, reason) trigger for the refresher."""

    __tablename__ = "kb_refresh_queue"
    __table_args__ = (
        Index("ix_kb_queue_entity_status", "entity_type", "status"),
        Index("ix_kb_queue_enqueued", "enqueued_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_key: Mapped[str] = mapped_column(String(128), nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    enqueued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    error_text: Mapped[Optional[str]] = mapped_column(Text)
    meta: Mapped[Optional[dict]] = mapped_column(JSONB)
