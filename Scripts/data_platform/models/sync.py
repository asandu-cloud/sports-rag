"""Sync bookkeeping: runs, watermarks, raw payload archive metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin
from .json_type import JSONB


class SyncRun(Base, TimestampMixin):
    """A single run of a sync / build pipeline.

    Used as a lineage anchor for feature snapshots, payload archival rows,
    and operational debugging.
    """

    __tablename__ = "sync_runs"
    __table_args__ = (
        Index("ix_sync_runs_kind_started", "run_kind", "started_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_kind: Mapped[str] = mapped_column(String(32), nullable=False)  # 'bootstrap' | 'incremental' | 'feature_build' | 'migrate'
    scope: Mapped[Optional[str]] = mapped_column(String(128))          # e.g. 'EPL:2025', 'ALL:current'
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")  # running|completed|failed|partial
    stats: Mapped[Optional[dict]] = mapped_column(JSONB)
    error_text: Mapped[Optional[str]] = mapped_column(Text)


class SyncWatermark(Base, TimestampMixin):
    """Per-scope cursor used to drive incremental fetches.

    Typical scopes:
      ``fixtures:<competition_code>:<season_year>`` - last fixture pull
      ``team_stats:<fixture_id>``                   - per-fixture details
      ``player_stats:<fixture_id>``                 - per-fixture details

    ``last_cursor`` is free-form (ISO date, fixture id, etc.).
    """

    __tablename__ = "sync_watermarks"
    __table_args__ = (
        UniqueConstraint("scope", name="uq_sync_watermarks_scope"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scope: Mapped[str] = mapped_column(String(128), nullable=False)
    last_successful_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_cursor: Mapped[Optional[str]] = mapped_column(Text)
    meta: Mapped[Optional[dict]] = mapped_column(JSONB)


class RawPayloadArchive(Base, TimestampMixin):
    """Metadata for archived raw provider payloads.

    The payload bytes themselves live in the object-storage backend (local
    disk in V1, S3 later); this table tracks where to find them and what
    they were. Keeping metadata in Postgres makes payloads queryable and
    replayable.
    """

    __tablename__ = "raw_payload_archive"
    __table_args__ = (
        Index("ix_raw_archive_provider_endpoint", "provider", "endpoint", "fetched_at"),
        Index("ix_raw_archive_digest", "payload_digest"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)  # 'api_football'
    endpoint: Mapped[str] = mapped_column(String(128), nullable=False) # '/fixtures', '/fixtures/statistics', ...
    params_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    params: Mapped[Optional[dict]] = mapped_column(JSONB)
    storage_backend: Mapped[str] = mapped_column(String(32), nullable=False)  # 'local' | 's3'
    storage_uri: Mapped[str] = mapped_column(Text, nullable=False)
    payload_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    bytes_size: Mapped[Optional[int]] = mapped_column(Integer)
    content_type: Mapped[Optional[str]] = mapped_column(String(64))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    sync_run_id: Mapped[Optional[int]] = mapped_column(ForeignKey("sync_runs.id", ondelete="SET NULL"))
