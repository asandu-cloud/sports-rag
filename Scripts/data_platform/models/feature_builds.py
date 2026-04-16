"""Feature build lineage table (V3).

Every feature-snapshot write is anchored to a ``feature_builds`` row. A
build row names the feature pipeline version, the upstream payloads that
fed it, the scope (competition/season/as_of_date), and the operator /
caller metadata needed to reason about data quality.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Date, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin
from .json_type import JSONB


class FeatureBuild(Base, TimestampMixin):
    """A single feature-engineering run. Anchor for lineage queries."""

    __tablename__ = "feature_builds"
    __table_args__ = (
        UniqueConstraint("scope", "as_of_date", "feature_version", name="uq_feature_builds_scope_date_version"),
        Index("ix_feature_builds_scope_date", "scope", "as_of_date"),
        Index("ix_feature_builds_version", "feature_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    feature_version: Mapped[str] = mapped_column(String(32), nullable=False)
    scope: Mapped[str] = mapped_column(String(128), nullable=False)  # e.g. 'EPL:2025'
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    pipeline: Mapped[str] = mapped_column(String(64), nullable=False, default="default")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")

    sync_run_id: Mapped[Optional[int]] = mapped_column(ForeignKey("sync_runs.id", ondelete="SET NULL"))
    upstream_payload_ids: Mapped[Optional[list]] = mapped_column(JSONB)   # [raw_payload_archive.id, ...]
    inputs_digest: Mapped[Optional[str]] = mapped_column(String(64))      # sha256 of the inputs blob
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    notes: Mapped[Optional[str]] = mapped_column(Text)
