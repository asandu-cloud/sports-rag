"""Operational records for scheduled Match Read refreshes.

``MatchRead`` itself is deliberately immutable: re-evaluating the same input
snapshot must return the existing version rather than manufacture a new
revision just because a worker ran again.  These records keep the operational
facts separate from that audit object:

* a short-lived lease prevents two schedulers from publishing the same slate;
* every refresh attempt is appended as an observation, including an unchanged
  card, a pending lineup, or a provider failure.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin
from .json_type import JSONB


class WorkerLease(Base, TimestampMixin):
    """A short-lived, named lease used by one-shot scheduled workers."""

    __tablename__ = "worker_leases"
    __table_args__ = (
        UniqueConstraint("lease_key", name="uq_worker_leases_key"),
        Index("ix_worker_leases_expires_at", "expires_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    lease_key: Mapped[str] = mapped_column(String(128), nullable=False)
    owner_id: Mapped[str] = mapped_column(String(96), nullable=False)
    acquired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    heartbeat_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB)


class MatchReadObservation(Base, TimestampMixin):
    """One append-only check of a fixture/stage by the Match Read worker."""

    __tablename__ = "match_read_observations"
    __table_args__ = (
        Index("ix_match_read_observations_fixture_stage_checked", "fixture_api_id", "stage", "checked_at"),
        Index("ix_match_read_observations_read_checked", "match_read_id", "checked_at"),
        Index("ix_match_read_observations_run", "sync_run_id"),
        Index("ix_match_read_observations_status_checked", "status", "checked_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sync_run_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("sync_runs.id", ondelete="SET NULL")
    )
    match_read_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("match_reads.id", ondelete="SET NULL")
    )
    fixture_api_id: Mapped[str] = mapped_column(String(64), nullable=False)
    league: Mapped[str] = mapped_column(String(32), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # ``verified`` / ``unchanged`` are successful fresh evaluations.
    # ``lineups_pending`` is an expected retry state, while ``failed`` keeps
    # provider/model failures visible to operators.
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="api_football")
    input_snapshot_id: Mapped[Optional[str]] = mapped_column(String(256))
    read_key: Mapped[Optional[str]] = mapped_column(String(64))
    provider_calls_json: Mapped[Optional[dict]] = mapped_column(JSONB)
    detail_json: Mapped[Optional[dict]] = mapped_column(JSONB)
    error_text: Mapped[Optional[str]] = mapped_column(Text)
