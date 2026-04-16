"""Live runtime tables — canonical replacement for Index/live.db contents.

Mirrors the shape of live_store.LiveStore tables so the shim can swap in
without touching the LivePoller / LiveEngine control flow.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Index, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin
from .json_type import JSONB


class LiveFixture(Base, TimestampMixin):
    __tablename__ = "live_fixtures"

    fixture_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    league: Mapped[str] = mapped_column(String(64), nullable=False)
    home_team: Mapped[str] = mapped_column(String(128), nullable=False)
    away_team: Mapped[str] = mapped_column(String(128), nullable=False)
    kickoff_utc: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class LiveStateSnapshot(Base, TimestampMixin):
    __tablename__ = "live_state_snapshots"
    __table_args__ = (
        Index("ix_live_state_fixture_time", "fixture_id", "captured_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(Integer, nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[Optional[str]] = mapped_column(String(16))
    elapsed_min: Mapped[Optional[float]] = mapped_column(Numeric(6, 2))
    state_json: Mapped[dict] = mapped_column(JSONB, nullable=False)


class LiveModelSnapshot(Base, TimestampMixin):
    __tablename__ = "live_model_snapshots"
    __table_args__ = (
        Index("ix_live_model_fixture_time", "fixture_id", "snapshot_ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot_ts: Mapped[str] = mapped_column(String(64), nullable=False)  # ISO string
    elapsed_min: Mapped[Optional[float]] = mapped_column(Numeric(6, 2))
    score: Mapped[Optional[str]] = mapped_column(String(16))
    snapshot_json: Mapped[dict] = mapped_column(JSONB, nullable=False)


class LiveOddsSnapshot(Base, TimestampMixin):
    __tablename__ = "live_odds_snapshots"
    __table_args__ = (
        Index("ix_live_odds_fixture_time", "fixture_id", "captured_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(Integer, nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    odds_json: Mapped[dict] = mapped_column(JSONB, nullable=False)


class LiveAlert(Base, TimestampMixin):
    __tablename__ = "live_alerts"
    __table_args__ = (
        Index("ix_live_alerts_fixture_time", "fixture_id", "snapshot_ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot_ts: Mapped[str] = mapped_column(String(64), nullable=False)
    emitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    alert_type: Mapped[Optional[str]] = mapped_column(String(64))
    severity: Mapped[Optional[str]] = mapped_column(String(16))
    message: Mapped[Optional[str]] = mapped_column(Text)
    alert_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
