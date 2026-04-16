"""Odds history / snapshot archive.

Stores per-event odds snapshots so the platform has auditable pricing
history without re-hitting API-Football every time. Used by PredictionService
for closing-odds capture and by the website for historical edge display.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin
from .json_type import JSONB


class OddsSnapshot(Base, TimestampMixin):
    __tablename__ = "odds_snapshots"
    __table_args__ = (
        Index("ix_odds_snapshots_event", "event_id", "captured_at"),
        Index("ix_odds_snapshots_league_date", "league", "kickoff_date"),
        UniqueConstraint("event_id", "captured_at", "kind", name="uq_odds_snapshot_event_time_kind"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[str] = mapped_column(String(64), nullable=False)
    league: Mapped[Optional[str]] = mapped_column(String(32))
    home_team: Mapped[Optional[str]] = mapped_column(String(128))
    away_team: Mapped[Optional[str]] = mapped_column(String(128))
    kickoff_utc: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    kickoff_date: Mapped[Optional[str]] = mapped_column(String(16))  # YYYY-MM-DD
    kind: Mapped[str] = mapped_column(String(16), nullable=False, default="prematch")  # 'prematch'|'closing'|'live'
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    odds_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
