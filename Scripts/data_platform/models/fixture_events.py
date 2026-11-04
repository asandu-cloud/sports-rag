"""Per-fixture match events (goals, cards, substitutions, VAR) from API-Football's ``/fixtures/events``."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base
from .json_type import JSONB


class FixtureEvent(Base):
    """One event of a finished fixture, in API-Football's order (``event_index`` from 0).

    The refresh replaces a fixture's events as a whole. Teams and players are API-Football ids,
    so an event never waits on a team or player row.
    """

    __tablename__ = "fixture_match_events"
    __table_args__ = (UniqueConstraint("fixture_id", "event_index", name="uq_fixture_match_events_fixture_index"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id", ondelete="CASCADE"), nullable=False)
    event_index: Mapped[int] = mapped_column(Integer, nullable=False)
    team_api_id: Mapped[Optional[int]] = mapped_column(Integer)
    player_api_id: Mapped[Optional[int]] = mapped_column(Integer)
    assist_api_id: Mapped[Optional[int]] = mapped_column(Integer)
    elapsed: Mapped[Optional[int]] = mapped_column(Integer)
    extra: Mapped[Optional[int]] = mapped_column(Integer)
    event_type: Mapped[Optional[str]] = mapped_column(String(32))
    detail: Mapped[Optional[str]] = mapped_column(String(128))
    comments: Mapped[Optional[str]] = mapped_column(Text)
    raw_json: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
