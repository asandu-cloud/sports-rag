"""Prediction tracking tables (canonical replacement for Index/predictions.db)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Date, DateTime, ForeignKey, Index, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin
from .json_type import JSONB


class Prediction(Base, TimestampMixin):
    """A single logged prediction (one leg of a broader recommendation)."""

    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_predictions_date", "prediction_date"),
        Index("ix_predictions_market", "market"),
        Index("ix_predictions_confidence", "confidence"),
        Index("ix_predictions_outcome", "outcome"),
        Index("ix_predictions_league", "league"),
        Index("ix_predictions_fixture_api_id", "fixture_api_id"),
        # Dedup key mirrors the SQLite logic in prediction_tracker.log_prediction
        UniqueConstraint(
            "prediction_date", "home_team", "away_team", "league",
            "market", "pick", "source",
            name="uq_predictions_dedup",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_api_id: Mapped[Optional[str]] = mapped_column(String(64))  # string to match legacy
    home_team: Mapped[str] = mapped_column(String(128), nullable=False)
    away_team: Mapped[str] = mapped_column(String(128), nullable=False)
    league: Mapped[str] = mapped_column(String(32), nullable=False)
    kickoff: Mapped[Optional[str]] = mapped_column(String(64))

    market: Mapped[str] = mapped_column(String(32), nullable=False)
    pick: Mapped[str] = mapped_column(String(255), nullable=False)
    side: Mapped[Optional[str]] = mapped_column(String(32))
    line: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    odds: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    bookmaker: Mapped[Optional[str]] = mapped_column(String(64))

    model_prob: Mapped[Optional[float]] = mapped_column(Numeric(10, 6))
    implied_prob: Mapped[Optional[float]] = mapped_column(Numeric(10, 6))
    value_edge: Mapped[Optional[float]] = mapped_column(Numeric(10, 6))
    confidence: Mapped[Optional[str]] = mapped_column(String(16))
    projected_total: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))

    source: Mapped[str] = mapped_column(String(32), nullable=False, default="standalone")

    actual_result: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    outcome: Mapped[Optional[str]] = mapped_column(String(16))  # 'win'|'loss'|'push'|'void'
    graded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    season: Mapped[Optional[str]] = mapped_column(String(16))
    prediction_date: Mapped[Optional[date]] = mapped_column(Date)

    closing_odds: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    closing_implied_prob: Mapped[Optional[float]] = mapped_column(Numeric(10, 6))
    clv: Mapped[Optional[float]] = mapped_column(Numeric(10, 6))

    extras: Mapped[Optional[dict]] = mapped_column(JSONB)


class TrialUser(Base, TimestampMixin):
    """Free-trial participant state (previously Index/predictions.db#trial_users)."""

    __tablename__ = "trial_users"
    __table_args__ = (
        UniqueConstraint("discord_id", name="uq_trial_users_discord_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    discord_id: Mapped[str] = mapped_column(String(64), nullable=False)
    discord_username: Mapped[Optional[str]] = mapped_column(String(128))
    activated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_parlays_seen: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_profit: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False, default=0.0)
    graduated: Mapped[bool] = mapped_column(Integer, nullable=False, default=0)
    graduated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class TrialParlay(Base, TimestampMixin):
    """Parlay posted to #free-picks during trial (previously Index/predictions.db#trial_parlays)."""

    __tablename__ = "trial_parlays"
    __table_args__ = (
        Index("ix_trial_parlays_matchday", "matchday_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    posted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    matchday_date: Mapped[date] = mapped_column(Date, nullable=False)
    legs: Mapped[dict] = mapped_column(JSONB, nullable=False)
    combined_odds: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)
    outcome: Mapped[Optional[str]] = mapped_column(String(16))
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    stake: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False, default=10.0)
    profit: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
