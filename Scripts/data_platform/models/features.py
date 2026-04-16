"""Feature snapshot tables.

These store derived features (rolling averages, indices) at a given
``as_of_date`` so backtests and live runs can retrieve the same view a
bot would have seen at any past time.

Variable names preserve the existing vocabulary used throughout
``rag_cli_v2.py``, ``core/projections.py``, and the normalizer so
downstream code can adopt the DB source without renames.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Date, DateTime, ForeignKey, Index, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin
from .json_type import JSONB


class TeamFeatureSnapshot(Base, TimestampMixin):
    """Season-to-date team feature snapshot as of ``as_of_date``."""

    __tablename__ = "team_feature_snapshots"
    __table_args__ = (
        UniqueConstraint("team_id", "season_id", "as_of_date", name="uq_team_feature_snapshot"),
        Index("ix_team_feature_season_date", "season_id", "as_of_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id", ondelete="CASCADE"), nullable=False)
    season_id: Mapped[int] = mapped_column(ForeignKey("seasons.id", ondelete="CASCADE"), nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    matches_played: Mapped[Optional[int]] = mapped_column(Integer)

    # Core per-match averages (preserved names from the current pipeline)
    goals_for_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    goals_against_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    assists_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    corners_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    corners_against_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    sot_for_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    sot_against_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    shots_for_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))

    cards_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    yellows_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    reds_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    cards_per_90_team: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    fouls_per_90_team: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    cards_per_foul_team: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))

    # Indices
    aggression_index_norm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    form_index_team: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    control_index: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    dominance_index: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    possession: Mapped[Optional[float]] = mapped_column(Numeric(6, 4))
    expected_goals: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    pass_accuracy_pct: Mapped[Optional[float]] = mapped_column(Numeric(6, 4))

    # Opponent-join aggregations
    opp_cards_induced_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))

    # Venue-split aggregations
    goals_home_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    goals_away_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    corners_home_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    corners_away_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    cards_home_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    cards_away_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    corners_against_home_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    corners_against_away_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    sot_against_home_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    sot_against_away_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    cards_induced_home_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    cards_induced_away_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    xg_home_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    xg_away_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    sot_home_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    sot_away_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    fouls_home_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    fouls_away_pm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))

    # Stat variance (used by projection uncertainty)
    corners_var: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    goals_var: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    cards_var: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    sot_var: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))

    archetype: Mapped[Optional[str]] = mapped_column(String(64))

    build_run_id: Mapped[Optional[int]] = mapped_column(ForeignKey("sync_runs.id", ondelete="SET NULL"))
    # V3 lineage — feature_version stays nullable to keep legacy rows valid
    feature_version: Mapped[Optional[str]] = mapped_column(String(32))
    feature_build_id: Mapped[Optional[int]] = mapped_column(ForeignKey("feature_builds.id", ondelete="SET NULL"))
    as_of_fixture_api_id: Mapped[Optional[int]] = mapped_column(Integer)
    built_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    extras: Mapped[Optional[dict]] = mapped_column(JSONB)


class PlayerFeatureSnapshot(Base, TimestampMixin):
    """Season-to-date player feature snapshot as of ``as_of_date``."""

    __tablename__ = "player_feature_snapshots"
    __table_args__ = (
        UniqueConstraint("player_id", "season_id", "as_of_date", name="uq_player_feature_snapshot"),
        Index("ix_player_feature_season_date", "season_id", "as_of_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id", ondelete="CASCADE"), nullable=False)
    team_id: Mapped[Optional[int]] = mapped_column(ForeignKey("teams.id", ondelete="SET NULL"))
    season_id: Mapped[int] = mapped_column(ForeignKey("seasons.id", ondelete="CASCADE"), nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)

    appearances_total: Mapped[Optional[int]] = mapped_column(Integer)
    appearances_as_starter: Mapped[Optional[int]] = mapped_column(Integer)
    start_rate: Mapped[Optional[float]] = mapped_column(Numeric(6, 4))
    avg_minutes_per_appearance: Mapped[Optional[float]] = mapped_column(Numeric(7, 2))
    minutes_risk: Mapped[Optional[str]] = mapped_column(String(32))
    recent_role_trend: Mapped[Optional[str]] = mapped_column(String(32))

    # Per-90 rates (all appearances)
    goals_per_90: Mapped[Optional[float]] = mapped_column(Numeric(7, 4))
    assists_per_90: Mapped[Optional[float]] = mapped_column(Numeric(7, 4))
    sot_per_90: Mapped[Optional[float]] = mapped_column(Numeric(7, 4))
    cards_per_90: Mapped[Optional[float]] = mapped_column(Numeric(7, 4))

    # Starter-only per-90 rates
    starter_goals_per_90: Mapped[Optional[float]] = mapped_column(Numeric(7, 4))
    starter_assists_per_90: Mapped[Optional[float]] = mapped_column(Numeric(7, 4))
    starter_sot_per_90: Mapped[Optional[float]] = mapped_column(Numeric(7, 4))
    starter_cards_per_90: Mapped[Optional[float]] = mapped_column(Numeric(7, 4))

    # Indices
    form_index: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    aggression_index_norm: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    control_index: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    expected_card_risk: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    expected_foul_pressure: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))

    total_minutes: Mapped[Optional[int]] = mapped_column(Integer)

    build_run_id: Mapped[Optional[int]] = mapped_column(ForeignKey("sync_runs.id", ondelete="SET NULL"))
    feature_version: Mapped[Optional[str]] = mapped_column(String(32))
    feature_build_id: Mapped[Optional[int]] = mapped_column(ForeignKey("feature_builds.id", ondelete="SET NULL"))
    as_of_fixture_api_id: Mapped[Optional[int]] = mapped_column(Integer)
    built_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    extras: Mapped[Optional[dict]] = mapped_column(JSONB)
