"""Fixtures and per-fixture stats (team + player)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin
from .json_type import JSONB


class Fixture(Base, TimestampMixin):
    """A single fixture. Mirrors API-Football's ``/fixtures`` row."""

    __tablename__ = "fixtures"
    __table_args__ = (
        Index("ix_fixtures_season_kickoff", "season_id", "kickoff_utc"),
        Index("ix_fixtures_competition_status", "competition_id", "status"),
        Index("ix_fixtures_home_team", "home_team_id"),
        Index("ix_fixtures_away_team", "away_team_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    api_football_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id", ondelete="RESTRICT"), nullable=False)
    season_id: Mapped[int] = mapped_column(ForeignKey("seasons.id", ondelete="RESTRICT"), nullable=False)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id", ondelete="RESTRICT"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id", ondelete="RESTRICT"), nullable=False)

    kickoff_utc: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    venue: Mapped[Optional[str]] = mapped_column(String(128))
    city: Mapped[Optional[str]] = mapped_column(String(64))
    status: Mapped[Optional[str]] = mapped_column(String(16))  # NS, 1H, HT, 2H, FT, AET, PEN, CANC, PST, ...
    elapsed: Mapped[Optional[int]] = mapped_column(Integer)
    home_goals: Mapped[Optional[int]] = mapped_column(Integer)
    away_goals: Mapped[Optional[int]] = mapped_column(Integer)
    round: Mapped[Optional[str]] = mapped_column(String(64))
    referee: Mapped[Optional[str]] = mapped_column(String(128))

    last_fetched_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_api_updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    payload_digest: Mapped[Optional[str]] = mapped_column(String(64))


class FixtureTeamStats(Base, TimestampMixin):
    """Per-fixture team statistics.

    Column names follow the existing variable vocabulary where possible so
    that downstream feature engineering can migrate to reading from this
    table without renaming.
    """

    __tablename__ = "fixture_team_stats"
    __table_args__ = (
        UniqueConstraint("fixture_id", "team_id", name="uq_team_stats_fixture_team"),
        Index("ix_team_stats_team", "team_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id", ondelete="CASCADE"), nullable=False)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id", ondelete="RESTRICT"), nullable=False)
    opponent_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id", ondelete="RESTRICT"), nullable=False)
    is_home: Mapped[bool] = mapped_column(Boolean, nullable=False)

    goals: Mapped[Optional[int]] = mapped_column(Integer)
    shots_total: Mapped[Optional[int]] = mapped_column(Integer)
    shots_on: Mapped[Optional[int]] = mapped_column(Integer)
    shots_off: Mapped[Optional[int]] = mapped_column(Integer)
    shots_blocked: Mapped[Optional[int]] = mapped_column(Integer)
    shots_inside_box: Mapped[Optional[int]] = mapped_column(Integer)
    shots_outside_box: Mapped[Optional[int]] = mapped_column(Integer)
    fouls_committed: Mapped[Optional[int]] = mapped_column(Integer)
    corners: Mapped[Optional[int]] = mapped_column(Integer)
    offsides: Mapped[Optional[int]] = mapped_column(Integer)
    possession: Mapped[Optional[float]] = mapped_column(Numeric(6, 4))
    yellow_cards: Mapped[Optional[int]] = mapped_column(Integer)
    red_cards: Mapped[Optional[int]] = mapped_column(Integer)
    goalkeeper_saves: Mapped[Optional[int]] = mapped_column(Integer)
    passes_total: Mapped[Optional[int]] = mapped_column(Integer)
    passes_accurate: Mapped[Optional[int]] = mapped_column(Integer)
    pass_accuracy: Mapped[Optional[float]] = mapped_column(Numeric(6, 4))
    expected_goals: Mapped[Optional[float]] = mapped_column(Numeric(7, 4))
    goals_prevented: Mapped[Optional[float]] = mapped_column(Numeric(7, 4))

    raw_payload_digest: Mapped[Optional[str]] = mapped_column(String(64))
    stats_json: Mapped[Optional[dict]] = mapped_column(JSONB)


class FixturePlayerStats(Base, TimestampMixin):
    """Per-fixture per-player statistics."""

    __tablename__ = "fixture_player_stats"
    __table_args__ = (
        UniqueConstraint("fixture_id", "player_id", name="uq_player_stats_fixture_player"),
        Index("ix_player_stats_player", "player_id"),
        Index("ix_player_stats_team", "team_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id", ondelete="CASCADE"), nullable=False)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id", ondelete="RESTRICT"), nullable=False)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id", ondelete="RESTRICT"), nullable=False)

    position: Mapped[Optional[str]] = mapped_column(String(32))
    minutes: Mapped[Optional[int]] = mapped_column(Integer)
    rating: Mapped[Optional[float]] = mapped_column(Numeric(4, 2))
    captain: Mapped[Optional[bool]] = mapped_column(Boolean)
    substitute: Mapped[Optional[bool]] = mapped_column(Boolean)

    goals: Mapped[Optional[int]] = mapped_column(Integer)
    assists: Mapped[Optional[int]] = mapped_column(Integer)
    shots_total: Mapped[Optional[int]] = mapped_column(Integer)
    shots_on: Mapped[Optional[int]] = mapped_column(Integer)

    passes_total: Mapped[Optional[int]] = mapped_column(Integer)
    passes_accurate: Mapped[Optional[int]] = mapped_column(Integer)
    pass_accuracy: Mapped[Optional[float]] = mapped_column(Numeric(6, 4))

    tackles: Mapped[Optional[int]] = mapped_column(Integer)
    interceptions: Mapped[Optional[int]] = mapped_column(Integer)
    duels_won: Mapped[Optional[int]] = mapped_column(Integer)
    duels_total: Mapped[Optional[int]] = mapped_column(Integer)

    yellow_cards: Mapped[Optional[int]] = mapped_column(Integer)
    red_cards: Mapped[Optional[int]] = mapped_column(Integer)
    fouls_committed: Mapped[Optional[int]] = mapped_column(Integer)
    fouls_drawn: Mapped[Optional[int]] = mapped_column(Integer)

    raw_payload_digest: Mapped[Optional[str]] = mapped_column(String(64))
    stats_json: Mapped[Optional[dict]] = mapped_column(JSONB)
