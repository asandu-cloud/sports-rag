"""Versioned, fixture-level Match Read records.

This is deliberately separate from ``Prediction``.  A prediction is one
priced market leg; a Match Read is the user-facing explanation of a fixture
and can contain several compatible legs, a no-bet decision, and (later) a
properly priced same-game package.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Index, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin
from .json_type import JSONB


class MatchRead(Base, TimestampMixin):
    """One immutable version of the model's fixture-level view."""

    __tablename__ = "match_reads"
    __table_args__ = (
        UniqueConstraint("read_key", name="uq_match_reads_key"),
        UniqueConstraint("fixture_api_id", "stage", "version", name="uq_match_reads_fixture_stage_version"),
        Index("ix_match_reads_fixture_stage", "fixture_api_id", "stage", "version"),
        Index("ix_match_reads_league_kickoff", "league", "kickoff"),
        Index("ix_match_reads_status", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Deterministic from the immutable read payload. Replays of the same
    # snapshot return the existing version instead of creating a duplicate.
    read_key: Mapped[str] = mapped_column(String(64), nullable=False)
    fixture_api_id: Mapped[str] = mapped_column(String(64), nullable=False)
    league: Mapped[str] = mapped_column(String(32), nullable=False)
    home_team: Mapped[str] = mapped_column(String(128), nullable=False)
    away_team: Mapped[str] = mapped_column(String(128), nullable=False)
    kickoff: Mapped[Optional[str]] = mapped_column(String(64))

    # ``pre_match`` and ``confirmed_lineups`` are separate version streams.
    # A lineup update therefore never silently rewrites an earlier read.
    stage: Mapped[str] = mapped_column(String(32), nullable=False, default="pre_match")
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    supersedes_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("match_reads.id", ondelete="RESTRICT")
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    thesis: Mapped[str] = mapped_column(Text, nullable=False)
    game_script_json: Mapped[Optional[dict]] = mapped_column(JSONB)

    input_snapshot_id: Mapped[str] = mapped_column(String(256), nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[Optional[str]] = mapped_column(String(128))
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    read_json: Mapped[dict] = mapped_column(JSONB, nullable=False)


class MatchReadSelection(Base, TimestampMixin):
    """A core, supporting, or alternative priced leg in a Match Read."""

    __tablename__ = "match_read_selections"
    __table_args__ = (
        UniqueConstraint("match_read_id", "position", name="uq_match_read_selection_position"),
        Index("ix_match_read_selections_read", "match_read_id", "position"),
        Index("ix_match_read_selections_published", "published_recommendation_id"),
        Index("ix_match_read_selections_role", "role"),
        Index("ix_match_read_selections_market", "market_group", "market_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_read_id: Mapped[int] = mapped_column(
        ForeignKey("match_reads.id", ondelete="CASCADE"), nullable=False
    )
    # Set only when this selection is deliberately released through the
    # existing immutable recommendation tracker.
    published_recommendation_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("published_recommendations.id", ondelete="RESTRICT")
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    market_group: Mapped[str] = mapped_column(String(32), nullable=False)
    market_key: Mapped[str] = mapped_column(String(128), nullable=False)
    pick: Mapped[str] = mapped_column(String(255), nullable=False)
    quote_odds: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)
    bookmaker: Mapped[Optional[str]] = mapped_column(String(64))
    model_probability: Mapped[Optional[float]] = mapped_column(Numeric(10, 6))
    value_edge: Mapped[Optional[float]] = mapped_column(Numeric(10, 6))
    selection_json: Mapped[dict] = mapped_column(JSONB, nullable=False)


class MatchReadPackage(Base, TimestampMixin):
    """A package of compatible selections, tracked independently from legs.

    Phase 0 stores actual same-game package pricing but intentionally leaves
    joint model probability and expected value nullable.  Those values must
    not be implied from multiplication of correlated leg probabilities.
    """

    __tablename__ = "match_read_packages"
    __table_args__ = (
        UniqueConstraint("package_key", name="uq_match_read_packages_key"),
        UniqueConstraint("match_read_id", "position", name="uq_match_read_package_position"),
        Index("ix_match_read_packages_read", "match_read_id", "position"),
        Index("ix_match_read_packages_status", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_read_id: Mapped[int] = mapped_column(
        ForeignKey("match_reads.id", ondelete="CASCADE"), nullable=False
    )
    package_key: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    rationale: Mapped[Optional[str]] = mapped_column(Text)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    quoted_odds: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)
    bookmaker: Mapped[str] = mapped_column(String(64), nullable=False)
    joint_model_probability: Mapped[Optional[float]] = mapped_column(Numeric(10, 6))
    expected_value: Mapped[Optional[float]] = mapped_column(Numeric(10, 6))
    stake_units: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False, default=1.0)
    profit_units: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    outcome: Mapped[Optional[str]] = mapped_column(String(16))
    settled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    package_json: Mapped[dict] = mapped_column(JSONB, nullable=False)


class MatchReadPackageSelection(Base, TimestampMixin):
    """The ordered selections included in one Match Read package."""

    __tablename__ = "match_read_package_selections"
    __table_args__ = (
        UniqueConstraint("package_id", "selection_id", name="uq_match_read_package_selection"),
        UniqueConstraint("package_id", "position", name="uq_match_read_package_selection_position"),
        Index("ix_match_read_package_selections_package", "package_id", "position"),
        Index("ix_match_read_package_selections_selection", "selection_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    package_id: Mapped[int] = mapped_column(
        ForeignKey("match_read_packages.id", ondelete="CASCADE"), nullable=False
    )
    selection_id: Mapped[int] = mapped_column(
        ForeignKey("match_read_selections.id", ondelete="CASCADE"), nullable=False
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)


class MatchReadDelivery(Base, TimestampMixin):
    """A user-visible delivery of a Match Read, including no-bet cards."""

    __tablename__ = "match_read_deliveries"
    __table_args__ = (
        UniqueConstraint("delivery_key", name="uq_match_read_delivery_key"),
        Index("ix_match_read_deliveries_read", "match_read_id"),
        Index("ix_match_read_deliveries_surface", "surface"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_read_id: Mapped[int] = mapped_column(
        ForeignKey("match_reads.id", ondelete="RESTRICT"), nullable=False
    )
    delivery_key: Mapped[str] = mapped_column(String(64), nullable=False)
    surface: Mapped[str] = mapped_column(String(32), nullable=False)
    external_reference: Mapped[Optional[str]] = mapped_column(String(256))
    delivered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB)
