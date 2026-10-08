"""Disposable, versioned research ranks; never prediction or settlement inputs."""
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class ResearchProfileBuild(Base):
    __tablename__ = "research_profile_builds"

    scope_key: Mapped[str] = mapped_column(String(96), primary_key=True)
    competition_id: Mapped[Optional[int]] = mapped_column(ForeignKey("competitions.id"))
    season_id: Mapped[Optional[int]] = mapped_column(ForeignKey("seasons.id"))
    season_year: Mapped[int] = mapped_column(Integer, nullable=False)
    build_id: Mapped[str] = mapped_column(String(64), nullable=False)
    built_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ResearchPeerStat(Base):
    __tablename__ = "research_peer_stats"
    __table_args__ = (
        UniqueConstraint("scope_key", "subject_type", "subject_key", "peer_group", "metric",
                         name="uq_research_peer_metric"),
        Index("ix_research_peer_subject", "subject_type", "subject_key", "season_year"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scope_key: Mapped[str] = mapped_column(ForeignKey("research_profile_builds.scope_key"), nullable=False)
    subject_type: Mapped[str] = mapped_column(String(16), nullable=False)
    subject_key: Mapped[str] = mapped_column(String(128), nullable=False)
    competition_id: Mapped[Optional[int]] = mapped_column(ForeignKey("competitions.id"))
    # A referee's cross-competition season has no single Season FK.
    season_id: Mapped[Optional[int]] = mapped_column(ForeignKey("seasons.id"))
    season_year: Mapped[int] = mapped_column(Integer, nullable=False)
    peer_group: Mapped[str] = mapped_column(String(32), nullable=False)
    metric: Mapped[str] = mapped_column(String(48), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    percentile: Mapped[int] = mapped_column(Integer, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    peer_count: Mapped[int] = mapped_column(Integer, nullable=False)
    league_average: Mapped[Optional[float]] = mapped_column(Float)
    sample_minutes: Mapped[Optional[int]] = mapped_column(Integer)
    sample_matches: Mapped[int] = mapped_column(Integer, nullable=False)
    build_id: Mapped[str] = mapped_column(String(64), nullable=False)
    built_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ResearchRefereeAlias(Base):
    __tablename__ = "research_referee_aliases"
    raw_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    referee_key: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
