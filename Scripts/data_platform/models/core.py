"""Core catalogue: competitions, seasons, teams, players."""

from __future__ import annotations

from datetime import date
from typing import List, Optional

from sqlalchemy import Boolean, Date, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin


class Competition(Base, TimestampMixin):
    """A competition (domestic league or continental cup)."""

    __tablename__ = "competitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    country: Mapped[Optional[str]] = mapped_column(String(64))
    api_football_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    competition_type: Mapped[str] = mapped_column(String(32), nullable=False, default="domestic_league")

    seasons: Mapped[List["Season"]] = relationship(back_populates="competition", cascade="all, delete-orphan")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Competition {self.code} ({self.api_football_id})>"


class Season(Base, TimestampMixin):
    """A season for a given competition. ``year`` follows API-Football: 2025 means 2025/26."""

    __tablename__ = "seasons"
    __table_args__ = (
        UniqueConstraint("competition_id", "year", name="uq_seasons_competition_year"),
        Index("ix_seasons_competition_current", "competition_id", "is_current"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id", ondelete="CASCADE"), nullable=False)
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str] = mapped_column(String(16), nullable=False)  # "2025/26"
    start_date: Mapped[Optional[date]] = mapped_column(Date)
    end_date: Mapped[Optional[date]] = mapped_column(Date)
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    competition: Mapped[Competition] = relationship(back_populates="seasons")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Season {self.label} competition_id={self.competition_id}>"


class Team(Base, TimestampMixin):
    """A team. Globally unique by ``api_football_id``."""

    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    api_football_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    canonical_name: Mapped[Optional[str]] = mapped_column(String(128))
    short_code: Mapped[Optional[str]] = mapped_column(String(16))
    country: Mapped[Optional[str]] = mapped_column(String(64))
    founded: Mapped[Optional[int]] = mapped_column(Integer)
    logo_url: Mapped[Optional[str]] = mapped_column(String(512))

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Team {self.name} ({self.api_football_id})>"


class Player(Base, TimestampMixin):
    """A player. Globally unique by ``api_football_id``."""

    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    api_football_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    position: Mapped[Optional[str]] = mapped_column(String(32))
    nationality: Mapped[Optional[str]] = mapped_column(String(64))
    birth_date: Mapped[Optional[date]] = mapped_column(Date)
    height_cm: Mapped[Optional[int]] = mapped_column(Integer)
    weight_kg: Mapped[Optional[int]] = mapped_column(Integer)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Player {self.name} ({self.api_football_id})>"
