"""Parlay session table — canonical replacement for parlay_sessions in predictions.db."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin
from .json_type import JSONB


class ParlaySession(Base, TimestampMixin):
    __tablename__ = "parlay_sessions"
    __table_args__ = (
        Index("ix_parlay_sessions_user", "discord_user_id"),
        Index("ix_parlay_sessions_thread", "thread_id"),
        Index("ix_parlay_sessions_parent", "parent_session_id"),
        Index("ix_parlay_sessions_message", "message_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    parent_session_id: Mapped[Optional[int]] = mapped_column(Integer)
    discord_user_id: Mapped[Optional[int]] = mapped_column(Integer)
    guild_id: Mapped[Optional[int]] = mapped_column(Integer)
    channel_id: Mapped[Optional[int]] = mapped_column(Integer)
    thread_id: Mapped[Optional[int]] = mapped_column(Integer)
    message_id: Mapped[Optional[int]] = mapped_column(Integer)
    source_command: Mapped[str] = mapped_column(String(64), nullable=False)
    action_type: Mapped[str] = mapped_column(String(64), nullable=False)
    request_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    result_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
