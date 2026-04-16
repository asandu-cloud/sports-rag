"""Canonical app users / subscriptions / referrals.

These replace the ``users``, ``subscription_events``, ``referrals`` tables
in ``Index/predictions.db`` over time. In V1 they live alongside SQLite
and the website/bot only use them when ``PLATFORM_ENABLED=1`` via the
compat shim in ``Scripts/web_app/users.py``.

Table names are prefixed with ``app_`` to avoid ambiguity with the
football-data catalogue (teams, players, ...).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin
from .json_type import JSONB


class AppUser(Base, TimestampMixin):
    __tablename__ = "app_users"
    __table_args__ = (
        UniqueConstraint("discord_id", name="uq_app_users_discord_id"),
        UniqueConstraint("stripe_customer_id", name="uq_app_users_stripe_customer"),
        UniqueConstraint("referral_code", name="uq_app_users_referral_code"),
        # Email uniqueness only matters for email-auth users; emulated in
        # application code because a partial index differs per dialect.
        Index("ix_app_users_email_provider", "email", "auth_provider"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    discord_id: Mapped[Optional[str]] = mapped_column(String(64))
    discord_username: Mapped[Optional[str]] = mapped_column(String(128))
    discord_avatar_url: Mapped[Optional[str]] = mapped_column(Text)
    email: Mapped[Optional[str]] = mapped_column(String(255))
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(String(64))
    stripe_subscription_id: Mapped[Optional[str]] = mapped_column(String(64))
    tier: Mapped[str] = mapped_column(String(32), nullable=False, default="free")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="inactive")
    trial_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    trial_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    subscription_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    subscription_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    cancelled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    is_founding_member: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    referral_code: Mapped[Optional[str]] = mapped_column(String(64))
    referred_by: Mapped[Optional[str]] = mapped_column(String(64))
    referral_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    password_hash: Mapped[Optional[str]] = mapped_column(Text)
    auth_provider: Mapped[str] = mapped_column(String(32), nullable=False, default="discord")


class AppSubscriptionEvent(Base, TimestampMixin):
    __tablename__ = "app_subscription_events"
    __table_args__ = (
        UniqueConstraint("stripe_event_id", name="uq_app_sub_events_stripe_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("app_users.id", ondelete="CASCADE"), nullable=False)
    stripe_event_id: Mapped[str] = mapped_column(String(128), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    tier: Mapped[Optional[str]] = mapped_column(String(32))
    status: Mapped[Optional[str]] = mapped_column(String(32))
    raw_json: Mapped[Optional[dict]] = mapped_column(JSONB)


class AppReferral(Base, TimestampMixin):
    __tablename__ = "app_referrals"
    __table_args__ = (
        Index("ix_app_referrals_referrer", "referrer_discord_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    referrer_discord_id: Mapped[str] = mapped_column(String(64), nullable=False)
    referred_discord_id: Mapped[str] = mapped_column(String(64), nullable=False)
    referred_username: Mapped[Optional[str]] = mapped_column(String(128))
    referral_code_used: Mapped[str] = mapped_column(String(64), nullable=False)
    stripe_event_id: Mapped[Optional[str]] = mapped_column(String(128))
    reward_type: Mapped[Optional[str]] = mapped_column(String(64))
    reward_granted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
