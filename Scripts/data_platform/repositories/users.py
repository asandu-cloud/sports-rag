"""PostgreSQL-backed user / subscription / referral repository.

This is the canonical V1 replacement for ``Scripts/web_app/users.py``'s
direct SQLite access. The public surface intentionally mirrors the
existing module so the compat shim can route calls straight through
without app-code changes.

The repository returns plain dicts (not ORM instances) so callers using
the old ``sqlite3.Row``-based API continue to work.
"""

from __future__ import annotations

import json
import logging
import random
import string
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import session_scope
from ..models import AppReferral, AppSubscriptionEvent, AppUser

logger = logging.getLogger(__name__)


TIER_HIERARCHY: Dict[str, int] = {
    "free": 0,
    "starter": 1,
    "pro": 2,
    "elite": 3,
}

REFERRAL_REWARDS = [
    (1, "7_days", "7 days free added"),
    (3, "free_month", "1 free month"),
    (5, "tier_upgrade", "Permanent upgrade to next tier"),
    (10, "lifetime", "Lifetime free at current tier"),
]


def has_tier_access(user_tier: str, required_tier: str) -> bool:
    return TIER_HIERARCHY.get(user_tier, 0) >= TIER_HIERARCHY.get(required_tier, 0)


# Columns that are safe to set dynamically from upsert_user(**kwargs)
_ALLOWED_COLUMNS = frozenset({
    "discord_id", "discord_username", "discord_avatar_url", "email",
    "password_hash", "auth_provider",
    "stripe_customer_id", "stripe_subscription_id",
    "tier", "status", "trial_start", "trial_end",
    "subscription_start", "subscription_end", "cancelled_at",
    "is_founding_member",
    "referral_code", "referred_by", "referral_count",
})


def _user_to_dict(row: Optional[AppUser]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {
        "id": row.id,
        "discord_id": row.discord_id,
        "discord_username": row.discord_username,
        "discord_avatar_url": row.discord_avatar_url,
        "email": row.email,
        "stripe_customer_id": row.stripe_customer_id,
        "stripe_subscription_id": row.stripe_subscription_id,
        "tier": row.tier,
        "status": row.status,
        "trial_start": _iso(row.trial_start),
        "trial_end": _iso(row.trial_end),
        "subscription_start": _iso(row.subscription_start),
        "subscription_end": _iso(row.subscription_end),
        "cancelled_at": _iso(row.cancelled_at),
        "is_founding_member": 1 if row.is_founding_member else 0,
        "referral_code": row.referral_code,
        "referred_by": row.referred_by,
        "referral_count": row.referral_count or 0,
        "password_hash": row.password_hash,
        "auth_provider": row.auth_provider,
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
    }


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            try:
                return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except ValueError:
                return None
    return None


def _generate_code_string(username: Optional[str]) -> str:
    clean = "".join(ch for ch in (username or "USER") if ch.isalnum())[:5].upper()
    if not clean:
        clean = "USER"
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"SPIX-{clean}-{suffix}"


class UserRepository:
    """All user/subscription/referral operations in one place."""

    def __init__(self, session_factory=session_scope):
        # session_factory must be a context manager factory that yields a
        # SQLAlchemy Session.
        self._factory = session_factory

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------
    def upsert_user(
        self,
        discord_id: str,
        discord_username: Optional[str] = None,
        discord_avatar_url: Optional[str] = None,
        email: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        fields: Dict[str, Any] = {}
        if discord_username is not None:
            fields["discord_username"] = discord_username
        if discord_avatar_url is not None:
            fields["discord_avatar_url"] = discord_avatar_url
        if email is not None:
            fields["email"] = email
        for key, value in kwargs.items():
            if key not in _ALLOWED_COLUMNS:
                logger.warning("upsert_user: rejected unknown column '%s'", key)
                continue
            fields[key] = value

        datetime_fields = {"trial_start", "trial_end", "subscription_start", "subscription_end", "cancelled_at"}
        for key in list(fields):
            if key in datetime_fields:
                fields[key] = _parse_dt(fields[key])
        if "is_founding_member" in fields:
            fields["is_founding_member"] = bool(fields["is_founding_member"])

        try:
            with self._factory() as session:
                row = session.scalar(select(AppUser).where(AppUser.discord_id == discord_id))
                if row is None:
                    row = AppUser(discord_id=discord_id, **fields)
                    session.add(row)
                else:
                    for key, value in fields.items():
                        setattr(row, key, value)
                session.flush()
                session.refresh(row)
                return _user_to_dict(row)
        except Exception:
            logger.exception("upsert_user failed for discord_id=%s", discord_id)
            return None

    def get_user_by_discord_id(self, discord_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._factory() as session:
                row = session.scalar(select(AppUser).where(AppUser.discord_id == discord_id))
                return _user_to_dict(row)
        except Exception:
            logger.exception("get_user_by_discord_id failed for discord_id=%s", discord_id)
            return None

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        try:
            with self._factory() as session:
                row = session.get(AppUser, user_id)
                return _user_to_dict(row)
        except Exception:
            logger.exception("get_user_by_id failed for id=%s", user_id)
            return None

    def get_user_by_stripe_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._factory() as session:
                row = session.scalar(select(AppUser).where(AppUser.stripe_customer_id == customer_id))
                return _user_to_dict(row)
        except Exception:
            logger.exception("get_user_by_stripe_customer failed for customer_id=%s", customer_id)
            return None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        try:
            with self._factory() as session:
                row = session.scalar(
                    select(AppUser).where(AppUser.email == email, AppUser.auth_provider == "email")
                )
                return _user_to_dict(row)
        except Exception:
            logger.exception("get_user_by_email failed for email=%s", email)
            return None

    def create_email_user(self, email: str, password_hash: str) -> Optional[Dict[str, Any]]:
        try:
            with self._factory() as session:
                existing = session.scalar(
                    select(AppUser).where(AppUser.email == email, AppUser.auth_provider == "email")
                )
                if existing is not None:
                    return None
                row = AppUser(
                    email=email, password_hash=password_hash,
                    auth_provider="email", discord_id=f"email_{email}",
                    tier="free", status="active",
                )
                session.add(row)
                session.flush()
                session.refresh(row)
                return _user_to_dict(row)
        except Exception:
            logger.exception("create_email_user failed for email=%s", email)
            return None

    def update_user_subscription(
        self,
        discord_id: str,
        tier: str,
        status: str,
        stripe_customer_id: Optional[str] = None,
        stripe_subscription_id: Optional[str] = None,
        trial_start: Optional[str] = None,
        trial_end: Optional[str] = None,
        subscription_start: Optional[str] = None,
        subscription_end: Optional[str] = None,
        is_founding_member: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            with self._factory() as session:
                row = session.scalar(select(AppUser).where(AppUser.discord_id == discord_id))
                if row is None:
                    return None
                row.tier = tier
                row.status = status
                if stripe_customer_id is not None:
                    row.stripe_customer_id = stripe_customer_id
                if stripe_subscription_id is not None:
                    row.stripe_subscription_id = stripe_subscription_id
                if trial_start is not None:
                    row.trial_start = _parse_dt(trial_start)
                if trial_end is not None:
                    row.trial_end = _parse_dt(trial_end)
                if subscription_start is not None:
                    row.subscription_start = _parse_dt(subscription_start)
                if subscription_end is not None:
                    row.subscription_end = _parse_dt(subscription_end)
                if is_founding_member is not None:
                    row.is_founding_member = bool(is_founding_member)
                session.flush()
                return _user_to_dict(row)
        except Exception:
            logger.exception("update_user_subscription failed for discord_id=%s", discord_id)
            return None

    def cancel_user_subscription(self, discord_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._factory() as session:
                row = session.scalar(select(AppUser).where(AppUser.discord_id == discord_id))
                if row is None:
                    return None
                now = datetime.now(timezone.utc)
                row.status = "cancelled"
                row.cancelled_at = now
                session.flush()
                return _user_to_dict(row)
        except Exception:
            logger.exception("cancel_user_subscription failed for discord_id=%s", discord_id)
            return None

    def log_subscription_event(
        self,
        user_id: int,
        stripe_event_id: str,
        event_type: str,
        tier: Optional[str] = None,
        status: Optional[str] = None,
        raw_json: Optional[Any] = None,
    ) -> bool:
        try:
            with self._factory() as session:
                if isinstance(raw_json, str):
                    try:
                        raw_json = json.loads(raw_json)
                    except json.JSONDecodeError:
                        raw_json = {"_raw": raw_json}
                session.add(AppSubscriptionEvent(
                    user_id=user_id,
                    stripe_event_id=stripe_event_id,
                    event_type=event_type,
                    tier=tier,
                    status=status,
                    raw_json=raw_json,
                ))
            return True
        except Exception:
            logger.exception("log_subscription_event failed for user_id=%s event=%s", user_id, stripe_event_id)
            return False

    def get_active_subscribers(self, tier: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            with self._factory() as session:
                stmt = select(AppUser).where(AppUser.status.in_(["active", "trialing"]))
                if tier is not None:
                    stmt = stmt.where(AppUser.tier == tier)
                rows = session.scalars(stmt).all()
                return [d for d in (_user_to_dict(r) for r in rows) if d]
        except Exception:
            logger.exception("get_active_subscribers failed")
            return []

    # ------------------------------------------------------------------
    # Referrals
    # ------------------------------------------------------------------
    def generate_referral_code(self, discord_id: str) -> Optional[str]:
        try:
            with self._factory() as session:
                row = session.scalar(select(AppUser).where(AppUser.discord_id == discord_id))
                if row is None:
                    return None
                if row.referral_code:
                    return row.referral_code
                for _ in range(5):
                    code = _generate_code_string(row.discord_username)
                    collision = session.scalar(select(AppUser).where(AppUser.referral_code == code))
                    if collision is None:
                        row.referral_code = code
                        session.flush()
                        return code
                logger.error("generate_referral_code: no unique code after 5 tries for %s", discord_id)
                return None
        except Exception:
            logger.exception("generate_referral_code failed for discord_id=%s", discord_id)
            return None

    def get_user_by_referral_code(self, code: str) -> Optional[Dict[str, Any]]:
        try:
            with self._factory() as session:
                row = session.scalar(select(AppUser).where(AppUser.referral_code == code.upper()))
                return _user_to_dict(row)
        except Exception:
            logger.exception("get_user_by_referral_code failed for code=%s", code)
            return None

    def record_referral(
        self,
        referrer_discord_id: str,
        referred_discord_id: str,
        referred_username: Optional[str],
        referral_code_used: str,
        stripe_event_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if referrer_discord_id == referred_discord_id:
            logger.warning("record_referral: user %s tried to refer themselves", referrer_discord_id)
            return None
        try:
            with self._factory() as session:
                referred = session.scalar(select(AppUser).where(AppUser.discord_id == referred_discord_id))
                if referred is not None and referred.referred_by:
                    return None

                session.add(AppReferral(
                    referrer_discord_id=referrer_discord_id,
                    referred_discord_id=referred_discord_id,
                    referred_username=referred_username,
                    referral_code_used=referral_code_used.upper(),
                    stripe_event_id=stripe_event_id,
                ))

                referrer = session.scalar(select(AppUser).where(AppUser.discord_id == referrer_discord_id))
                if referrer is None:
                    return None
                referrer.referral_count = (referrer.referral_count or 0) + 1
                if referred is not None:
                    referred.referred_by = referrer_discord_id
                session.flush()
                new_count = referrer.referral_count
                tier = referrer.tier
            return {
                "referral_count": new_count,
                "referrer_tier": tier,
                "reward": _check_reward_threshold(new_count),
            }
        except Exception:
            logger.exception("record_referral failed: referrer=%s referred=%s",
                             referrer_discord_id, referred_discord_id)
            return None

    def grant_referral_reward(self, referrer_discord_id: str, reward_type: str) -> bool:
        try:
            with self._factory() as session:
                latest = session.scalar(
                    select(AppReferral)
                    .where(AppReferral.referrer_discord_id == referrer_discord_id,
                           AppReferral.reward_granted.is_(False))
                    .order_by(AppReferral.created_at.desc())
                    .limit(1)
                )
                if latest is None:
                    return False
                latest.reward_type = reward_type
                latest.reward_granted = True
            return True
        except Exception:
            logger.exception("grant_referral_reward failed for %s", referrer_discord_id)
            return False

    def get_referral_stats(self, discord_id: str) -> Dict[str, Any]:
        default: Dict[str, Any] = {
            "total_referrals": 0, "reward_tier": None,
            "next_reward_at": None, "next_reward_description": None,
            "referral_code": None, "referrals": [],
        }
        try:
            with self._factory() as session:
                user = session.scalar(select(AppUser).where(AppUser.discord_id == discord_id))
                if user is None:
                    return default
                count = user.referral_count or 0
                code = user.referral_code

                current_reward = None
                next_reward_at = None
                next_reward_desc = None
                for threshold, reward_type, description in REFERRAL_REWARDS:
                    if count >= threshold:
                        current_reward = reward_type
                    else:
                        if next_reward_at is None:
                            next_reward_at = threshold
                            next_reward_desc = description
                        break
                if count >= REFERRAL_REWARDS[-1][0]:
                    next_reward_at = None
                    next_reward_desc = None

                ref_rows = session.scalars(
                    select(AppReferral)
                    .where(AppReferral.referrer_discord_id == discord_id)
                    .order_by(AppReferral.created_at.desc())
                    .limit(20)
                ).all()
                referrals = [{
                    "referred_discord_id": r.referred_discord_id,
                    "referred_username": r.referred_username,
                    "reward_type": r.reward_type,
                    "reward_granted": 1 if r.reward_granted else 0,
                    "created_at": _iso(r.created_at),
                } for r in ref_rows]
            return {
                "total_referrals": count,
                "reward_tier": current_reward,
                "next_reward_at": next_reward_at,
                "next_reward_description": next_reward_desc,
                "referral_code": code,
                "referrals": referrals,
            }
        except Exception:
            logger.exception("get_referral_stats failed for discord_id=%s", discord_id)
            return default


def _check_reward_threshold(count: int) -> Optional[Dict[str, Any]]:
    for threshold, reward_type, description in REFERRAL_REWARDS:
        if count == threshold:
            return {
                "reward_type": reward_type,
                "description": description,
                "threshold": threshold,
            }
    return None
