"""Compatibility helpers for moving ``web_app/users.py`` off SQLite.

Two responsibilities:

1. Expose a singleton ``UserRepository`` and an ``is_platform_enabled``
   flag so the legacy module can route reads/writes through the new
   store when the platform is turned on.

2. Provide a one-off migration routine to copy users, referrals, and
   subscription events out of ``Index/predictions.db`` into the
   canonical tables. Idempotent on ``discord_id`` / ``stripe_event_id``.
"""

from __future__ import annotations

import logging
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from .. import config as _config
from ..db import session_scope
from ..models import AppReferral, AppSubscriptionEvent, AppUser
from ..repositories.users import UserRepository, _parse_dt

logger = logging.getLogger(__name__)


def is_platform_enabled() -> bool:
    """Has the operator opted into routing traffic through PostgreSQL yet?

    Re-reads ``data_platform.config.SETTINGS`` each call so changing the
    env + rebinding ``SETTINGS`` at runtime (e.g. in tests) takes effect
    immediately.
    """
    return _config.SETTINGS.platform_enabled


@lru_cache(maxsize=1)
def get_user_repository() -> UserRepository:
    """Return the process-wide repo instance."""
    return UserRepository()


# ---------------------------------------------------------------------------
# One-off migration from Index/predictions.db → Postgres
# ---------------------------------------------------------------------------


def _sqlite_rows(conn: sqlite3.Connection, sql: str) -> List[Dict[str, Any]]:
    cur = conn.execute(sql)
    return [dict(r) for r in cur.fetchall()]


def _iso(value: Any) -> Any:
    return value


def migrate_sqlite_users(
    sqlite_path: Optional[Path] = None,
    *,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Copy users / subscription_events / referrals from SQLite to Postgres.

    Idempotent: existing rows (matched on ``discord_id``, ``stripe_event_id``,
    or the ``referrer+referred`` pair) are updated in-place rather than
    duplicated. Safe to run repeatedly.
    """
    sqlite_path = Path(sqlite_path) if sqlite_path else _config.SETTINGS.project_root / "Index" / "predictions.db"
    if not sqlite_path.exists():
        logger.info("No predictions.db at %s; nothing to migrate.", sqlite_path)
        return {"users": 0, "events": 0, "referrals": 0, "skipped": 0}

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row

    users = _sqlite_rows(conn, "SELECT * FROM users")
    events = _sqlite_rows(conn, "SELECT * FROM subscription_events") if _table_exists(conn, "subscription_events") else []
    referrals = _sqlite_rows(conn, "SELECT * FROM referrals") if _table_exists(conn, "referrals") else []
    conn.close()

    counts = {"users": 0, "events": 0, "referrals": 0, "skipped": 0}
    if dry_run:
        return {"users": len(users), "events": len(events), "referrals": len(referrals), "skipped": 0, "dry_run": True}

    with session_scope() as session:
        # --- users ---------------------------------------------------------
        for row in users:
            discord_id = row.get("discord_id")
            if not discord_id:
                counts["skipped"] += 1
                continue
            existing = session.scalar(select(AppUser).where(AppUser.discord_id == discord_id))
            fields = {
                "discord_username": row.get("discord_username"),
                "discord_avatar_url": row.get("discord_avatar_url"),
                "email": row.get("email"),
                "stripe_customer_id": row.get("stripe_customer_id"),
                "stripe_subscription_id": row.get("stripe_subscription_id"),
                "tier": row.get("tier") or "free",
                "status": row.get("status") or "inactive",
                "trial_start": _parse_dt(row.get("trial_start")),
                "trial_end": _parse_dt(row.get("trial_end")),
                "subscription_start": _parse_dt(row.get("subscription_start")),
                "subscription_end": _parse_dt(row.get("subscription_end")),
                "cancelled_at": _parse_dt(row.get("cancelled_at")),
                "is_founding_member": bool(row.get("is_founding_member") or 0),
                "referral_code": row.get("referral_code"),
                "referred_by": row.get("referred_by"),
                "referral_count": int(row.get("referral_count") or 0),
                "password_hash": row.get("password_hash"),
                "auth_provider": row.get("auth_provider") or "discord",
            }
            if existing is None:
                session.add(AppUser(discord_id=discord_id, **fields))
            else:
                for key, value in fields.items():
                    setattr(existing, key, value)
            counts["users"] += 1
        session.flush()

        # --- subscription events ------------------------------------------
        old_to_new_user: Dict[int, int] = {}
        if events:
            new_users = session.scalars(select(AppUser)).all()
            discord_to_new = {u.discord_id: u.id for u in new_users if u.discord_id}
            for ur in users:
                new_id = discord_to_new.get(ur.get("discord_id"))
                if new_id is not None:
                    old_to_new_user[int(ur["id"])] = new_id

        for row in events:
            stripe_eid = row.get("stripe_event_id")
            if not stripe_eid:
                counts["skipped"] += 1
                continue
            existing = session.scalar(
                select(AppSubscriptionEvent).where(AppSubscriptionEvent.stripe_event_id == stripe_eid)
            )
            new_uid = old_to_new_user.get(int(row.get("user_id") or 0))
            if new_uid is None:
                counts["skipped"] += 1
                continue
            raw_json = row.get("raw_json")
            payload = None
            if raw_json:
                try:
                    import json as _json
                    payload = _json.loads(raw_json)
                except Exception:
                    payload = {"_raw": raw_json}
            fields = {
                "user_id": new_uid,
                "event_type": row.get("event_type") or "unknown",
                "tier": row.get("tier"),
                "status": row.get("status"),
                "raw_json": payload,
            }
            if existing is None:
                session.add(AppSubscriptionEvent(stripe_event_id=stripe_eid, **fields))
            else:
                for key, value in fields.items():
                    setattr(existing, key, value)
            counts["events"] += 1

        # --- referrals ----------------------------------------------------
        for row in referrals:
            referrer = row.get("referrer_discord_id")
            referred = row.get("referred_discord_id")
            if not referrer or not referred:
                counts["skipped"] += 1
                continue
            existing = session.scalar(
                select(AppReferral).where(
                    AppReferral.referrer_discord_id == referrer,
                    AppReferral.referred_discord_id == referred,
                )
            )
            fields = {
                "referred_username": row.get("referred_username"),
                "referral_code_used": (row.get("referral_code_used") or "").upper() or "LEGACY",
                "stripe_event_id": row.get("stripe_event_id"),
                "reward_type": row.get("reward_type"),
                "reward_granted": bool(row.get("reward_granted") or 0),
            }
            if existing is None:
                session.add(AppReferral(
                    referrer_discord_id=referrer,
                    referred_discord_id=referred,
                    **fields,
                ))
            else:
                for key, value in fields.items():
                    setattr(existing, key, value)
            counts["referrals"] += 1

    return counts


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (name,)
    )
    return cur.fetchone() is not None
