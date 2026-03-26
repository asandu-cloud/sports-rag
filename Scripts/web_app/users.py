"""
User database model and CRUD operations for the Matchwise subscription system.

Shared by both the Discord bot and the FastAPI web app via the same SQLite
database at Index/predictions.db.  All functions handle errors gracefully,
logging warnings and returning safe defaults rather than raising.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path defaults
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = _PROJECT_ROOT / "Index" / "predictions.db"

# ---------------------------------------------------------------------------
# Tier hierarchy
# ---------------------------------------------------------------------------

TIER_HIERARCHY: Dict[str, int] = {
    "free": 0,
    "starter": 1,
    "pro": 2,
    "elite": 3,
}


def has_tier_access(user_tier: str, required_tier: str) -> bool:
    """Return True if *user_tier* meets or exceeds *required_tier*."""
    return TIER_HIERARCHY.get(user_tier, 0) >= TIER_HIERARCHY.get(required_tier, 0)


# ---------------------------------------------------------------------------
# SQL DDL
# ---------------------------------------------------------------------------

_SCHEMA_USERS = """\
CREATE TABLE IF NOT EXISTS users (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    discord_id              TEXT    NOT NULL UNIQUE,
    discord_username        TEXT,
    discord_avatar_url      TEXT,
    email                   TEXT,
    stripe_customer_id      TEXT    UNIQUE,
    stripe_subscription_id  TEXT,
    tier                    TEXT    NOT NULL DEFAULT 'free',
    status                  TEXT    NOT NULL DEFAULT 'inactive',
    trial_start             TEXT,
    trial_end               TEXT,
    subscription_start      TEXT,
    subscription_end        TEXT,
    cancelled_at            TEXT,
    is_founding_member      INTEGER NOT NULL DEFAULT 0,
    created_at              TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at              TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

_SCHEMA_EVENTS = """\
CREATE TABLE IF NOT EXISTS subscription_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id),
    stripe_event_id TEXT    NOT NULL UNIQUE,
    event_type      TEXT    NOT NULL,
    tier            TEXT,
    status          TEXT,
    raw_json        TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def _resolve_path(db_path: Optional[str | Path] = None) -> Path:
    """Resolve the database file path, falling back to the project default."""
    return Path(db_path) if db_path else DB_PATH


def get_db(db_path: Optional[str | Path] = None) -> sqlite3.Connection:
    """Return a connection with WAL mode and ``sqlite3.Row`` row factory.

    Parameters
    ----------
    db_path:
        Optional override for the database file location.  Defaults to
        ``Index/predictions.db`` relative to the project root.
    """
    path = _resolve_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: Optional[str | Path] = None) -> None:
    """Create tables if they do not already exist (idempotent).

    Parameters
    ----------
    db_path:
        Optional override for the database file location.
    """
    try:
        conn = get_db(db_path)
        conn.executescript(_SCHEMA_USERS + _SCHEMA_EVENTS)
        conn.commit()
        conn.close()
        logger.info("Database initialised at %s", _resolve_path(db_path))
    except Exception:
        logger.exception("Failed to initialise database")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    """Convert a ``sqlite3.Row`` to a plain dict, or return ``None``."""
    if row is None:
        return None
    return dict(row)


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def upsert_user(
    discord_id: str,
    discord_username: Optional[str] = None,
    discord_avatar_url: Optional[str] = None,
    email: Optional[str] = None,
    *,
    db_path: Optional[str | Path] = None,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Insert or update a user keyed by *discord_id*.

    Any additional keyword arguments whose names match column names on the
    ``users`` table will be included in the upsert.

    Returns the resulting user dict, or ``None`` on failure.
    """
    try:
        conn = get_db(db_path)

        # Merge explicit args with kwargs ---------------------------------
        fields: Dict[str, Any] = {}
        if discord_username is not None:
            fields["discord_username"] = discord_username
        if discord_avatar_url is not None:
            fields["discord_avatar_url"] = discord_avatar_url
        if email is not None:
            fields["email"] = email
        fields.update(kwargs)
        fields["updated_at"] = _now_iso()

        existing = conn.execute(
            "SELECT * FROM users WHERE discord_id = ?", (discord_id,)
        ).fetchone()

        if existing is None:
            # INSERT -------------------------------------------------------
            fields["discord_id"] = discord_id
            cols = ", ".join(fields.keys())
            placeholders = ", ".join("?" for _ in fields)
            conn.execute(
                f"INSERT INTO users ({cols}) VALUES ({placeholders})",
                list(fields.values()),
            )
        else:
            # UPDATE -------------------------------------------------------
            set_clause = ", ".join(f"{k} = ?" for k in fields)
            conn.execute(
                f"UPDATE users SET {set_clause} WHERE discord_id = ?",
                [*fields.values(), discord_id],
            )

        conn.commit()
        row = conn.execute(
            "SELECT * FROM users WHERE discord_id = ?", (discord_id,)
        ).fetchone()
        conn.close()
        return _row_to_dict(row)

    except Exception:
        logger.exception("upsert_user failed for discord_id=%s", discord_id)
        return None


def get_user_by_discord_id(
    discord_id: str,
    *,
    db_path: Optional[str | Path] = None,
) -> Optional[Dict[str, Any]]:
    """Return the user dict matching *discord_id*, or ``None``."""
    try:
        conn = get_db(db_path)
        row = conn.execute(
            "SELECT * FROM users WHERE discord_id = ?", (discord_id,)
        ).fetchone()
        conn.close()
        return _row_to_dict(row)
    except Exception:
        logger.exception(
            "get_user_by_discord_id failed for discord_id=%s", discord_id
        )
        return None


def get_user_by_stripe_customer(
    customer_id: str,
    *,
    db_path: Optional[str | Path] = None,
) -> Optional[Dict[str, Any]]:
    """Return the user dict matching *stripe_customer_id*, or ``None``."""
    try:
        conn = get_db(db_path)
        row = conn.execute(
            "SELECT * FROM users WHERE stripe_customer_id = ?", (customer_id,)
        ).fetchone()
        conn.close()
        return _row_to_dict(row)
    except Exception:
        logger.exception(
            "get_user_by_stripe_customer failed for customer_id=%s",
            customer_id,
        )
        return None


def update_user_subscription(
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
    *,
    db_path: Optional[str | Path] = None,
) -> Optional[Dict[str, Any]]:
    """Update subscription-related fields for a user.

    Returns the updated user dict, or ``None`` on failure.
    """
    try:
        conn = get_db(db_path)
        fields: Dict[str, Any] = {
            "tier": tier,
            "status": status,
            "updated_at": _now_iso(),
        }
        if stripe_customer_id is not None:
            fields["stripe_customer_id"] = stripe_customer_id
        if stripe_subscription_id is not None:
            fields["stripe_subscription_id"] = stripe_subscription_id
        if trial_start is not None:
            fields["trial_start"] = trial_start
        if trial_end is not None:
            fields["trial_end"] = trial_end
        if subscription_start is not None:
            fields["subscription_start"] = subscription_start
        if subscription_end is not None:
            fields["subscription_end"] = subscription_end
        if is_founding_member is not None:
            fields["is_founding_member"] = is_founding_member

        set_clause = ", ".join(f"{k} = ?" for k in fields)
        conn.execute(
            f"UPDATE users SET {set_clause} WHERE discord_id = ?",
            [*fields.values(), discord_id],
        )
        conn.commit()

        row = conn.execute(
            "SELECT * FROM users WHERE discord_id = ?", (discord_id,)
        ).fetchone()
        conn.close()
        return _row_to_dict(row)

    except Exception:
        logger.exception(
            "update_user_subscription failed for discord_id=%s", discord_id
        )
        return None


def cancel_user_subscription(
    discord_id: str,
    *,
    db_path: Optional[str | Path] = None,
) -> Optional[Dict[str, Any]]:
    """Mark a subscription as cancelled (status='cancelled', cancelled_at=now).

    Returns the updated user dict, or ``None`` on failure.
    """
    try:
        conn = get_db(db_path)
        now = _now_iso()
        conn.execute(
            "UPDATE users SET status = 'cancelled', cancelled_at = ?, "
            "updated_at = ? WHERE discord_id = ?",
            (now, now, discord_id),
        )
        conn.commit()

        row = conn.execute(
            "SELECT * FROM users WHERE discord_id = ?", (discord_id,)
        ).fetchone()
        conn.close()
        return _row_to_dict(row)

    except Exception:
        logger.exception(
            "cancel_user_subscription failed for discord_id=%s", discord_id
        )
        return None


def log_subscription_event(
    user_id: int,
    stripe_event_id: str,
    event_type: str,
    tier: Optional[str] = None,
    status: Optional[str] = None,
    raw_json: Optional[Any] = None,
    *,
    db_path: Optional[str | Path] = None,
) -> bool:
    """Insert an audit row into ``subscription_events``.

    *raw_json* may be a dict/list (will be JSON-serialised) or a string.

    Returns ``True`` on success, ``False`` on failure.
    """
    try:
        conn = get_db(db_path)
        if raw_json is not None and not isinstance(raw_json, str):
            raw_json = json.dumps(raw_json)

        conn.execute(
            "INSERT INTO subscription_events "
            "(user_id, stripe_event_id, event_type, tier, status, raw_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, stripe_event_id, event_type, tier, status, raw_json),
        )
        conn.commit()
        conn.close()
        return True

    except Exception:
        logger.exception(
            "log_subscription_event failed for user_id=%s event=%s",
            user_id,
            stripe_event_id,
        )
        return False


def get_active_subscribers(
    tier: Optional[str] = None,
    *,
    db_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Return all users whose status is 'active' or 'trialing'.

    Parameters
    ----------
    tier:
        If provided, further filter by this tier value.
    """
    try:
        conn = get_db(db_path)
        query = "SELECT * FROM users WHERE status IN ('active', 'trialing')"
        params: list[Any] = []

        if tier is not None:
            query += " AND tier = ?"
            params.append(tier)

        rows = conn.execute(query, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    except Exception:
        logger.exception("get_active_subscribers failed")
        return []
