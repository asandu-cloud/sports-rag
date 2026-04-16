"""Migration from legacy Index/predictions.db into canonical app tables.

Creates a throwaway SQLite file with the old schema, seeds it, runs
``migrate_sqlite_users``, then asserts the canonical tables match. Also
proves idempotency by migrating twice.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def _seed_legacy_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            discord_id TEXT NOT NULL UNIQUE,
            discord_username TEXT,
            tier TEXT NOT NULL DEFAULT 'free',
            status TEXT NOT NULL DEFAULT 'inactive',
            referral_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE subscription_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            stripe_event_id TEXT NOT NULL UNIQUE,
            event_type TEXT NOT NULL,
            tier TEXT,
            status TEXT,
            raw_json TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE referrals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            referrer_discord_id TEXT NOT NULL,
            referred_discord_id TEXT NOT NULL,
            referred_username TEXT,
            referral_code_used TEXT NOT NULL,
            stripe_event_id TEXT,
            reward_type TEXT,
            reward_granted INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    conn.executemany(
        "INSERT INTO users (discord_id, discord_username, tier, status, referral_count) VALUES (?, ?, ?, ?, ?)",
        [
            ("u-1", "alice", "pro", "active", 2),
            ("u-2", "bob", "free", "inactive", 0),
        ],
    )
    conn.executemany(
        "INSERT INTO subscription_events (user_id, stripe_event_id, event_type, tier, status, raw_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, "evt_1", "customer.subscription.created", "pro", "active", '{"k": "v"}'),
        ],
    )
    conn.executemany(
        "INSERT INTO referrals (referrer_discord_id, referred_discord_id, referred_username, "
        "referral_code_used, reward_granted) VALUES (?, ?, ?, ?, ?)",
        [
            ("u-1", "u-2", "bob", "SPIX-ALICE-X1Y2", 0),
        ],
    )
    conn.commit()
    conn.close()


def test_migrate_users_idempotent(tmp_path, session_factory):
    legacy = tmp_path / "predictions.db"
    _seed_legacy_db(legacy)

    from data_platform.compat import migrate_sqlite_users

    counts_1 = migrate_sqlite_users(sqlite_path=legacy)
    counts_2 = migrate_sqlite_users(sqlite_path=legacy)
    assert counts_1["users"] == 2
    assert counts_1["events"] == 1
    assert counts_1["referrals"] == 1
    assert counts_2["users"] == 2  # re-migrated, still 2 users

    from data_platform.models import AppReferral, AppSubscriptionEvent, AppUser
    with session_factory() as s:
        assert s.query(AppUser).count() == 2
        assert s.query(AppSubscriptionEvent).count() == 1
        assert s.query(AppReferral).count() == 1

        alice = s.query(AppUser).filter_by(discord_id="u-1").first()
        assert alice.tier == "pro"
        assert alice.referral_count == 2
