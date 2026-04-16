"""Migration tests for V2 SQLite→Postgres copy commands."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def _seed_predictions_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_id TEXT,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            league TEXT NOT NULL,
            kickoff TEXT,
            market TEXT NOT NULL,
            pick TEXT NOT NULL,
            side TEXT,
            line REAL,
            odds REAL,
            bookmaker TEXT,
            model_prob REAL,
            implied_prob REAL,
            value_edge REAL,
            confidence TEXT,
            projected_total REAL,
            source TEXT DEFAULT 'standalone',
            actual_result REAL,
            outcome TEXT,
            graded_at TEXT,
            created_at TEXT,
            season TEXT,
            prediction_date TEXT,
            closing_odds REAL,
            closing_implied_prob REAL,
            clv REAL
        );
        CREATE TABLE parlay_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_session_id INTEGER,
            discord_user_id INTEGER,
            guild_id INTEGER,
            channel_id INTEGER,
            thread_id INTEGER,
            message_id INTEGER,
            source_command TEXT NOT NULL,
            action_type TEXT NOT NULL,
            request_json TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT
        );
        CREATE TABLE trial_users (
            id INTEGER PRIMARY KEY,
            discord_id TEXT NOT NULL,
            discord_username TEXT,
            activated_at TEXT,
            hits INTEGER DEFAULT 0,
            total_parlays_seen INTEGER DEFAULT 0,
            total_profit REAL DEFAULT 0,
            graduated INTEGER DEFAULT 0,
            graduated_at TEXT
        );
        """
    )
    conn.executemany(
        "INSERT INTO predictions (home_team, away_team, league, market, pick, prediction_date, source, outcome, model_prob, odds) VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            ("Arsenal", "Spurs", "EPL", "goals", "Over 2.5", "2026-04-10", "standalone", "win", 0.55, 1.95),
            ("Arsenal", "Spurs", "EPL", "btts", "BTTS Yes", "2026-04-10", "standalone", "loss", 0.6, 1.85),
            ("Chelsea", "City", "EPL", "moneyline", "City Win", "2026-04-11", "standalone", None, 0.4, 2.5),
        ],
    )
    conn.executemany(
        "INSERT INTO parlay_sessions (parent_session_id, discord_user_id, guild_id, channel_id, thread_id, source_command, action_type, request_json, result_json) VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (None, 1, 2, 3, 4, "parlay", "build", '{"target_date": "2026-04-10"}', '{"combined_odds": 4.2}'),
        ],
    )
    conn.executemany(
        "INSERT INTO trial_users (discord_id, discord_username, activated_at) VALUES (?,?,?)",
        [("11", "alice", "2026-04-01 00:00:00"), ("22", "bob", "2026-04-02 00:00:00")],
    )
    conn.commit()
    conn.close()


def _seed_live_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE live_fixtures (fixture_id INTEGER PRIMARY KEY, league TEXT, home_team TEXT, away_team TEXT, kickoff_utc TEXT, created_at TEXT, finished_at TEXT);
        CREATE TABLE live_state_snapshots (id INTEGER PRIMARY KEY, fixture_id INTEGER, captured_at TEXT, status TEXT, elapsed_min REAL, state_json TEXT);
        CREATE TABLE live_model_snapshots (id INTEGER PRIMARY KEY, fixture_id INTEGER, snapshot_ts TEXT, elapsed_min REAL, score TEXT, snapshot_json TEXT);
        CREATE TABLE live_odds_snapshots (id INTEGER PRIMARY KEY, fixture_id INTEGER, captured_at TEXT, odds_json TEXT);
        CREATE TABLE live_alerts (id INTEGER PRIMARY KEY, fixture_id INTEGER, snapshot_ts TEXT, emitted_at TEXT, alert_type TEXT, severity TEXT, message TEXT, alert_json TEXT);
        """
    )
    conn.execute(
        "INSERT INTO live_fixtures (fixture_id, league, home_team, away_team, kickoff_utc, created_at) VALUES (?,?,?,?,?,?)",
        (123, "EPL", "Arsenal", "Spurs", "2026-04-14T14:00:00+00:00", "2026-04-14T10:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO live_model_snapshots (fixture_id, snapshot_ts, elapsed_min, score, snapshot_json) VALUES (?,?,?,?,?)",
        (123, "2026-04-14T14:15:00+00:00", 15.0, "0-0", '{"model": {"goals": 2.1}}'),
    )
    conn.execute(
        "INSERT INTO live_alerts (fixture_id, snapshot_ts, emitted_at, alert_type, severity, message, alert_json) VALUES (?,?,?,?,?,?,?)",
        (123, "2026-04-14T14:15:00+00:00", "2026-04-14T14:15:05+00:00", "goals_over", "high", "value", '{"type": "goals_over"}'),
    )
    conn.commit()
    conn.close()


def test_migrate_predictions_idempotent(tmp_path, session_factory, settings):
    legacy = tmp_path / "predictions.db"
    _seed_predictions_db(legacy)

    from data_platform.compat import migrate_sqlite_predictions
    counts_1 = migrate_sqlite_predictions(sqlite_path=legacy)
    counts_2 = migrate_sqlite_predictions(sqlite_path=legacy)
    assert counts_1["rows"] == 3
    assert counts_2["rows"] == 3

    from data_platform.models import Prediction
    with session_factory() as s:
        rows = s.query(Prediction).all()
        assert len(rows) == 3
        ars = [r for r in rows if r.home_team == "Arsenal" and r.market == "goals"][0]
        assert ars.outcome == "win"
        assert float(ars.odds) == 1.95


def test_migrate_parlays_and_trial_users(tmp_path, session_factory, settings):
    legacy = tmp_path / "predictions.db"
    _seed_predictions_db(legacy)

    from data_platform.compat import migrate_sqlite_parlay_sessions
    counts = migrate_sqlite_parlay_sessions(sqlite_path=legacy)
    assert counts["parlay_sessions"] == 1
    assert counts["trial_users"] == 2
    from data_platform.models import ParlaySession, TrialUser
    with session_factory() as s:
        ps = s.query(ParlaySession).first()
        assert ps.discord_user_id == 1
        assert ps.request_json["target_date"] == "2026-04-10"
        assert s.query(TrialUser).count() == 2


def test_migrate_live(tmp_path, session_factory, settings):
    legacy = tmp_path / "live.db"
    _seed_live_db(legacy)

    from data_platform.compat import migrate_sqlite_live
    counts = migrate_sqlite_live(sqlite_path=legacy)
    assert counts["fixtures"] == 1
    assert counts["model_snapshots"] == 1
    assert counts["alerts"] == 1

    from data_platform.models import LiveFixture, LiveModelSnapshot, LiveAlert
    with session_factory() as s:
        assert s.query(LiveFixture).count() == 1
        assert s.query(LiveModelSnapshot).first().snapshot_json["model"]["goals"] == 2.1
        assert s.query(LiveAlert).count() == 1
