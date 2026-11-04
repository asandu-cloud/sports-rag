"""The dashboard's own SQLite file: played matches and their facts, the Most likely log, stage 4's Value picks."""
from __future__ import annotations

import sqlite3

from Scripts.ops.odds_archive import ROOT

DEFAULT_PATH = ROOT / "Index" / "owner_dashboard.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    fixture_api_id TEXT PRIMARY KEY, league TEXT, home TEXT, away TEXT, kickoff TEXT, status TEXT,
    home_goals INTEGER, away_goals INTEGER, corners_home INTEGER, corners_away INTEGER,
    yellow_home INTEGER, yellow_away INTEGER, red_home INTEGER, red_away INTEGER, sot_home INTEGER, sot_away INTEGER,
    read_id INTEGER, read_at TEXT, logged_at TEXT
);
CREATE TABLE IF NOT EXISTS most_likely (
    fixture_api_id TEXT NOT NULL, position INTEGER NOT NULL, market TEXT NOT NULL, side TEXT NOT NULL, line REAL,
    label TEXT NOT NULL, chance REAL NOT NULL, typical_price REAL, version TEXT NOT NULL,
    outcome TEXT, pending_reason TEXT, graded_at TEXT,
    UNIQUE (fixture_api_id, market, side, line)
);
CREATE TABLE IF NOT EXISTS value_picks (
    selection_key TEXT PRIMARY KEY, fixture_api_id TEXT NOT NULL, kickoff TEXT, league TEXT, home TEXT, away TEXT,
    market TEXT, side TEXT, line REAL, label TEXT, pick TEXT, odds REAL, bookmaker TEXT, said REAL,
    outcome TEXT, actual_result REAL, published_at TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def connect(path=DEFAULT_PATH) -> sqlite3.Connection:
    """The dashboard's file, created with its tables if missing."""
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn
