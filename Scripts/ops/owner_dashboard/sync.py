"""Update the dashboard's file from platform.db: read only (``mode=ro``), inside the data gate, short statements.

A match is logged once it is over, from its last read evaluated before its own kickoff (either stage; an
``unavailable`` read carries no prices and is passed over), so a postponed match logs the read for its new date.
Read JSON (~0.7 MB each) is fetched ``BATCH`` at a time, each batch under its own gate, so the refresh can take
the gate between batches; a busy gate ends the update and keeps what it wrote.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from typing import Dict, Iterable, List

from Scripts.data_platform import recommendations
from Scripts.data_platform.calibration_data import _time
from Scripts.data_platform.services.refresh_coordination import RefreshBusy, data_access

FINISHED = ("FT", "AET", "PEN")
SCAN = 500   # match_reads metadata rows per statement
BATCH = 20   # read JSON per gate hold
CHUNK = 200  # ids per IN (...) statement


def _platform(path):
    return closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10))


def _iso(value):
    moment = _time(value)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ") if moment else None


def _chunks(items: Iterable, size: int = CHUNK):
    items = list(items)
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _marks(size: int) -> str:
    return ",".join("?" * size)


def _candidates(db, done) -> Dict[str, dict]:
    """Each finished match not yet logged: its last usable read before its kickoff."""
    reads, last = {}, 0
    while chunk := db.execute("SELECT rowid, fixture_api_id, evaluated_at, status, league, home_team, away_team "
                              "FROM match_reads WHERE rowid > ? ORDER BY rowid LIMIT ?", (last, SCAN)).fetchall():
        last = chunk[-1][0]
        for row in chunk:
            if str(row[1]) not in done and row[3] != "unavailable" and str(row[1]).isdigit():
                reads.setdefault(str(row[1]), []).append(row)
    best = {}
    for ids in _chunks(reads):
        for api_id, kickoff, status in db.execute(
                f"SELECT api_football_id, kickoff_utc, status FROM fixtures WHERE api_football_id IN ({_marks(len(ids))}) "
                f"AND status IN ({_marks(len(FINISHED))})", [int(i) for i in ids] + list(FINISHED)):
            cut = _time(kickoff)
            usable = [r for r in reads[str(api_id)] if cut and _time(r[2]) and _time(r[2]) < cut]
            if usable:
                row = max(usable, key=lambda r: (_time(r[2]), r[0]))
                best[str(api_id)] = {"rowid": row[0], "read_at": _iso(row[2]), "league": row[4], "home": row[5],
                                     "away": row[6], "kickoff": _iso(kickoff), "status": status}
    return best


def _read_json(db, ids: List[int]) -> Dict[int, dict]:
    return {rowid: json.loads(raw) if isinstance(raw, str) else raw for rowid, raw in db.execute(
        f"SELECT rowid, read_json FROM match_reads WHERE rowid IN ({_marks(len(ids))})", ids)}


def _store_read(conn, fixture_api_id: str, match: dict, read: dict, now: datetime) -> int:
    picks = recommendations.block(read.get("canonical_results") or [], selections=[], metrics={},
                                  prices_as_of=match["read_at"])["most_likely"]
    with conn:
        conn.execute(
            "INSERT INTO matches (fixture_api_id, league, home, away, kickoff, status, read_id, read_at, logged_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (fixture_api_id) DO UPDATE SET league = excluded.league, "
            "home = excluded.home, away = excluded.away, kickoff = excluded.kickoff, status = excluded.status, "
            "read_id = excluded.read_id, read_at = excluded.read_at, logged_at = excluded.logged_at",
            (fixture_api_id, match["league"], match["home"], match["away"], match["kickoff"], match["status"],
             match["rowid"], match["read_at"], _iso(now.isoformat())))
        conn.executemany(
            "INSERT OR IGNORE INTO most_likely (fixture_api_id, position, market, side, line, label, chance, "
            "typical_price, version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(fixture_api_id, position, p["market"], p["side"], p.get("line"), p["label"], p["chance"],
              p.get("typical_price"), recommendations.VERSION) for position, p in enumerate(picks)])
    return len(picks)


def log_most_likely(conn, platform_path, *, now: datetime, gate=data_access) -> dict:
    """Log the Most likely of every finished match not logged yet; ``skipped_busy`` when the refresh held the gate."""
    out = {"logged": 0, "picks": 0, "skipped_busy": False}
    done = {row[0] for row in conn.execute("SELECT fixture_api_id FROM matches WHERE read_id IS NOT NULL")}
    try:
        with gate(), _platform(platform_path) as db:
            todo = _candidates(db, done)
        for batch in _chunks(sorted(todo, key=lambda f: (todo[f]["kickoff"], f)), BATCH):
            with gate(), _platform(platform_path) as db:
                reads = _read_json(db, [todo[f]["rowid"] for f in batch])
            for fixture_api_id in batch:
                match = todo[fixture_api_id]
                out["picks"] += _store_read(conn, fixture_api_id, match, reads.get(match["rowid"]) or {}, now)
                out["logged"] += 1
    except RefreshBusy:
        out["skipped_busy"] = True
    return out
