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


FACT_COLUMNS = ("status", "home_goals", "away_goals", "corners_home", "corners_away", "yellow_home", "yellow_away",
                "red_home", "red_away", "sot_home", "sot_away")


def _facts(db, ids: List[str]) -> Dict[str, dict]:
    """Score, status and both teams' corners, cards and shots on target per match (API-Football id)."""
    found, internal = {}, {}
    for api_id, fixture_id, kickoff, status, home_goals, away_goals in db.execute(
            f"SELECT api_football_id, id, kickoff_utc, status, home_goals, away_goals FROM fixtures "
            f"WHERE api_football_id IN ({_marks(len(ids))})", [int(i) for i in ids]):
        found[str(api_id)] = {"kickoff": _iso(kickoff), "status": status, "home_goals": home_goals,
                              "away_goals": away_goals}
        internal[fixture_id] = str(api_id)
    if internal:
        for fixture_id, is_home, corners, shots_on, yellow, red in db.execute(
                f"SELECT fixture_id, is_home, corners, shots_on, yellow_cards, red_cards FROM fixture_team_stats "
                f"WHERE fixture_id IN ({_marks(len(internal))})", list(internal)):
            end = "home" if is_home else "away"
            found[internal[fixture_id]].update({f"corners_{end}": corners, f"sot_{end}": shots_on,
                                                f"yellow_{end}": yellow, f"red_{end}": red})
    return found


def _store_facts(conn, fixture_api_id: str, facts: dict) -> None:
    values = [facts.get(column) for column in FACT_COLUMNS]
    with conn:
        if conn.execute("SELECT 1 FROM matches WHERE fixture_api_id = ?", (fixture_api_id,)).fetchone():
            conn.execute(f"UPDATE matches SET {', '.join(f'{c} = ?' for c in FACT_COLUMNS)} WHERE fixture_api_id = ?",
                         values + [fixture_api_id])
            return
        pick = conn.execute("SELECT league, home, away, kickoff FROM value_picks WHERE fixture_api_id = ? LIMIT 1",
                            (fixture_api_id,)).fetchone()
        conn.execute(f"INSERT INTO matches (fixture_api_id, league, home, away, kickoff, {', '.join(FACT_COLUMNS)}) "
                     f"VALUES (?, ?, ?, ?, ?, {_marks(len(FACT_COLUMNS))})",
                     [fixture_api_id, pick["league"] if pick else None, pick["home"] if pick else None,
                      pick["away"] if pick else None, facts.get("kickoff") or (pick["kickoff"] if pick else None)]
                     + values)


def _outcome(pick, match):
    """("hit" | "miss", None) for a graded Most likely pick, else (None, why it waits)."""
    if match["status"] in ("AET", "PEN"):
        return None, "extra time"  # the stored score includes extra time; these markets settle on 90 minutes
    if match["status"] != "FT":
        return None, "not final"
    if pick["market"] == "corners":
        if match["corners_home"] is None or match["corners_away"] is None:
            return None, "corners not in yet"
        total = match["corners_home"] + match["corners_away"]
        won = total > pick["line"] if pick["side"] == "over" else total < pick["line"]
    elif match["home_goals"] is None or match["away_goals"] is None:
        return None, "score not in yet"
    else:
        won = recommendations.wins({"market": pick["market"], "side": pick["side"], "line": pick["line"]},
                                   match["home_goals"], match["away_goals"])
    return ("hit" if won else "miss"), None


def grade(conn, platform_path, *, now: datetime, gate=data_access) -> dict:
    """Refresh the facts of every match with an ungraded Most likely pick or a Value pick still settling, then
    grade the Most likely picks."""
    out = {"graded": 0, "pending": 0, "skipped_busy": False}
    ids = {row[0] for row in conn.execute("SELECT DISTINCT fixture_api_id FROM most_likely WHERE outcome IS NULL")}
    ids |= {row[0] for row in conn.execute(
        "SELECT DISTINCT v.fixture_api_id FROM value_picks v LEFT JOIN matches m USING (fixture_api_id) "
        "WHERE m.fixture_api_id IS NULL OR m.status IS NULL OR m.status NOT IN ('FT', 'AET', 'PEN') "
        "OR m.corners_home IS NULL OR m.corners_away IS NULL OR v.outcome IS NULL")}
    try:
        for chunk in _chunks(sorted(i for i in ids if str(i).isdigit())):
            with gate(), _platform(platform_path) as db:
                facts = _facts(db, chunk)
            for fixture_api_id, values in facts.items():
                _store_facts(conn, fixture_api_id, values)
    except RefreshBusy:
        out["skipped_busy"] = True
    graded_at = _iso(now.isoformat())
    with conn:
        for pick in conn.execute("SELECT p.rowid, p.*, m.status, m.home_goals, m.away_goals, m.corners_home, "
                                 "m.corners_away FROM most_likely p JOIN matches m USING (fixture_api_id) "
                                 "WHERE p.outcome IS NULL").fetchall():
            outcome, reason = _outcome(pick, pick)
            conn.execute("UPDATE most_likely SET outcome = ?, pending_reason = ?, graded_at = ? WHERE rowid = ?",
                         (outcome, reason, graded_at if outcome else None, pick["rowid"]))
            out["graded" if outcome else "pending"] += 1
    return out


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
