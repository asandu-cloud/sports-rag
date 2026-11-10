"""The question log: every question, what was chosen for it, the answer, the time, the cost, and the owner's verdict.

Its own small file beside the stats file, never platform.db. The trial's "9 in 10 understood" is read from here.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "Index" / "stats_questions" / "log.db"  # beside stats.db
MODEL_MODES = ("ask", "sql_ask")  # the modes that call the model, and so count towards the daily ceiling
VERDICTS = ("right", "wrong")
JSON_FIELDS = ("calls", "answers", "usage")

SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
  id INTEGER PRIMARY KEY, asked_at TEXT NOT NULL, mode TEXT NOT NULL, question TEXT, calls TEXT, answers TEXT,
  model TEXT, ms INTEGER, usage TEXT, cost_usd REAL, error TEXT, flagged_bet INTEGER, verdict TEXT, note TEXT);
CREATE INDEX IF NOT EXISTS questions_day ON questions (asked_at);
"""
# Added for the website's members (D94): who asked, on which plan, and from where ("website"; empty on this Mac).
MEMBER_COLUMNS = {"user_id": "INTEGER", "plan": "TEXT", "source": "TEXT"}


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(path: Path = DEFAULT_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    have = {row[1] for row in conn.execute("PRAGMA table_info(questions)")}
    with conn:
        for name, kind in MEMBER_COLUMNS.items():
            if name not in have:
                conn.execute(f"ALTER TABLE questions ADD COLUMN {name} {kind}")
    return conn


def record(conn, *, mode, question=None, calls=None, answers=None, model=None, ms=None, usage=None, cost_usd=None,
           error=None, flagged_bet=None, user_id=None, plan=None, source=None, now=None) -> int:
    row = {"asked_at": _iso(now or datetime.now(timezone.utc)), "mode": mode, "question": question,
           "calls": calls, "answers": answers, "model": model, "ms": ms, "usage": usage, "cost_usd": cost_usd,
           "error": error, "flagged_bet": None if flagged_bet is None else int(flagged_bet), "user_id": user_id,
           "plan": plan, "source": source}
    for field in JSON_FIELDS:
        row[field] = None if row[field] is None else json.dumps(row[field], ensure_ascii=False)
    with conn:
        cursor = conn.execute(f"INSERT INTO questions ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                              tuple(row.values()))
    return cursor.lastrowid


def set_verdict(conn, entry_id: int, verdict: str, note: str | None = None) -> bool:
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}")
    with conn:
        changed = conn.execute("UPDATE questions SET verdict = ?, note = ? WHERE id = ?",
                               (verdict, (note or "").strip()[:500] or None, entry_id)).rowcount
    return bool(changed)


def recent(conn, limit: int = 50) -> list[dict]:
    out = []
    for row in conn.execute("SELECT * FROM questions ORDER BY id DESC LIMIT ?", (limit,)):
        entry = dict(row)
        for field in JSON_FIELDS:
            entry[field] = None if entry[field] is None else json.loads(entry[field])
        out.append(entry)
    return out


def asked_today(conn, *, user_id=None, now=None) -> int:
    """Model questions since midnight UTC: every one, for the site's daily ceiling, or one member's answered ones,
    for their daily limit (a failed question does not use it up)."""
    day = _iso(now or datetime.now(timezone.utc))[:10]
    sql = (f"SELECT count(*) FROM questions WHERE asked_at >= ? AND mode IN ({', '.join('?' * len(MODEL_MODES))})")
    args = [day, *MODEL_MODES]
    if user_id is not None:
        sql += " AND user_id = ? AND error IS NULL"
        args.append(user_id)
    return conn.execute(sql, args).fetchone()[0]
