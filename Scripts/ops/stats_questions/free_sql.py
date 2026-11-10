"""Free SQL, owner only: one read-only query over the stats file, with the SQL shown beside its rows.

Three locks: the file is opened read-only with ``query_only``, an authorizer allows reading and nothing else (no
ATTACH, no PRAGMA, no writes), and a progress handler stops the query after ``seconds``.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from Scripts.ops.stats_questions import store

SECONDS, MAX_ROWS = 2.0, 500
ALLOWED = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}


def run(path: Path, sql: str, *, seconds: float = SECONDS, max_rows: int = MAX_ROWS) -> dict:
    """{"sql", "columns", "rows", "truncated", "ms"}, or {"sql", "error": {"kind", "message"}} with kind refused,
    timeout or sql."""
    sql = (sql or "").strip()
    refused, deadline = [], None

    def authorize(action, *args):
        if action in ALLOWED:
            return sqlite3.SQLITE_OK
        refused.append(action)
        return sqlite3.SQLITE_DENY

    def stop():
        return int(deadline is not None and time.monotonic() > deadline)

    conn = store.connect(path)
    started = time.monotonic()
    try:
        conn.set_authorizer(authorize)
        conn.set_progress_handler(stop, 1000)
        deadline = started + seconds
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        columns = [c[0] for c in cursor.description or ()]
    except sqlite3.Error as exc:
        if refused:
            kind, message = "refused", "Only reading is allowed: SELECT over the stats file's tables."
        elif deadline is not None and time.monotonic() > deadline:
            kind, message = "timeout", f"Stopped after {seconds:g} seconds."
        else:
            kind, message = "sql", str(exc)
        return {"sql": sql, "error": {"kind": kind, "message": message}}
    finally:
        conn.close()
    return {"sql": sql, "columns": columns, "rows": [list(r) for r in rows[:max_rows]],
            "truncated": len(rows) > max_rows, "ms": round((time.monotonic() - started) * 1000)}
