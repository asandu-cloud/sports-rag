"""Prospective pre-match odds collection, isolated from predictions and settlement.

No arguments prints a read-only plan. ``--execute`` appends API observations to
Index/odds_archive/odds.sqlite3. ``--status`` reads collection metadata only.
This archive is not an approved training set, executable-price certificate or
closing-price feed. Original values, missingness and source times are retained.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import shutil
import time
import uuid

import requests

from Scripts.rag_ingest.odds_provider import LEAGUE_TO_API_ID, api_football_season_for_date

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "spix-prospective-odds.v1"
URL = "https://v3.football.api-sports.io/odds"
INTERVAL = 3 * 60 * 60
SAFE_HEADERS = ("date", "content-type", "x-ratelimit-requests-limit",
                "x-ratelimit-requests-remaining", "x-ratelimit-limit", "x-ratelimit-remaining")
CONTRACT = {
    "schema": SCHEMA, "purpose": "prospective_quote_observation_only",
    "evaluation_assignment": "unassigned_existing_reserves_remain_protected",
    "outcomes_collected": False, "training_authorized": False,
    "execution_and_settlement_equivalence": "unverified",
}


def now():
    return datetime.now(timezone.utc)


def stamp(value):
    if value.tzinfo is None:
        raise ValueError("Timezone-aware time required")
    return value.astimezone(timezone.utc).isoformat()


def parse_time(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result.astimezone(timezone.utc) if result.tzinfo else None
    except (ValueError, TypeError, AttributeError):
        return None


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def archive_path(root):
    """Never alias an existing platform DB or an external path."""
    root = Path(root).resolve()
    directory = root / "Index/odds_archive"
    for path in (root / "Index", directory, directory / "odds.sqlite3",
                 directory / ".collector.lock", directory / "odds.sqlite3-journal"):
        if path.is_symlink() or (path.is_file() and path.stat().st_nlink != 1):
            raise ValueError("Archive paths must not alias other files")
    return directory / "odds.sqlite3"


def initialize(db):
    existing = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if existing and "archive_metadata" not in existing:
        raise ValueError("Refusing to modify an unrecognized database")
    if existing:
        row = db.execute("SELECT value FROM archive_metadata WHERE key='contract'").fetchone()
        if not row or json.loads(row[0]) != CONTRACT:
            raise ValueError("Archive contract mismatch")
        initialize_compact_index(db)
        return
    db.executescript("""
        CREATE TABLE archive_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE runs(id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT,
            status TEXT NOT NULL, config_json TEXT NOT NULL, source_sha256 TEXT NOT NULL);
        CREATE TABLE scopes(run_id TEXT NOT NULL REFERENCES runs(id), league_id INTEGER NOT NULL,
            fixture_date TEXT NOT NULL, status TEXT NOT NULL, reason TEXT,
            PRIMARY KEY(run_id, league_id, fixture_date));
        CREATE TABLE raw_bodies(sha256 TEXT PRIMARY KEY, body_gzip BLOB NOT NULL);
        CREATE TABLE requests(id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
            requested_at TEXT NOT NULL, received_at TEXT, params_json TEXT NOT NULL,
            http_status INTEGER, headers_json TEXT, raw_sha256 TEXT REFERENCES raw_bodies(sha256),
            status TEXT NOT NULL, error_code TEXT, page_total INTEGER);
        CREATE INDEX request_time ON requests(requested_at);
        CREATE TABLE quotes(id INTEGER PRIMARY KEY, request_id TEXT NOT NULL REFERENCES requests(id),
            ordinal TEXT NOT NULL, fixture_id INTEGER, league_id INTEGER, season INTEGER,
            kickoff_utc TEXT, source_updated_at TEXT, source_age_seconds REAL,
            bookmaker_id INTEGER, bookmaker_name TEXT, market_id INTEGER, market_name TEXT,
            value_json TEXT NOT NULL, odd_json TEXT NOT NULL, decimal_odds TEXT,
            issues_json TEXT NOT NULL, UNIQUE(request_id, ordinal));
        CREATE INDEX fixture_quote_time ON quotes(fixture_id, source_updated_at);
        CREATE INDEX quote_market ON quotes(league_id, market_id, bookmaker_id);
    """)
    db.execute("INSERT INTO archive_metadata VALUES('contract', ?)", (encoded(CONTRACT),))
    db.execute("INSERT INTO archive_metadata VALUES('created_at', ?)", (encoded(stamp(now())),))
    # Run/request status is operational state. Evidence already received cannot
    # be edited or deleted through the normal connection.
    for table in ("quotes", "raw_bodies", "archive_metadata"):
        for action in ("UPDATE", "DELETE"):
            db.execute(f"CREATE TRIGGER {table}_{action} BEFORE {action} ON {table} "
                       "BEGIN SELECT RAISE(ABORT, 'append-only archive'); END")
    for action in ("UPDATE", "DELETE"):
        db.execute(f"CREATE TRIGGER requests_{action} BEFORE {action} ON requests "
                   "WHEN OLD.status != 'pending' BEGIN SELECT RAISE(ABORT, 'sealed receipt'); END")
    db.commit()
    initialize_compact_index(db)


def initialize_compact_index(db):
    """Additive layout: preserve the first cycle's expanded index verbatim.

    Full quotes remain in raw_bodies. A fixture index avoids repeating strings
    hundreds of thousands of times per cycle; quote_rows provides lossless
    on-demand reconstruction using the original observation time.
    """
    db.execute("""CREATE TABLE IF NOT EXISTS fixture_snapshots(
        request_id TEXT NOT NULL REFERENCES requests(id), ordinal INTEGER NOT NULL,
        fixture_id INTEGER, league_id INTEGER, season INTEGER, kickoff_utc TEXT,
        source_updated_at TEXT, source_age_seconds REAL, quote_count INTEGER NOT NULL,
        flagged_quote_count INTEGER NOT NULL, PRIMARY KEY(request_id, ordinal))""")
    db.execute("CREATE INDEX IF NOT EXISTS fixture_snapshot_time ON fixture_snapshots(fixture_id,source_updated_at)")
    for action in ("UPDATE", "DELETE"):
        db.execute(f"CREATE TRIGGER IF NOT EXISTS fixture_snapshots_{action} BEFORE {action} ON fixture_snapshots "
                   "BEGIN SELECT RAISE(ABORT, 'append-only archive'); END")
    db.execute("INSERT OR IGNORE INTO archive_metadata VALUES('storage_layout', ?)",
               (encoded("compressed_receipts_fixture_index.v1_with_preserved_legacy_quotes"),))
    db.commit()


def price(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = Decimal(str(value))
        return str(result) if result.is_finite() and result > 1 else None
    except InvalidOperation:
        return None


def identity(value):
    return value if type(value) is int and value > 0 else None


def normalize(payload, params, received_at):
    """Return index rows; keep source labels untouched, never guess market rules."""
    if not isinstance(payload, dict) or payload.get("get") not in ("odds", "/odds"):
        raise ValueError("wrong_envelope")
    if payload.get("errors"):
        raise ValueError("provider_error")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict) or any(str(parameters.get(k)) != str(v) for k, v in params.items()):
        raise ValueError("parameter_mismatch")
    rows = payload.get("response")
    if not isinstance(rows, list) or type(payload.get("results")) is not int or payload["results"] != len(rows):
        raise ValueError("invalid_result_count")
    paging = payload.get("paging") or {}
    current, total = paging.get("current"), paging.get("total")
    if type(current) is not int or type(total) is not int or current != params["page"] or not current <= total <= 1000:
        raise ValueError("invalid_pagination")
    if not rows and total > 1:
        raise ValueError("empty_paginated_page")
    quotes = []
    for i, event in enumerate(rows):
        fixture, league = event.get("fixture") or {}, event.get("league") or {}
        kickoff = parse_time(fixture.get("date"))
        updated = parse_time(event.get("update"))
        issues = []
        if not identity(fixture.get("id")):
            issues.append("missing_fixture_identity")
        if league.get("id") != params["league"] or league.get("season") != params["season"]:
            raise ValueError("league_or_season_mismatch")
        if kickoff is None:
            issues.append("missing_kickoff")
        elif kickoff.date().isoformat() != params["date"]:
            raise ValueError("fixture_date_mismatch")
        elif kickoff <= received_at:
            issues.append("observed_at_or_after_kickoff")
        if updated is None:
            issues.append("missing_source_update")
        elif updated > received_at:
            issues.append("future_source_update")
        if kickoff and updated and updated >= kickoff:
            issues.append("source_update_at_or_after_kickoff")
        books = event.get("bookmakers")
        if not isinstance(books, list):
            raise ValueError("invalid_bookmakers")
        for b, book in enumerate(books):
            for m, market in enumerate(book["bets"]):
                if any(name is not None and not isinstance(name, str) for name in (book.get("name"), market.get("name"))):
                    raise ValueError("invalid_market_name")
                for v, value in enumerate(market["values"]):
                    row_issues = list(issues)
                    if not identity(book.get("id")) or not identity(market.get("id")):
                        row_issues.append("missing_bookmaker_or_market_identity")
                    decimal = price(value.get("odd"))
                    if decimal is None:
                        row_issues.append("missing_or_invalid_price")
                    if value.get("value") is None:
                        row_issues.append("missing_selection_label")
                    quotes.append((f"{i}/{b}/{m}/{v}", identity(fixture.get("id")), league["id"], league["season"],
                        stamp(kickoff) if kickoff else None, stamp(updated) if updated else None,
                        (received_at - updated).total_seconds() if updated else None,
                        identity(book.get("id")), book.get("name"), identity(market.get("id")), market.get("name"),
                        encoded(value.get("value")), encoded(value.get("odd")), decimal, encoded(row_issues)))
    return total, quotes


def receive(db, request_id, response, params, received_at):
    body = response.content  # Original HTTP entity bytes, after transport decompression.
    digest = hashlib.sha256(body).hexdigest()
    headers = {k: response.headers[k] for k in SAFE_HEADERS if k in response.headers}
    status, error, total, quotes = "ok", None, None, []
    if response.status_code != 200:
        status, error = "failed", f"http_{response.status_code}"
    else:
        try:
            payload = json.loads(body, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non_finite_json")))
            total, quotes = normalize(payload, params, received_at)
        except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
            status, error = "failed", "invalid_or_error_envelope"
    with db:
        db.execute("INSERT OR IGNORE INTO raw_bodies VALUES(?, ?)", (digest, gzip.compress(body, mtime=0)))
        db.execute("""UPDATE requests SET received_at=?, http_status=?, headers_json=?,
            raw_sha256=?, status=?, error_code=?, page_total=? WHERE id=?""",
            (stamp(received_at), response.status_code, encoded(headers), digest, status, error, total, request_id))
        if status == "ok":
            counts = {}
            for row in quotes:
                ordinal = int(row[0].split("/")[0])
                count, flagged = counts.get(ordinal, (0, 0))
                counts[ordinal] = (count + 1, flagged + int(row[-1] != "[]"))
            fixtures = []
            for ordinal, event in enumerate(payload["response"]):
                fixture = event.get("fixture") or {}
                kickoff = parse_time(fixture.get("date"))
                updated = parse_time(event.get("update"))
                fixtures.append((request_id, ordinal, identity(fixture.get("id")),
                    event["league"]["id"], event["league"]["season"], stamp(kickoff) if kickoff else None,
                    stamp(updated) if updated else None, (received_at - updated).total_seconds() if updated else None,
                    *counts.get(ordinal, (0, 0))))
            db.executemany("INSERT INTO fixture_snapshots VALUES(?,?,?,?,?,?,?,?,?,?)", fixtures)
    return status, total, headers


def quote_rows(db, request_id):
    """Read exact archived selections without networking or current-time substitution."""
    row = db.execute("""SELECT r.params_json,r.received_at,b.sha256,b.body_gzip
        FROM requests r JOIN raw_bodies b ON b.sha256=r.raw_sha256
        WHERE r.id=? AND r.status='ok'""", (request_id,)).fetchone()
    if row is None:
        raise ValueError("No valid archived receipt")
    raw = gzip.decompress(row[3])
    if hashlib.sha256(raw).hexdigest() != row[2]:
        raise ValueError("Archive checksum mismatch")
    return normalize(json.loads(raw), json.loads(row[0]), parse_time(row[1]))[1]


def config(leagues=None, days_ahead=14, max_requests=600, daily_budget=3600, reserve=1000, rpm=60):
    leagues = list(leagues or LEAGUE_TO_API_ID)
    if not leagues or len(set(leagues)) != len(leagues) or set(leagues) - LEAGUE_TO_API_ID.keys():
        raise ValueError("Invalid leagues")
    if not 0 <= days_ahead <= 14 or not 1 <= max_requests <= 2000 or not 1 <= daily_budget <= 10000 or reserve < 0 or not 1 <= rpm <= 120:
        raise ValueError("Invalid collection limits")
    return dict(leagues=leagues, days_ahead=days_ahead, max_requests=max_requests,
                daily_budget=daily_budget, reserve=reserve, rpm=rpm)


def collect(db, key, settings, *, session=None, clock=now, sleep=time.sleep, monotonic=time.monotonic):
    transport = session or requests.Session()
    run_id, started = uuid.uuid4().hex, clock()
    scopes = [(LEAGUE_TO_API_ID[league], (started + timedelta(days=day)).date())
              for day in range(settings["days_ahead"] + 1) for league in settings["leagues"]]
    # Schedule every requested scope before networking, so failures/omissions
    # have a denominator. Outcomes and the live platform DB are never opened.
    with db:
        db.execute("UPDATE runs SET status='interrupted', finished_at=? WHERE status='running'", (stamp(started),))
        db.execute("UPDATE scopes SET status='incomplete', reason='interrupted' WHERE status IN ('pending','running')")
        db.execute("UPDATE requests SET status='failed', error_code='interrupted' WHERE status='pending'")
        db.execute("INSERT INTO runs VALUES(?,?,NULL,'running',?,?)", (run_id, stamp(started), encoded(settings),
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        db.executemany("INSERT INTO scopes VALUES(?,?,?,'pending',NULL)",
                       [(run_id, league, day.isoformat()) for league, day in scopes])
    used, remaining, stop, last_start = 0, None, None, None
    last = db.execute("SELECT headers_json FROM requests WHERE requested_at>=? AND headers_json IS NOT NULL ORDER BY requested_at DESC LIMIT 1",
                      (stamp(started.replace(hour=0, minute=0, second=0, microsecond=0)),)).fetchone()
    if last:
        raw = json.loads(last[0]).get("x-ratelimit-requests-remaining")
        remaining = int(raw) if str(raw).isdigit() else None
    budget_day = started.date()
    try:
        for league, day in scopes:
            if stop:
                break
            scope_key = (run_id, league, day.isoformat())
            page, total, seen_fixtures = 1, None, set()
            with db:
                db.execute("UPDATE scopes SET status='running' WHERE run_id=? AND league_id=? AND fixture_date=?", scope_key)
            while True:
                if last_start is not None:
                    sleep(max(0, 60 / settings["rpm"] - (monotonic() - last_start)))
                current = clock()
                if current.date() != budget_day:
                    # End instead of assigning yesterday's quota to a new day.
                    stop = "utc_day_changed"
                midnight = stamp(current.replace(hour=0, minute=0, second=0, microsecond=0))
                daily = db.execute("SELECT count(*) FROM requests WHERE requested_at>=?", (midnight,)).fetchone()[0]
                if used >= settings["max_requests"] or daily >= settings["daily_budget"]:
                    stop = "local_request_budget"
                if remaining is not None and remaining <= settings["reserve"]:
                    stop = "provider_quota_reserve"
                if stop:
                    break
                database_path = db.execute("PRAGMA database_list").fetchone()[2]
                if database_path and shutil.disk_usage(Path(database_path).parent).free < 2 * 1024 ** 3:
                    stop = "low_disk_space"
                    break
                params = dict(league=league, season=api_football_season_for_date(day), date=day.isoformat(), page=page)
                request_id = uuid.uuid4().hex
                with db:
                    db.execute("INSERT INTO requests(id,run_id,requested_at,params_json,status) VALUES(?,?,?,?,'pending')",
                               (request_id, run_id, stamp(current), encoded(params)))
                used += 1
                last_start = monotonic()
                response = None
                try:
                    response = transport.get(URL, params=params, headers={"x-apisports-key": key},
                                             timeout=(5, 30), allow_redirects=False)
                    state, pages, headers = receive(db, request_id, response, params, clock())
                except requests.RequestException:
                    with db:
                        db.execute("UPDATE requests SET received_at=?, status='failed', error_code='transport_error' WHERE id=?",
                                   (stamp(clock()), request_id))
                    state, pages, headers = "failed", None, {}
                finally:
                    if response is not None:
                        response.close()
                raw = headers.get("x-ratelimit-requests-remaining")
                remaining = int(raw) if str(raw).isdigit() else (remaining - 1 if remaining is not None else None)
                if response is not None and response.status_code in (401, 403, 429):
                    stop = "provider_auth_or_rate_limit"
                # A 200 error envelope may also signal account exhaustion. Stop
                # this cycle; do not hammer every other scope with the same error.
                if state != "ok":
                    stop = stop or "request_failed"
                    break
                if str(headers.get("x-ratelimit-remaining")) == "0":
                    stop = "provider_minute_quota"
                if total is not None and pages != total:
                    with db:
                        db.execute("UPDATE scopes SET status='incomplete', reason='pagination_changed' WHERE run_id=? AND league_id=? AND fixture_date=?", scope_key)
                    break
                fixture_ids = [r[0] for r in db.execute(
                    "SELECT DISTINCT fixture_id FROM fixture_snapshots WHERE request_id=? AND fixture_id IS NOT NULL", (request_id,))]
                if seen_fixtures.intersection(fixture_ids):
                    with db:
                        db.execute("UPDATE scopes SET status='incomplete', reason='duplicate_fixture_across_pages' WHERE run_id=? AND league_id=? AND fixture_date=?", scope_key)
                    break
                seen_fixtures.update(fixture_ids)
                total = pages
                if page == total:
                    with db:
                        db.execute("UPDATE scopes SET status='complete' WHERE run_id=? AND league_id=? AND fixture_date=?", scope_key)
                    break
                page += 1
                if stop:
                    break
        with db:
            db.execute("UPDATE scopes SET status='incomplete', reason=? WHERE run_id=? AND status IN ('running','pending')",
                       (stop or "not_completed", run_id))
            incomplete = db.execute("SELECT count(*) FROM scopes WHERE run_id=? AND status!='complete'", (run_id,)).fetchone()[0]
            state = "incomplete" if incomplete else "complete"
            db.execute("UPDATE runs SET status=?, finished_at=? WHERE id=?", (state, stamp(clock()), run_id))
        return dict(run_id=run_id, status=state, requests=used, scopes=len(scopes),
                    incomplete_scopes=incomplete, stop_reason=stop)
    finally:
        if session is None:
            transport.close()


def status(path):
    if not path.exists():
        return {"initialized": False, "database": str(path)}
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        db.row_factory = sqlite3.Row
        compact = db.execute("SELECT 1 FROM sqlite_master WHERE name='fixture_snapshots'").fetchone() is not None
        quote_count = """SELECT
            (SELECT count(*) FROM quotes q WHERE NOT EXISTS(SELECT 1 FROM fixture_snapshots f WHERE f.request_id=q.request_id))
            + coalesce((SELECT sum(quote_count) FROM fixture_snapshots),0)""" if compact else "SELECT count(*) FROM quotes"
        fixture_count = "SELECT count(DISTINCT fixture_id) FROM (SELECT fixture_id FROM quotes UNION SELECT fixture_id FROM fixture_snapshots)" if compact else "SELECT count(DISTINCT fixture_id) FROM quotes"
        return dict(initialized=True, database=str(path),
            contract=json.loads(db.execute("SELECT value FROM archive_metadata WHERE key='contract'").fetchone()[0]),
            requests=db.execute("SELECT count(*) FROM requests").fetchone()[0],
            quotes=db.execute(quote_count).fetchone()[0],
            fixtures=db.execute(fixture_count).fetchone()[0],
            last_run=dict(row) if (row := db.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()) else None)


def run(root=ROOT, *, execute=False, show_status=False, settings=None):
    path = archive_path(root)
    settings = settings or config()
    if show_status:
        return status(path)
    if not execute:
        return dict(action="plan", database=str(path), settings=settings, contract=CONTRACT,
                    minimum_requests_per_cycle=len(settings["leagues"]) * (settings["days_ahead"] + 1),
                    interval_seconds=INTERVAL)
    from dotenv import load_dotenv
    load_dotenv(Path(root) / ".env", override=False)
    key = os.getenv("API_FOOTBALL_KEY") or os.getenv("API-FOOTBALL-KEY")
    if not key:
        raise ValueError("API-Football key is not configured")
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / ".collector.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "already_running"}
        with closing(sqlite3.connect(path, timeout=10)) as db:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA synchronous=FULL")
            initialize(db)
            return collect(db, key, settings)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--status", action="store_true")
    parser.add_argument("--leagues", nargs="+", choices=tuple(LEAGUE_TO_API_ID))
    parser.add_argument("--days-ahead", type=int, default=14)
    parser.add_argument("--max-requests", type=int, default=600)
    parser.add_argument("--daily-budget", type=int, default=3600)
    parser.add_argument("--reserve", type=int, default=1000)
    parser.add_argument("--rpm", type=int, default=60)
    args = parser.parse_args(argv)
    try:
        settings = config(args.leagues, args.days_ahead, args.max_requests, args.daily_budget, args.reserve, args.rpm)
        result = run(execute=args.execute, show_status=args.status, settings=settings)
        print(json.dumps(result, indent=2))
        return 2 if result.get("status") == "incomplete" else 0
    except (ValueError, OSError, sqlite3.Error):
        # Do not emit credentials, provider error bodies or transport objects.
        print(json.dumps({"status": "failed", "reason": "configuration_or_storage_error"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
