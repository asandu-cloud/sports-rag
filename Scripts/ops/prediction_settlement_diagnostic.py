"""Capped provider diagnostic. SQLite is opened read-only; outcomes cannot be written.

Response bodies and an attempt ledger are saved in a new private evidence folder.
Contract probes are synthetic selections on real results, NOT recovered bets.
"""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
from pathlib import Path
import sqlite3
import time
from urllib.parse import quote

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from Scripts.data_platform import config
from Scripts.data_platform.repositories.predictions import PredictionRepository
from Scripts.data_platform.services.measurement_runtime import BoundedFootballClient, BudgetExceeded
from Scripts.data_platform.settlement import grade_selection, parse_fixture_result, utc_datetime


def fingerprint(rows):
    return sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()


class EvidenceBudget:
    def __init__(self, directory):
        self.directory, self.attempts = directory, []

    def reserve_request(self, *, day, limit):
        if len(self.attempts) >= limit:
            raise BudgetExceeded("Diagnostic request cap reached")
        self.attempts.append({"number": len(self.attempts) + 1,
                              "reserved_at": datetime.now(timezone.utc).isoformat()})
        self.save()

    def save(self):
        (self.directory / "requests.json").write_text(json.dumps(self.attempts, indent=2))


def probes(fixture, statistics):
    """Explicit expected outcomes from raw provider counts, not the parser's values."""
    from Scripts.data_platform.publication_identity import tracking_identity
    result = parse_fixture_result(fixture, statistics)
    fid = str(fixture["fixture"]["id"])
    output = []
    def check(group, key, side, line, expected):
        p = {"fixture": {"event_id": fid, "league": "diagnostic"}, "market": {"group": group, "key": key},
             "decision": {"quote": {"side": side, "line": line, "bookmaker": "CONTRACT PROBE ONLY",
                                     "market_key": key, "period": "regulation_time"}}}
        prediction = {"fixture_id": fid, "market": group, "side": side, "line": line,
                      "pick": side if group == "correct_score" else f"{side} {line}",
                      "bookmaker": "CONTRACT PROBE ONLY", "tracking": tracking_identity(p)}
        actual = grade_selection(prediction, result).outcome
        output.append({"market": key, "side": side, "line": line,
                       "expected": expected, "actual": actual, "passed": actual == expected})
    if result["status"] not in {"FT", "AET", "PEN"}:
        return output
    score = (fixture.get("score") or {}).get("fulltime") or {}
    h, a = score.get("home"), score.get("away")
    if type(h) is int and type(a) is int:
        total = h + a
        check("goals", "totals", "over", total + .25, "half_miss")
        check("goals", "totals", "under", total + .25, "half_hit")
        check("goals", "totals", "over", total, "push")
        check("btts", "btts", "yes", None, "hit" if h and a else "miss")
        check("moneyline", "h2h", "home", None, "hit" if h > a else "miss")
        check("correct_score", "correct_score", f"{h}-{a}", None, "hit")
        check("spreads", "spreads", "home", a - h + .25, "half_hit")
        check("spreads", "spreads", "away", h - a - .25, "half_miss")
    if result["status"] == "FT":
        ids = {fixture["teams"][s]["id"] for s in ("home", "away")}
        paired = [r for r in (statistics or []) if r["team"]["id"] in ids]
        if len(paired) == 2 and len({r["team"]["id"] for r in paired}) == 2:
            for group, key, label in [("corners", "totals_corners_over_under", "Corner Kicks"),
                                      ("sot", "shots_on_target_over_under", "Shots on Goal")]:
                values = [next((v.get("value") for v in r["statistics"] if v["type"] == label), None) for r in paired]
                if all(type(v) is int and v >= 0 for v in values):
                    total = sum(values)
                    check(group, key, "over", total + .25, "half_miss")
                    check(group, key, "under", total + .25, "half_hit")
                    check(group, key, "over", total, "push")
    # No real bookmaker card policy is inferred from a fixture's team totals.
    check("cards", "totals_cards_over_under", "over", 4.5, None)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-fixtures", type=int, default=8)
    parser.add_argument("--max-requests", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-from", type=Path, help="Replay checksum-verified saved responses; zero API requests")
    args = parser.parse_args(argv)
    if not (1 <= args.max_fixtures <= 8 and 1 <= args.max_requests <= 16):
        parser.error("This diagnostic is limited to 8 fixtures / 16 requests")
    url = make_url(config.SETTINGS.database_url)
    if url.get_backend_name() != "sqlite" or not url.database:
        parser.error("Diagnostic requires a local SQLite database for enforced read-only access")
    database = Path(url.database).resolve()
    def connection():
        conn = sqlite3.connect(f"file:{quote(str(database))}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only=ON")
        return conn
    engine = create_engine("sqlite://", creator=connection)
    @contextmanager
    def read_only():
        with Session(engine) as session:
            yield session
    repo = PredictionRepository(read_only)
    before = repo.settlement_candidates(include_graded=True)
    eligible = [p for p in before if p.get("recommendation_id") and
                utc_datetime(p.get("scheduled_kickoff") or p.get("kickoff")) and
                utc_datetime(p.get("scheduled_kickoff") or p.get("kickoff")) < datetime.now(timezone.utc)]
    # Round-robin across market groups, most recent first; deterministic IDs.
    ordered = sorted(eligible, key=lambda p: (p.get("kickoff") or "", p["id"]), reverse=True)
    buckets = {m: [p for p in ordered if p["market"] == m] for m in sorted({p["market"] for p in ordered})}
    selected = []
    while len(selected) < args.max_fixtures and any(buckets.values()):
        for items in buckets.values():
            if items and len(selected) < args.max_fixtures:
                fid = items.pop(0)["fixture_id"]
                if fid not in selected:
                    selected.append(fid)
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    budget = EvidenceBudget(args.output)
    deadline = time.monotonic() + 180
    def guard():
        if time.monotonic() >= deadline:
            raise BudgetExceeded("Diagnostic wall time reached")
    cache = {}
    if args.replay_from:
        for record in json.loads((args.replay_from / "requests.json").read_text()):
            if record.get("success"):
                source = args.replay_from / record["file"]
                body = source.read_text()
                if sha256(body.encode()).hexdigest() != record["sha256"]:
                    raise ValueError("Saved provider response checksum mismatch")
                cache[(record["path"], record["fixture_id"])] = body
    client = None if args.replay_from else BoundedFootballClient(budget,
        cycle_limit=args.max_requests, daily_limit=args.max_requests, guard=guard, settings=config.SETTINGS)
    def fetch(path, fid):
        if args.replay_from:
            body = cache[(path, fid)]
            (args.output / f"{fid}-{path.rsplit('/', 1)[-1]}.json").write_text(body)
            return json.loads(body)["response"]
        try:
            data = client._get(path, {"id" if path == "/fixtures" else "fixture": int(fid)})
            filename = f"{fid}-{path.rsplit('/', 1)[-1]}.json"
            body = json.dumps(data, indent=2)
            (args.output / filename).write_text(body)
            budget.attempts[-1].update(path=path, fixture_id=fid, success=True,
                                       file=filename, sha256=sha256(body.encode()).hexdigest())
            return data["response"]
        except Exception as exc:
            if budget.attempts and "path" not in budget.attempts[-1]:
                budget.attempts[-1].update(path=path, fixture_id=fid, success=False, error_type=type(exc).__name__)
            raise
        finally:
            budget.save()
    report = {"read_only": True, "database": str(database), "selected_fixture_ids": selected,
              "replay_source": str(args.replay_from) if args.replay_from else None,
              "before_fingerprint": fingerprint(before), "rows_before": len(before),
              "published_candidates": len(eligible), "results": [], "errors": []}
    for fid in selected:
        try:
            rows = fetch("/fixtures", fid)
            if len(rows) != 1 or str(rows[0]["fixture"]["id"]) != fid:
                raise ValueError("Fixture identity mismatch")
            fx = rows[0]
            stats = fetch("/fixtures/statistics", fid) if fx["fixture"]["status"]["short"] == "FT" else None
            parsed = parse_fixture_result(fx, stats)
            actual = []
            for p in eligible:
                if p["fixture_id"] == fid:
                    from Scripts.data_platform.sync.apifootball import COMPETITIONS
                    spec = COMPETITIONS.get(p["league"])
                    if spec is None or fx["league"]["id"] != spec.api_football_id:
                        raise ValueError("Fixture competition mismatch")
                    actual.append({"prediction_id": p["id"], "bookmaker": p["bookmaker"],
                                   "market": p["market"], **grade_selection(p, parsed).to_dict()})
            report["results"].append({"fixture_id": fid, "status": parsed["status"],
                "published_dry_run": actual, "contract_probes_NOT_published_bets": probes(fx, stats)})
        except Exception as exc:
            report["errors"].append({"fixture_id": fid, "error_type": type(exc).__name__})
            if isinstance(exc, BudgetExceeded):
                break
    report["after_fingerprint"] = fingerprint(repo.settlement_candidates(include_graded=True))
    report["database_rows_unchanged"] = report["before_fingerprint"] == report["after_fingerprint"]
    checks = [p for r in report["results"] for p in r["contract_probes_NOT_published_bets"]]
    actual = [p for r in report["results"] for p in r["published_dry_run"]]
    report["summary"] = {"provider_attempts": len(budget.attempts), "fixtures_verified": len(report["results"]),
        "published_selections": len(actual), "would_grade": sum(p["outcome"] is not None for p in actual),
        "pending_reasons": dict(Counter(p["pending_reason"] for p in actual if p["pending_reason"])),
        "synthetic_contract_probes": len(checks), "contract_probe_failures": sum(not p["passed"] for p in checks),
        "database_rows_unchanged": report["database_rows_unchanged"], "errors": len(report["errors"])}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({**report["summary"], "report": str(args.output / "report.json")}, indent=2))
    return int(bool(report["errors"]) or not report["database_rows_unchanged"] or any(not p["passed"] for p in checks))


if __name__ == "__main__":
    raise SystemExit(main())
