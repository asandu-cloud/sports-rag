"""Audit/recover Phase 1 historical inputs without training or publishing.

prepare: eight bounded fixture-list requests by default, no detail downloads.
apply: additive canonical import with an online backup and refresh writer gate.
audit-cards: read-only comparison against checksum-verified local raw archives.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Scripts.data_platform.features.historical_recovery import (
    LEGACY_FILES, apply_recovery_plan, build_recovery_plan, numeric,
)
from Scripts.ops.prediction_features import experiment_directory
from Scripts.ops.prediction_baseline import sqlite_backup


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def prepare(root, name, season, leagues, max_api_calls, client):
    if len(leagues) != len(set(leagues)) or len(leagues) > max_api_calls:
        raise ValueError("Duplicate leagues or fixture-list request budget exceeded")
    target = experiment_directory(root, name)
    sources, legacy, metadata = {}, {}, {}
    for code in leagues:
        path = root / "Output" / LEGACY_FILES[code].format(season=season)
        raw = path.read_bytes()
        legacy[code] = json.loads(raw)
        sources[code] = {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(raw).hexdigest()}
        # Copy exact exports so reports can be independently reconstructed.
        with (target / f"legacy-{code}.json").open("xb") as handle:
            handle.write(raw)
    from Scripts.data_platform.sync.apifootball import COMPETITIONS
    for code in leagues:
        print(f"Fetching fixture identities: {code}:{season} (no statistics/player calls)", flush=True)
        rows = client.fixtures(league=COMPETITIONS[code].api_football_id, season=season, status="FT")
        if not rows:
            raise ValueError(f"Empty historical fixture list for {code}; incomplete preparation retained")
        metadata[code] = rows
        write(target / f"fixtures-{code}.json", {"fetched_at": datetime.now(timezone.utc).isoformat(), "response": rows})
        sources[code]["metadata_sha256"] = digest(target / f"fixtures-{code}.json")
    plan = build_recovery_plan(metadata, legacy, season=season, sources=sources)
    write(target / "plan.json", plan)
    write(target / "report.json", {k: v for k, v in plan.items() if k != "entries"})
    write(target / "COMPLETE.json", {p.name: digest(p) for p in sorted(target.glob("*.json"))})
    return target


def verified_plan(path):
    manifest = json.loads((path / "COMPLETE.json").read_text())
    if "plan.json" not in manifest or "report.json" not in manifest:
        raise ValueError("Incomplete history preparation")
    for name, checksum in manifest.items():
        if Path(name).name != name or (path / name).is_symlink() or digest(path / name) != checksum:
            raise ValueError("History artifact checksum/path mismatch")
    plan = json.loads((path / "plan.json").read_text())
    # Reconstruct the plan from frozen source evidence rather than trusting
    # a hand-edited derived plan with a freshly recomputed checksum.
    codes = list(plan["coverage"])
    metadata = {code: json.loads((path / f"fixtures-{code}.json").read_text())["response"] for code in codes}
    legacy = {code: json.loads((path / f"legacy-{code}.json").read_text()) for code in codes}
    if not plan["entries"]:
        raise ValueError("No eligible historical fixtures")
    sources = {entry["league"]: entry["source"] for entry in plan["entries"]}
    for code in codes:
        if code not in sources:
            raise ValueError("No accepted fixtures for a requested competition")
        if (sources[code]["sha256"] != digest(path / f"legacy-{code}.json")
                or sources[code]["metadata_sha256"] != digest(path / f"fixtures-{code}.json")):
            raise ValueError("Source provenance does not match frozen files")
    rebuilt = build_recovery_plan(metadata, legacy, season=plan["entries"][0]["season"], sources=sources,
                                  as_of=datetime.fromisoformat(plan["prepared_at"]))
    # JSON objects have no order. The original CLI league order need not match
    # the sorted object keys written into the artifact.
    def comparable(value):
        return {**value, "entries": sorted(value["entries"], key=lambda e: (e["league"], e["api_row"]["fixture"]["id"])),
                "quarantined": sorted(value["quarantined"], key=lambda e: json.dumps(e, sort_keys=True))}
    if comparable(rebuilt) != comparable(plan):
        raise ValueError("Historical plan differs from frozen source evidence")
    return plan


def apply(root, name, prepared):
    from sqlalchemy.engine import make_url
    from Scripts.data_platform.config import SETTINGS
    from Scripts.data_platform.db import session_scope
    from Scripts.data_platform.services.refresh_coordination import data_access
    plan = verified_plan(prepared)
    url = make_url(SETTINGS.database_url)
    if url.get_backend_name() != "sqlite" or not url.database or not Path(url.database).is_file():
        raise ValueError("Recovery requires the existing file-backed canonical SQLite database")
    target = experiment_directory(root, name)
    database = Path(url.database).resolve()
    with data_access(writer=True, wait_seconds=180):
        backup = sqlite_backup(database, target / "platform-before.db")
        # Record rollback location *before* the atomic import starts.
        write(target / "before.json", {"database": str(database), "backup": backup,
                                      "prepared": str(prepared.resolve()), "plan_sha256": digest(prepared / "plan.json")})
        with session_scope() as session:
            result = apply_recovery_plan(session, plan)
        write(target / "result.json", result)
    print(json.dumps(result, indent=2))
    return target


def audit_cards(database):
    """No implicit null repair: report supported values/conflicts explicitly."""
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        archives = db.execute("SELECT * FROM raw_payload_archive WHERE provider='api_football' "
                              "AND endpoint='/fixtures/statistics' ORDER BY fetched_at,id").fetchall()
        stats = db.execute("SELECT f.api_football_id AS fid, tm.api_football_id AS tid, t.yellow_cards,t.red_cards, "
                           "c.code,s.year FROM fixture_team_stats t JOIN fixtures f ON f.id=t.fixture_id "
                           "JOIN teams tm ON tm.id=t.team_id JOIN competitions c ON c.id=f.competition_id "
                           "JOIN seasons s ON s.id=f.season_id WHERE f.status='FT'").fetchall()
    raw, bad = {}, []
    for archive in archives:
        try:
            uri = urlparse(archive["storage_uri"])
            if archive["storage_backend"] != "local" or uri.scheme != "file" or uri.netloc:
                raise ValueError("Not a local archive")
            path = Path(unquote(uri.path)).resolve()
            if not path.is_relative_to(database.resolve().parent / "raw_archive"):
                raise ValueError("Archive outside expected root")
            content = gzip.decompress(path.read_bytes())
            if hashlib.sha256(content).hexdigest() != archive["payload_digest"]:
                raise ValueError("Archive payload hash mismatch")
            fid = json.loads(archive["params"])["fixture"]
            payload = json.loads(content)
            ids = [t["team"]["id"] for t in payload]
            if len(ids) != 2 or len(set(ids)) != 2:
                raise ValueError("Invalid archived team pair")
            for team in payload:
                entries = team["statistics"]
                if len({s["type"] for s in entries}) != len(entries):
                    raise ValueError("Duplicate statistics in archived team")
                raw[(fid, team["team"]["id"])] = ({s["type"]: s["value"] for s in entries}, archive["id"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            bad.append({"archive_id": archive["id"], "reason": str(exc)})
    cohorts, supported, conflicts = {}, [], []
    for row in stats:
        cohort = cohorts.setdefault(f"{row['code']}:{row['year']}", Counter())
        cohort["team_rows"] += 1
        if row["yellow_cards"] is not None and row["red_cards"] is not None:
            cohort["complete_card_rows"] += 1
        evidence, archive_id = raw.get((row["fid"], row["tid"]), ({}, None))
        for col, key in (("yellow_cards", "Yellow Cards"), ("red_cards", "Red Cards")):
            if row[col] is None:
                cohort[col + "_missing"] += 1
                value = numeric(evidence.get(key))
                if value is not None and value >= 0 and value.is_integer():
                    supported.append({"fixture_id": row["fid"], "team_id": row["tid"], "column": col,
                                      "value": int(value), "archive_id": archive_id})
                    cohort[col + "_numeric_archive_evidence"] += 1
                elif archive_id is not None:
                    cohort[col + "_archive_also_unknown"] += 1
                else:
                    cohort[col + "_no_archive"] += 1
            elif numeric(evidence.get(key)) is not None and numeric(evidence[key]) != row[col]:
                conflicts.append({"fixture_id": row["fid"], "team_id": row["tid"], "column": col,
                                  "stored": row[col], "archived": evidence[key], "archive_id": archive_id})
    return {"cohorts": {key: dict(value) for key, value in cohorts.items()}, "supported_repair_candidates": supported,
            "conflicts": conflicts, "invalid_archives": bad, "writes_performed": False,
            "policy": "Unknown is not zero; player-card sums do not prove complete team-card coverage."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--experiment", required=True)
    p.add_argument("--season", required=True, type=int)
    p.add_argument("--league", nargs="+", choices=sorted(LEGACY_FILES), default=list(LEGACY_FILES))
    p.add_argument("--max-api-calls", type=int, default=8, help="Fixture-list calls; transport retries may add requests")
    p = sub.add_parser("apply")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--experiment", required=True)
    p = sub.add_parser("audit-cards")
    p.add_argument("--database", type=Path, default=ROOT / "Index/platform.db")
    p.add_argument("--experiment", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        from Scripts.data_platform.sync.apifootball import ApiFootballClient
        print(prepare(ROOT, args.experiment, args.season, args.league, args.max_api_calls, ApiFootballClient()))
    elif args.command == "apply":
        print(apply(ROOT, args.experiment, args.prepared))
    else:
        target = experiment_directory(ROOT, args.experiment)
        write(target / "card-audit.json", audit_cards(args.database))
        print(target)


if __name__ == "__main__":
    main()
