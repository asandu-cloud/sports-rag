"""Offline, evidence-backed repair of missing team card counts.

Only accepts the saved Phase 1 diagnostic format. No provider requests,
null-to-zero inference, model changes, or settlement-rule conversion.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from Scripts.ops.prediction_features import experiment_directory
from Scripts.ops.prediction_history import digest, write
from Scripts.ops.prediction_history_diagnostic import verify_prepared, request_key
from Scripts.ops.prediction_baseline import sqlite_backup


ROOT = Path(__file__).resolve().parents[2]
FIELDS = {"Yellow Cards": "yellow_cards", "Red Cards": "red_cards"}
SCHEMA = "evidenced-card-stat-repair.v1"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _count(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
        return int(number) if number >= 0 and number.is_integer() else None
    except (TypeError, ValueError, OverflowError):
        return None


def _verified_facts(directory):
    """Rebuild facts from checksum-verified responses, not an analysis report."""
    approved = {request_key(r) for r in verify_prepared(directory)["requests"]}
    local = json.loads((directory / "local-snapshot.json").read_text())
    fixtures = {f["fixture_id"]: f for f in local["fixtures"]}
    facts = {}
    for receipt in sorted((directory / "provider").glob("*.receipt.json")):
        if receipt.is_symlink():
            raise ValueError("Symlink provider evidence is not allowed")
        info = json.loads(receipt.read_text())
        if not info.get("success"):
            continue
        request_path = receipt.with_name(receipt.name.replace(".receipt.json", ".request.json"))
        body_path = receipt.with_name(receipt.name.replace(".receipt.json", ".body"))
        if request_path.is_symlink() or body_path.is_symlink():
            raise ValueError("Symlink provider evidence is not allowed")
        request = json.loads(request_path.read_text())["request"]
        if request_key(request) not in approved:
            raise ValueError("Provider request was not in the verified diagnostic plan")
        if request["endpoint"] != "/fixtures/statistics":
            continue
        if digest(body_path) != info["body_sha256"]:
            raise ValueError("Provider evidence checksum mismatch")
        fid = request["params"]["fixture"]
        fixture = fixtures[fid]
        payload = json.loads(body_path.read_text())
        if payload.get("errors") or not isinstance(payload.get("response"), list):
            raise ValueError("Invalid provider response")
        if str(payload.get("parameters", {}).get("fixture")) != str(fid):
            raise ValueError("Provider response fixture identity mismatch")
        if payload.get("paging", {}).get("total", 1) != 1:
            raise ValueError("Incomplete paginated statistics")
        blocks = payload["response"]
        if not blocks:
            continue
        ids = [b["team"]["id"] for b in blocks]
        if len(ids) != 2 or set(ids) != {fixture["home_id"], fixture["away_id"]}:
            raise ValueError("Provider team pair differs from verified fixture")
        observed = datetime.fromisoformat(info["completed_at"])
        if observed.tzinfo is None:
            raise ValueError("Provider observation needs a timezone")
        for block in blocks:
            labels = [s["type"] for s in block["statistics"]]
            if len(labels) != len(set(labels)):
                raise ValueError("Duplicate provider statistic")
            for stat in block["statistics"]:
                field = FIELDS.get(stat["type"])
                value = _count(stat.get("value"))
                if field is None:
                    continue
                key = (fid, block["team"]["id"], field)
                if value is None:
                    # A later unknown is not positive evidence for a repair.
                    facts.pop(key, None)
                    continue
                facts[key] = {"fixture_id": fid, "team_id": key[1], "field": field, "value": value,
                              "observed_at": info["completed_at"], "body_sha256": digest(body_path),
                              "receipt_sha256": digest(receipt), "request_sha256": digest(request_path),
                              "body_file": str(body_path.relative_to(directory))}
    return fixtures, facts


def _row(db, fixture_id, team_id):
    return db.execute("""SELECT t.*, f.status, f.api_football_id AS fixture_api_id,
        f.kickoff_utc, f.home_goals, f.away_goals, c.code, s.year,
        h.api_football_id AS home_api_id, a.api_football_id AS away_api_id
        FROM fixture_team_stats t JOIN fixtures f ON f.id=t.fixture_id
        JOIN teams tm ON tm.id=t.team_id JOIN teams h ON h.id=f.home_team_id
        JOIN teams a ON a.id=f.away_team_id JOIN competitions c ON c.id=f.competition_id
        JOIN seasons s ON s.id=f.season_id WHERE f.api_football_id=? AND tm.api_football_id=?""",
        (fixture_id, team_id)).fetchone()


def _same_fixture(row, fixture):
    return row is not None and row["status"] == fixture["status"] == "FT" and all([
        row["code"] == fixture["league"], row["year"] == fixture["season"],
        row["home_api_id"] == fixture["home_id"], row["away_api_id"] == fixture["away_id"],
        row["kickoff_utc"] == fixture["kickoff"],
        row["home_goals"] == fixture["home_goals"], row["away_goals"] == fixture["away_goals"],
    ])


def build_plan(database, evidence):
    fixtures, facts = _verified_facts(evidence)
    updates, conflicts = [], []
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        for key, fact in sorted(facts.items()):
            row = _row(db, *key[:2])
            if not _same_fixture(row, fixtures[key[0]]):
                conflicts.append({**fact, "reason": "fixture_identity_or_result_changed"})
            elif row[key[2]] is None:
                updates.append(fact)
            elif row[key[2]] != fact["value"]:
                conflicts.append({**fact, "stored": row[key[2]], "reason": "existing_value_preserved"})
    return {"schema": SCHEMA, "evidence_directory": str(evidence.resolve()),
            "prepared_sha256": digest(evidence / "PREPARED.json"), "updates": updates, "conflicts": conflicts,
            "policy": "fill_missing_explicit_counts_only; retain_conflicts; no_historical_availability_claim"}


def apply_updates(db, plan):
    """Caller supplies one transaction; any stale field aborts the whole repair."""
    if plan.get("schema") != SCHEMA:
        raise ValueError("Unsupported card repair plan")
    evidence = Path(plan["evidence_directory"])
    if digest(evidence / "PREPARED.json") != plan["prepared_sha256"]:
        raise ValueError("Diagnostic preparation changed")
    fixtures, facts = _verified_facts(evidence)
    plan_id = _hash(plan)
    now = datetime.now(timezone.utc).isoformat()
    changed, already = [], []
    seen = set()
    for update in plan["updates"]:
        key = (update["fixture_id"], update["team_id"], update["field"])
        if key in seen or facts.get(key) != update:
            raise ValueError("Repair plan differs from provider evidence")
        seen.add(key)
        row = _row(db, *key[:2])
        if not _same_fixture(row, fixtures[key[0]]):
            raise ValueError("Fixture identity/result changed after review")
        metadata = json.loads(row["stats_json"] or "{}")
        repairs = metadata.setdefault("_card_stat_repairs", [])
        if row[key[2]] is not None:
            if row[key[2]] == update["value"] and any(r["plan_id"] == plan_id and r["field"] == key[2] for r in repairs):
                already.append(update)
                continue
            raise ValueError("Card field changed after review; existing value preserved")
        repairs.append({**update, "plan_id": plan_id, "applied_at": now,
                        "previous_raw_payload_digest": row["raw_payload_digest"]})
        metadata[key[2]] = update["value"]
        # Field is verified against the fixed FIELDS allowlist via facts above.
        db.execute(f"UPDATE fixture_team_stats SET {key[2]}=?,stats_json=?,raw_payload_digest=?,updated_at=? WHERE id=?",
                   (update["value"], json.dumps(metadata, sort_keys=True), _hash(metadata), now, row["id"]))
        changed.append(update)
    return {"plan_id": plan_id, "updated_fields": changed, "already_applied": already,
            "publication_or_model_changes": False}


def apply(root, name, plan_path):
    from sqlalchemy.engine import make_url
    from Scripts.data_platform.config import SETTINGS
    from Scripts.data_platform.services.refresh_coordination import data_access, RefreshBusy
    url = make_url(SETTINGS.database_url)
    if url.get_backend_name() != "sqlite" or not url.database:
        raise ValueError("Repair requires the existing canonical SQLite database")
    database = Path(url.database).resolve()
    if not database.is_file():
        raise ValueError("Canonical database is missing")
    plan = json.loads(plan_path.read_text())
    target = experiment_directory(root, name)
    target.chmod(0o700)
    with data_access(writer=True, wait_seconds=10):
        if Path(str(database) + ".refresh-incomplete").exists():
            raise RefreshBusy("Incomplete refresh must recover before card repairs")
        backup = sqlite_backup(database, target / "platform-before.db")
        write(target / "before.json", {"database": str(database), "backup": backup, "plan": plan})
        with closing(sqlite3.connect(database)) as db:
            db.row_factory = sqlite3.Row
            with db:
                db.execute("BEGIN IMMEDIATE")
                result = apply_updates(db, plan)
        write(target / "result.json", result)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--database", type=Path, default=ROOT / "Index/platform.db")
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--experiment", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--experiment", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        plan = build_plan(args.database, args.evidence)
        target = experiment_directory(ROOT, args.experiment)
        write(target / "plan.json", plan)
        print(json.dumps({"directory": str(target), "updates": len(plan["updates"]), "conflicts": len(plan["conflicts"])}))
    else:
        print(apply(ROOT, args.experiment, args.plan))


if __name__ == "__main__":
    main()
