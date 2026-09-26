#!/usr/bin/env python3
"""Phase 3 isolated preparation, smoke and chronological development runner.

Held-out targets and label-rich source history live in separate audit/lockbox
stores. The development reader opens only explicitly permitted partitions.
This is an auditable workflow boundary, not secrecy for public match results.
``smoke`` validates one fold; ``develop`` runs the frozen forward-development
procedure. Neither can publish/promote models or open reserved test outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import importlib.metadata
import json
from pathlib import Path
import re
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.ops.prediction_features import _hash_file, _write, experiment_directory
from Scripts.ops.prediction_baseline import sqlite_backup
from Scripts.data_platform.features.model_dataset import supported_competitions
from Scripts.data_platform.features.phase3_features import HistoryIndex, MARKETS, apply_support, build_reference, feature_contract
from Scripts.rag_ingest.core import model_features as core

VERSION = "phase3-dataset.v1"
DEVELOPMENT = {"initial_training", "development"}
SOURCE_PATHS = (
    "Scripts/rag_ingest/core/model_features.py",
    "Scripts/data_platform/features/model_dataset.py",
    "Scripts/data_platform/features/phase3_features.py",
    "Scripts/data_platform/features/market_eligibility.py",
    "Scripts/data_platform/features/chronological_splits.py",
    "Scripts/data_platform/features/staged_history.py",
    "Scripts/data_platform/sync/upserts.py",
    "Scripts/data_platform/registry/loader.py",
    "Scripts/data_platform/registry/competitions.yaml",
    "Scripts/team_stat_export.py",
    "Scripts/ops/prediction_benchmark.py",
    "docs/phase3-technical-plan-2026-09-26.md",
)


def _jsonl(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def _line(handle, value):
    handle.write(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")


def _store(partition):
    return "development" if partition in DEVELOPMENT else f"lockbox/{partition}"


def _active_hashes(root):
    base = root / "Index/ml_models"
    return {str(p.relative_to(root)): _hash_file(p) for p in sorted(base.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def coverage_report(rows, evidence):
    groups = defaultdict(lambda: {"fixtures": 0, "markets": {m: {"eligible": 0, "primary_exclusions": Counter(),
                                      "overlapping_exclusions": Counter(), "support_bands": Counter()}
                                      for m in (*MARKETS, "cards")}})
    missing = Counter()
    for row in rows:
        fixture = row["fixture"]
        ev = evidence[fixture["fixture_id"]]
        keys = ["all", f"league:{fixture['competition']}", f"season:{fixture['season']}",
                f"league_season:{fixture['competition']}:{fixture['season']}",
                f"source:{ev['source_class']}", f"round:{ev['round_group']}"]
        for key in keys:
            group = groups[key]
            group["fixtures"] += 1
            for market, decision in row["market_eligibility"].items():
                result = group["markets"][market]
                result["eligible"] += int(decision["eligible"])
                result["overlapping_exclusions"].update(decision["reasons"])
                if decision["reasons"]:
                    result["primary_exclusions"][decision["reasons"][0]] += 1
                if market in MARKETS:
                    for side in ("home", "away"):
                        result["support_bands"][side + ":" + row["support"][market][side]["band"]] += 1
        for index, value in enumerate(row["values"]):
            if value is None:
                missing[str(index)] += 1
    return {"groups": dict(groups), "missing_feature_index_counts": dict(missing),
            "qualification": "Eligibility and availability only. No estimator fitted, scores or deployment qualification."}


def prepare_dataset(*, root, name, database, as_of, existing_snapshot=False):
    from Scripts.data_platform.features.history_evidence import load_eligible_inputs
    from Scripts.data_platform.features.chronological_splits import choose_boundaries, assign_partitions

    root = root.resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", name):
        raise ValueError("Experiment ID must be a simple new name")
    target = root / "Index/prediction_experiments" / name
    if existing_snapshot:
        if target.resolve() != target or not target.is_dir() or set(p.name for p in target.iterdir()) - {
            "platform-snapshot.db", "snapshot-capture.json", "preflight-checks.json"
        }:
            raise ValueError("Existing research directory must contain only the newly captured snapshot/preflight")
        if not (target / "platform-snapshot.db").is_file() or (target / "platform-snapshot.db").is_symlink():
            raise ValueError("Missing captured snapshot")
    else:
        target = experiment_directory(root, name)
        _write(target / "snapshot-capture.json", sqlite_backup(database, target / "platform-snapshot.db"))
    as_of = core.utc(as_of)
    protected = _active_hashes(root)
    (target / "audit").mkdir()
    (target / "source").mkdir()
    sources = {}
    for name in SOURCE_PATHS:
        path = ROOT / name
        sources[name] = _hash_file(path)
        (target / "source" / path.name).write_bytes(path.read_bytes())
    # Extra evidence helper, when present, is also captured and fingerprinted.
    extra = ROOT / "Scripts/data_platform/features/history_evidence.py"
    if extra.exists():
        sources[str(extra.relative_to(ROOT))] = _hash_file(extra)
        (target / "source" / extra.name).write_bytes(extra.read_bytes())
    print("Checking frozen local evidence...", flush=True)
    fixtures, history, evidence, evidence_report = load_eligible_inputs(target / "platform-snapshot.db", root, as_of=as_of)
    fixtures = sorted(fixtures, key=lambda f: (core.utc(f["kickoff"]), f["fixture_id"]))
    history = sorted(history, key=lambda f: (core.utc(f["kickoff"]), f["fixture_id"]))
    competitions = supported_competitions()
    _write(target / "audit/inputs.json", {"fixtures": fixtures, "history": history, "competitions": competitions})
    _write(target / "audit/evidence-report.json", evidence_report)
    with (target / "audit/evidence.jsonl").open("x") as handle:
        for fid in sorted(evidence):
            _line(handle, {"fixture_id": fid, **evidence[fid]})
    index = HistoryIndex(history)
    rows = []
    contract = None
    with (target / "audit/feature-rows.jsonl").open("x") as handle:
        for n, fixture in enumerate(sorted(fixtures, key=lambda f: (f["kickoff"], f["fixture_id"])), 1):
            snapshot, features, view = build_reference(fixture, index, competitions)
            if contract is None:
                contract = feature_contract(features["names"])
            if view["feature_contract_id"] != core.digest(contract):
                raise ValueError("Feature contract changed during export")
            fid = fixture["fixture_id"]
            decision = apply_support(evidence[fid]["market_eligibility"], view["support"])
            row = {"fixture": fixture, "as_of": snapshot["as_of"], "snapshot_id": snapshot["snapshot_id"],
                   "availability": "assumed_final", "observed_at": evidence[fid].get("actual_observed_at"),
                   "actual_observed_at": evidence[fid].get("actual_observed_at"),
                   "label_available_at": (core.utc(fixture["kickoff"]) + timedelta(hours=3)).isoformat(),
                   "forecast_stage": "reconstructed_immediately_before_kickoff", **view,
                   "market_eligibility": decision, "source_class": evidence[fid]["source_class"],
                   "round_group": evidence[fid]["round_group"]}
            _line(handle, row)
            rows.append(row)
            if n % 250 == 0:
                print(f"Features {n}/{len(fixtures)}", flush=True)
    if not rows:
        raise ValueError("No completed fixtures; retained incomplete diagnosis")
    boundaries = choose_boundaries(rows)
    splits = assign_partitions(rows, boundaries)
    # Membership API uses one explicit entry per provider fixture.
    memberships = splits["memberships"]
    by_id = {item["fixture_id"]: item["partition"] for item in memberships}
    files = {}
    try:
        for row in rows:
            fid = row["fixture"]["fixture_id"]
            partition = by_id.get(fid, "quarantine")
            location = _store(partition)
            if location not in files:
                directory = target / location
                directory.mkdir(parents=True, exist_ok=False)
                files[location] = ((directory / "features.jsonl").open("x"), (directory / "labels.jsonl").open("x"))
            row["partition"] = partition
            features_file, labels_file = files[location]
            _line(features_file, row)
            _line(labels_file, {"fixture_id": fid, "labels": evidence[fid]["labels"],
                               "team_labels": evidence[fid]["team_labels"]})
    finally:
        for pair in files.values():
            for handle in pair:
                handle.close()
    _write(target / "feature-schema.json", contract)
    _write(target / "splits.json", splits)
    coverage = coverage_report(rows, evidence)
    coverage["missing_feature_counts"] = {contract["names"][int(i)]: n for i, n in coverage.pop("missing_feature_index_counts").items()}
    _write(target / "coverage.json", coverage)
    _write(target / "lockbox-policy.json", {"version": "phase3-lockbox.v1", "inspection_events": [],
            "status": "unopened_for_model_evaluation", "development_partitions": sorted(DEVELOPMENT),
            "heldout_locations": sorted(name for name in files if name.startswith("lockbox/")),
            "audit_sources_contain_outcomes": True,
            "boundary": "Development reader cannot open held-out target stores or audit history. Human filesystem access is not a security boundary.",
            "confirmation_opening": "Requires frozen candidate/evaluation hashes and separate permanent inspection event in Batch D; not implemented here."})
    if _active_hashes(root) != protected:
        raise RuntimeError("Active artifacts changed during export; completion withheld")
    if any(_hash_file(ROOT / name) != sha for name, sha in sources.items()):
        raise RuntimeError("Research source changed during export; completion withheld")
    manifest = {"version": VERSION, "created_at": as_of.isoformat(), "availability": "assumed_final",
                "source_hashes": sources, "source_database": "platform-snapshot.db",
                "snapshot_sha256": _hash_file(target / "platform-snapshot.db"),
                "row_count": len(rows), "history_count": len(history), "seed": 42,
                "feature_contract_id": core.digest(contract), "training_performed": False,
                "publication_enabled": False, "promotion_allowed": False,
                "protected_model_hashes": protected, "league_weighting": "equal_fixture_before_recency",
                "profile_variants": "Reference two-season features exported; approved three/five-season paired comparisons require separately versioned rebuilds before Batch C.",
                "limitations": ["Retrospective final archives; not original provider vintages.",
                                "Historical referee assignments may have been unavailable before kickoff.",
                                "Regulation/source/round evidence exclusions may select the sample.",
                                "No historical betting-price dataset, probability calibration or accuracy claim."],
                "dependencies": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata.get("Name"))),
                "python": sys.version}
    manifest["dataset_id"] = core.digest({"snapshot": manifest["snapshot_sha256"], "features": manifest["feature_contract_id"],
                                         "inputs": _hash_file(target / "audit/inputs.json"), "splits": core.digest(splits),
                                         "evidence": _hash_file(target / "audit/evidence.jsonl"), "sources": sources})
    _write(target / "manifest.json", manifest)
    artifacts = {str(p.relative_to(target)): _hash_file(p) for p in sorted(target.rglob("*")) if p.is_file()}
    _write(target / "COMPLETE.json", artifacts)
    return target


def verify_dataset(path, *, development_only=False):
    complete = json.loads((path / "COMPLETE.json").read_text())
    required = {"manifest.json", "feature-schema.json", "splits.json", "coverage.json", "platform-snapshot.db",
                "audit/inputs.json", "audit/evidence.jsonl", "lockbox-policy.json"}
    if not required.issubset(complete):
        raise ValueError("Incomplete Phase 3 artifact manifest")
    allowed = {"manifest.json", "feature-schema.json", "splits.json", "lockbox-policy.json",
               "development/features.jsonl", "development/labels.jsonl"}
    if development_only and not allowed.issubset(complete):
        raise ValueError("Incomplete development dataset")
    for name, checksum in complete.items():
        if development_only and name not in allowed:
            continue
        item = path / name
        if item.resolve().is_relative_to(path.resolve()) is False or _hash_file(item) != checksum:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("version") != VERSION or manifest.get("training_performed") is not False:
        raise ValueError("Invalid Batch A dataset contract")
    schema = json.loads((path / "feature-schema.json").read_text())
    if manifest["feature_contract_id"] != core.digest(schema):
        raise ValueError("Feature contract mismatch")
    return manifest


def replay_dataset(path, *, limit=None):
    manifest = verify_dataset(path)
    frozen = ModuleType("_phase3_frozen_features")
    sys.modules[frozen.__name__] = frozen
    source = path / "source/model_features.py"
    if _hash_file(source) != manifest["source_hashes"]["Scripts/rag_ingest/core/model_features.py"]:
        raise ValueError("Frozen builder source mismatch")
    exec(compile(source.read_bytes(), str(source), "exec"), frozen.__dict__)
    view = ModuleType("_phase3_frozen_view")
    exec(compile((path / "source/phase3_features.py").read_bytes(), "phase3_features.py", "exec"), view.__dict__)
    view.reference = frozen
    inputs = json.loads((path / "audit/inputs.json").read_text())
    index = view.HistoryIndex(inputs["history"])
    count = 0
    for row in _jsonl(path / "audit/feature-rows.jsonl"):
        snapshot, _, result = view.build_reference(row["fixture"], index, inputs["competitions"], core=frozen)
        if snapshot["snapshot_id"] != row["snapshot_id"] or any(result[key] != row[key] for key in result):
            raise ValueError(f"Feature replay mismatch: {row['fixture']['fixture_id']}")
        count += 1
        if count % 500 == 0:
            print(f"Replay {count}/{manifest['row_count']}", flush=True)
        if limit and count >= limit:
            break
    if not limit and count != manifest["row_count"]:
        raise ValueError("Replay row count mismatch")
    return {"verified_rows": count, "full_replay": count == manifest["row_count"],
            "dataset_id": manifest["dataset_id"], "training_performed": False}


def load_development(path, *, market, fit_cutoff=None):
    """Only development targets; never decode held-out or audit outcomes."""
    from Scripts.data_platform.features.chronological_splits import training_rows
    if market not in MARKETS:
        raise ValueError("Market is not qualified for Phase 3")
    verify_dataset(path, development_only=True)
    splits = json.loads((path / "splits.json").read_text())
    if not splits["boundaries"]:
        raise ValueError("Insufficient eligible coverage for development partitions")
    end = splits["boundaries"]["phase3_confirmation_start"]
    if fit_cutoff is not None and core.utc(fit_cutoff) > core.utc(end):
        raise ValueError("Development fit cutoff crosses the held-out boundary")
    rows = list(_jsonl(path / "development/features.jsonl"))
    membership = {item["fixture_id"]: item for item in splits["memberships"]}
    if any(row["partition"] not in DEVELOPMENT or core.utc(row["fixture"]["kickoff"]) >= core.utc(end)
           or membership.get(row["fixture"]["fixture_id"], {}).get("partition") != row["partition"] for row in rows):
        raise ValueError("Held-out fixture in development store")
    labels = {r["fixture_id"]: r["labels"] for r in _jsonl(path / "development/labels.jsonl")}
    rows = [{**r, "target": labels[r["fixture"]["fixture_id"]][market]} for r in rows
            if r["market_eligibility"][market]["eligible"]]
    if any(row["target"] is None for row in rows):
        raise ValueError("Eligible row has a missing target")
    if fit_cutoff is not None:
        return training_rows(rows, fit_cutoff, availability="assumed_final", market=market)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--experiment", required=True)
    prepare.add_argument("--database", type=Path, default=ROOT / "Index/platform.db")
    prepare.add_argument("--as-of", required=True)
    prepare.add_argument("--existing-snapshot", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("--experiment", type=Path, required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--experiment", type=Path, required=True)
    replay.add_argument("--limit", type=int)
    baseline = sub.add_parser("prepare-baselines", help="Prepare a new development-only statistical sidecar; no fitting")
    baseline.add_argument("--dataset", type=Path, required=True)
    baseline.add_argument("--experiment", required=True)
    variants = sub.add_parser("prepare-variants", help="Prepare immutable development-only feature variants; no fitting")
    variants.add_argument("--dataset", type=Path, required=True)
    variants.add_argument("--experiment", required=True)
    develop = sub.add_parser("develop", help="Frozen Batch C forward search, OOF blends and paired feature comparisons")
    develop.add_argument("--dataset", type=Path, required=True)
    develop.add_argument("--baselines", type=Path, required=True)
    develop.add_argument("--variants", type=Path, required=True)
    develop.add_argument("--experiment", required=True)
    develop.add_argument("--resume", action="store_true")
    smoke = sub.add_parser("smoke", help="One fixed development fold; requires isolated locked research environment")
    smoke.add_argument("--dataset", type=Path, required=True)
    smoke.add_argument("--baselines", type=Path, required=True)
    smoke.add_argument("--experiment", required=True)
    smoke.add_argument("--fold", default="development-01")
    smoke.add_argument("--market", action="append", choices=MARKETS)
    smoke.add_argument("--lookback-days", choices=("all", "730", "1460"), default="all")
    smoke.add_argument("--half-life-days", choices=("none", "180", "365", "730"), default="365")
    verify_run = sub.add_parser("verify-run", help="Verify completed sidecar/run artifacts without scoring")
    verify_run.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare_dataset(root=ROOT, name=args.experiment, database=args.database,
                              as_of=args.as_of, existing_snapshot=args.existing_snapshot))
    elif args.command == "replay":
        print(json.dumps(replay_dataset(args.experiment, limit=args.limit), indent=2))
    elif args.command == "verify":
        manifest = verify_dataset(args.experiment)
        print(json.dumps({"dataset_id": manifest["dataset_id"], "rows": manifest["row_count"], "verified": True}))
    elif args.command == "prepare-baselines":
        from Scripts.data_platform.features.benchmarks.artifacts import prepare_baselines
        print(prepare_baselines(root=ROOT, dataset=args.dataset, name=args.experiment))
    elif args.command == "prepare-variants":
        from Scripts.data_platform.features.benchmarks.variants import prepare_variants
        print(prepare_variants(root=ROOT, dataset=args.dataset, name=args.experiment))
    elif args.command == "develop":
        from Scripts.data_platform.features.benchmarks.forward import run_development
        path = run_development(root=ROOT, dataset=args.dataset, baselines=args.baselines,
                               variants=args.variants, name=args.experiment, resume=args.resume)
        done = (path / "COMPLETE.json").exists()
        print(json.dumps({"experiment": str(path), "status": "complete" if done else "paused",
                          "qualification": "UNCONFIRMED_BATCH_C_DEVELOPMENT"}))
        if not done:
            raise SystemExit(2)
    elif args.command == "verify-run":
        from Scripts.data_platform.features.benchmarks.artifacts import verify_complete
        verified = verify_complete(args.experiment)
        print(json.dumps({"verified": True, "artifacts": len(verified)}))
    elif args.command == "smoke":
        from Scripts.data_platform.features.benchmarks.runner import run_smoke
        path = run_smoke(root=ROOT, dataset=args.dataset, baselines=args.baselines,
                         name=args.experiment, fold_id=args.fold, markets=args.market or MARKETS,
                         lookback_days=None if args.lookback_days == "all" else int(args.lookback_days),
                         half_life_days=None if args.half_life_days == "none" else int(args.half_life_days))
        report = json.loads((path / "report.json").read_text())
        print(json.dumps({"experiment": str(path), "status": report["status"],
                          "qualification": report["qualification"]}))
        if report["status"] != "complete":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
