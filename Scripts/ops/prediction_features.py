#!/usr/bin/env python3
"""Export/replay isolated candidate feature datasets. Never train or publish."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.data_platform.features.model_dataset import capture_fixture_snapshot, load_canonical_inputs, supported_competitions
from Scripts.rag_ingest.core.model_features import (FeaturePolicy, MARKETS, RATE_FIELDS, SCHEMA_VERSION,
                                                  capture_snapshot, inference_features, training_features)


def experiment_directory(root: Path, name: str) -> Path:
    """New directory only, inside the dedicated tree; reject symlink escapes."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", name):
        raise ValueError("Experiment ID must be a simple new name, not a path")
    root = root.resolve()
    base = root / "Index" / "prediction_experiments"
    if base.resolve() != base:
        raise ValueError("Experiment root cannot use symlinks")
    target = base / name
    base.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    return target


def _write(path: Path, value) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _hash_file(path: Path) -> str:
    import hashlib
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def export_dataset(*, root: Path, database: Path, name: str, seasons: list[int], leagues: list[str] | None = None,
                   availability: str = "observed", limit: int | None = None) -> Path:
    competitions = supported_competitions()
    requested = leagues or list(competitions)
    if not seasons or not set(requested).issubset(competitions) or (limit is not None and limit < 1):
        raise ValueError("Specify valid provider seasons, registry competitions and a positive limit")
    targets, history, source_report = load_canonical_inputs(database)
    labels = {row["fixture_id"]: row for row in history}
    selected = [f for f in targets if f["competition"] in requested and f["season"] in seasons and f["status"] == "FT"]
    selected.sort(key=lambda row: (row["kickoff"], row["fixture_id"]))
    if limit:
        selected = selected[:limit]
    target = experiment_directory(root, name)
    source_paths = (ROOT / "Scripts/rag_ingest/core/model_features.py", Path(__file__).resolve(),
                    ROOT / "Scripts/data_platform/features/model_dataset.py")
    source_hashes = {str(path.relative_to(ROOT)): _hash_file(path) for path in source_paths}
    with (target / "feature-builder.py").open("xb") as handle:
        handle.write(source_paths[0].read_bytes())
    _write(target / "inputs.json", {"fixtures": selected, "history": history, "competitions": competitions})
    coverage = {f"{league}:{season}": {"exported": 0, "markets": {market: {"labelled": 0, "usable_features": 0} for market in MARKETS}}
                for league in requested for season in seasons}
    missing = Counter()
    schema = None
    with (target / "features.jsonl").open("x") as handle:
        for fixture in selected:
            snapshot = capture_snapshot(fixture, history, as_of=fixture["kickoff"], competitions=competitions, availability=availability)
            features = training_features(snapshot)
            if schema is None:
                schema = {"version": SCHEMA_VERSION, "names": features["names"], "policy": asdict(FeaturePolicy()),
                          "missing_values": "null plus explicit indicator; imputation must be fitted on training partition only"}
            label = labels.get(fixture["fixture_id"])
            outcomes = {m: label["home"][m] + label["away"][m] if label and label["home"][m] is not None and label["away"][m] is not None else None for m in MARKETS}
            cohort = coverage[f"{fixture['competition']}:{fixture['season']}"]
            cohort["exported"] += 1
            eligible = {}
            for market in MARKETS:
                usable = outcomes[market] is not None and all(features["profiles"][side][RATE_FIELDS[market]] is not None for side in ("home", "away"))
                eligible[market] = usable
                cohort["markets"][market]["labelled"] += int(outcomes[market] is not None)
                cohort["markets"][market]["usable_features"] += int(usable)
            missing.update(name for name, value in zip(features["names"], features["values"]) if value is None)
            # History is stored once in inputs.json; snapshot hashes let every
            # row be reconstructed/verified without repeating megabytes of data.
            handle.write(json.dumps({"fixture": fixture, "as_of": snapshot["as_of"], "snapshot_id": snapshot["snapshot_id"],
                                     "values": features["values"], "labels": outcomes, "feature_eligible": eligible,
                                     "profile_audit": features["profile_audit"]}, sort_keys=True, allow_nan=False) + "\n")
    _write(target / "feature-schema.json", schema or {"version": SCHEMA_VERSION, "names": [], "policy": asdict(FeaturePolicy())})
    _write(target / "coverage.json", {**source_report, "cohorts": coverage, "missing_feature_counts": dict(missing),
                                     "qualification": "Feature availability only; no model/league is approved for promotion"})
    _write(target / "manifest.json", {"feature_version": SCHEMA_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
                                     "availability": availability, "seasons": seasons, "leagues": requested, "row_count": len(selected),
                                     "limited_export": limit, "policy": asdict(FeaturePolicy()),
                                     "publication_enabled": False, "training_performed": False,
                                     "replay_boundary": "canonical fixture history -> pre-match feature vector (not baseline predictions)",
                                     "source_hashes": source_hashes})
    if any(_hash_file(ROOT / path) != checksum for path, checksum in source_hashes.items()):
        raise RuntimeError("Feature implementation changed during export; incomplete export retained for diagnosis")
    artifacts = ("inputs.json", "features.jsonl", "feature-schema.json", "coverage.json", "manifest.json", "feature-builder.py")
    _write(target / "COMPLETE.json", {name: _hash_file(target / name) for name in artifacts})
    return target


def replay_dataset(path: Path) -> dict:
    complete = json.loads((path / "COMPLETE.json").read_text())
    expected = {"inputs.json", "features.jsonl", "feature-schema.json", "coverage.json", "manifest.json", "feature-builder.py"}
    is_snapshot = set(complete) == {"input-snapshot.json", "feature-vector.json", "feature-builder.py", "manifest.json"}
    if is_snapshot:
        expected = set(complete)
    if set(complete) != expected:
        raise ValueError("Invalid experiment artifact manifest")
    for name, checksum in complete.items():
        if _hash_file(path / name) != checksum:
            raise ValueError(f"Experiment artifact checksum mismatch: {name}")
    manifest = json.loads((path / "manifest.json").read_text())
    if _hash_file(path / "feature-builder.py") != manifest["source_hashes"]["Scripts/rag_ingest/core/model_features.py"]:
        raise ValueError("Frozen feature implementation hash mismatch")
    # The pure builder has no repo imports. Replay trusted local experiments
    # with their captured implementation, even after the working tree changes.
    frozen = ModuleType("_frozen_prematch_features")
    frozen.__file__ = str(path / "feature-builder.py")
    sys.modules[frozen.__name__] = frozen
    # Avoid importlib's bytecode writes inside an immutable experiment bundle.
    exec(compile((path / "feature-builder.py").read_bytes(), frozen.__file__, "exec"), frozen.__dict__)
    if is_snapshot:
        snapshot = json.loads((path / "input-snapshot.json").read_text())
        features = json.loads((path / "feature-vector.json").read_text())
        if frozen.inference_features(snapshot) != features:
            raise ValueError("Captured fixture feature replay mismatch")
        return {"verified_rows": 1, "feature_version": features["feature_version"],
                "availability": snapshot["availability"], "model_promoted": False}
    inputs = json.loads((path / "inputs.json").read_text())
    schema = json.loads((path / "feature-schema.json").read_text())
    count = 0
    with (path / "features.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            snapshot = frozen.capture_snapshot(row["fixture"], inputs["history"], as_of=row["as_of"], competitions=inputs["competitions"],
                                               availability=manifest["availability"], policy=frozen.FeaturePolicy(**manifest["policy"]))
            features = frozen.inference_features(snapshot)
            if features["snapshot_id"] != row["snapshot_id"] or features["values"] != row["values"] or features["names"] != schema["names"]:
                raise ValueError(f"Training/inference replay mismatch: {row['fixture']['fixture_id']}")
            count += 1
    if count != manifest["row_count"]:
        raise ValueError("Experiment row count mismatch")
    return {"verified_rows": count, "feature_version": manifest["feature_version"], "availability": manifest["availability"], "model_promoted": False}


def export_fixture(*, root: Path, database: Path, name: str, fixture_id: int, as_of: datetime) -> Path:
    """Freeze a real pre-match input for subsequent offline numerical replay."""
    builder = ROOT / "Scripts/rag_ingest/core/model_features.py"
    source_hash = _hash_file(builder)
    source = builder.read_bytes()
    snapshot = capture_fixture_snapshot(database, fixture_id, as_of=as_of)
    features = inference_features(snapshot)
    target = experiment_directory(root, name)
    _write(target / "input-snapshot.json", snapshot)
    _write(target / "feature-vector.json", features)
    with (target / "feature-builder.py").open("xb") as handle:
        handle.write(source)
    _write(target / "manifest.json", {"kind": "prospective_feature_snapshot", "feature_version": SCHEMA_VERSION,
                                     "publication_enabled": False, "training_performed": False,
                                     "source_hashes": {"Scripts/rag_ingest/core/model_features.py": source_hash}})
    if _hash_file(builder) != source_hash or _hash_file(target / "feature-builder.py") != source_hash:
        raise RuntimeError("Feature implementation changed during capture")
    artifacts = ("input-snapshot.json", "feature-vector.json", "feature-builder.py", "manifest.json")
    _write(target / "COMPLETE.json", {name: _hash_file(target / name) for name in artifacts})
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--experiment", required=True)
    export.add_argument("--database", type=Path, default=ROOT / "Index/platform.db")
    export.add_argument("--season", nargs="+", type=int, required=True)
    export.add_argument("--league", nargs="+", choices=sorted(supported_competitions()))
    export.add_argument("--availability", choices=("observed", "assumed_final"), default="observed")
    export.add_argument("--limit", type=int)
    replay = sub.add_parser("replay")
    replay.add_argument("--experiment", type=Path, required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--experiment", required=True)
    capture.add_argument("--fixture-id", type=int, required=True)
    capture.add_argument("--database", type=Path, default=ROOT / "Index/platform.db")
    capture.add_argument("--as-of", type=datetime.fromisoformat, default=None)
    args = parser.parse_args()
    if args.command == "export":
        print(export_dataset(root=ROOT, database=args.database, name=args.experiment, seasons=args.season,
                             leagues=args.league, availability=args.availability, limit=args.limit))
    elif args.command == "capture":
        print(export_fixture(root=ROOT, database=args.database, name=args.experiment, fixture_id=args.fixture_id,
                             as_of=args.as_of or datetime.now(timezone.utc)))
    else:
        print(json.dumps(replay_dataset(args.experiment), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
