"""Immutable development sidecars and explicitly isolated candidate bundles."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

from Scripts.rag_ingest.core.model_features import digest, utc

ROOT = Path(__file__).resolve().parents[4]
BASELINE_VERSION = "phase3-development-baselines.v1"
BUNDLE_VERSION = "phase3-benchmark-candidate.v1"
ALLOWED_DATASET_FILES = ("COMPLETE.json", "manifest.json", "feature-schema.json", "splits.json",
                         "lockbox-policy.json", "development/features.jsonl", "development/labels.jsonl")
PACKAGES = ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl",
            "lightgbm", "xgboost", "catboost")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def new_experiment(root, name):
    root = Path(root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", name):
        raise ValueError("Use a simple new experiment name")
    base = root / "Index/prediction_experiments"
    if base.resolve() != base:
        raise ValueError("Experiment tree cannot contain symlinks")
    base.mkdir(parents=True, exist_ok=True)
    target = base / name
    target.mkdir(exist_ok=False)
    return target


def complete(directory):
    directory = Path(directory)
    hashes = {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob("*"))
              if p.is_file() and p != directory / "COMPLETE.json" and ".runtime" not in p.relative_to(directory).parts}
    write_json(directory / "COMPLETE.json", hashes)


def verify_complete(directory, *, required=()):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "COMPLETE.json")
    if not isinstance(manifest, dict) or not set(required).issubset(manifest):
        raise ValueError("Incomplete benchmark artifact manifest")
    for name, expected in manifest.items():
        relative = Path(name)
        item = directory / relative
        if (relative.is_absolute() or ".." in relative.parts or item.is_symlink()
                or not item.resolve().is_relative_to(directory)):
            raise ValueError("Unsafe artifact path")
        if not isinstance(expected, str) or sha(item) != expected:
            raise ValueError(f"Benchmark artifact checksum mismatch: {name}")
    return manifest


def dependencies():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def source_hashes():
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths += [ROOT / "Scripts/ops/prediction_benchmark.py", ROOT / "Scripts/rag_ingest/core/model_features.py",
              ROOT / "Scripts/data_platform/features/phase3_features.py", ROOT / "requirements-phase3.lock.txt"]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def prepare_baselines(*, root, dataset, name):
    """Preparation only: decode frozen audit history, emit development predictions.

    This operation is explicitly separate from model fitting. Reserved labels
    are never scored; its new sidecar has no audit history or held-out rows.
    """
    from .data import DevelopmentDataset
    from .baselines import baseline_metadata, build_baseline_rows
    from .isolation import offline_guard

    dataset = Path(dataset).resolve()
    data = DevelopmentDataset(dataset)
    checked = read_json(dataset / "COMPLETE.json")
    history_path = dataset / "audit/inputs.json"
    if history_path.resolve() != history_path or sha(history_path) != checked.get("audit/inputs.json"):
        raise ValueError("Frozen baseline history checksum mismatch")
    end = data.splits["boundaries"]["phase3_confirmation_start"]
    # Freeze all baseline rules/source fingerprints before any forecast or score.
    baseline = baseline_metadata(ROOT)
    hashes = source_hashes()
    target = new_experiment(root, name)
    readable = [dataset / file for file in ALLOWED_DATASET_FILES] + [history_path]
    write_json(target / "preparation.json", {"dataset_id": data.manifest["dataset_id"],
                "kind": BASELINE_VERSION, "confirmation_start": end, "baseline": baseline,
                "source_hashes": hashes, "training_performed": False, "publication_enabled": False,
                "source_audit_sha256": checked["audit/inputs.json"]})
    try:
        with offline_guard(root=root, output=target, readable_files=readable):
            inputs = read_json(history_path)
            hours = data.schema["policy"]["assumed_completion_hours"]
            history = [row for row in inputs["history"]
                       if utc(row["kickoff"]) + timedelta(hours=hours) < utc(end)]
            del inputs["history"]
            rows = build_baseline_rows(data.rows, history, inputs["competitions"],
                                       confirmation_start=end, scoring=baseline["scoring"])
            with (target / "predictions.jsonl").open("x") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            write_json(target / "manifest.json", {"version": BASELINE_VERSION,
                "dataset_id": data.manifest["dataset_id"], "feature_contract_id": data.manifest["feature_contract_id"],
                "baseline": baseline, "confirmation_start": end, "rows": len(rows),
                "scope": "development_only", "training_performed": False,
                "publication_enabled": False, "promotion_allowed": False,
                "source_hashes": hashes, "created_at": datetime.now(timezone.utc).isoformat()})
        if source_hashes() != hashes or baseline_metadata(ROOT) != baseline:
            raise RuntimeError("Baseline preparation source/config changed; completion withheld")
        complete(target)
    except Exception as exc:
        write_json(target / "FAILED.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    return target


def load_baselines(path, data):
    path = Path(path)
    verify_complete(path, required={"manifest.json", "preparation.json", "predictions.jsonl"})
    manifest = read_json(path / "manifest.json")
    if (manifest.get("version") != BASELINE_VERSION or manifest.get("scope") != "development_only"
            or manifest.get("dataset_id") != data.manifest["dataset_id"]
            or manifest.get("feature_contract_id") != data.manifest["feature_contract_id"]
            or manifest.get("publication_enabled") is not False):
        raise ValueError("Baseline sidecar contract mismatch")
    features = {row["fixture"]["fixture_id"]: row for row in data.rows}
    result = {}
    with (path / "predictions.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["fixture_id"], row["market"])
            original = features.get(row["fixture_id"])
            if (key in result or original is None or row["snapshot_id"] != original["snapshot_id"]
                    or row["feature_contract_id"] != data.manifest["feature_contract_id"]
                    or row["market"] not in ("goals", "corners", "sot")
                    or not original["market_eligibility"][row["market"]]["eligible"]):
                raise ValueError("Baseline row identity/cohort mismatch")
            result[key] = row
    if len(result) != manifest["rows"]:
        raise ValueError("Baseline sidecar row count mismatch")
    return manifest, result


def save_candidate(path, model, *, metadata):
    """Only trusted isolated research estimators; never the active model folder."""
    import joblib
    path = Path(path)
    if (metadata.get("market") != model.market or metadata.get("names") != list(model.preprocessor.names)
            or model.metadata.get("market") != model.market or model.metadata.get("family") != model.family):
        raise ValueError("Candidate identity does not match its fitted estimator")
    if path.resolve().is_relative_to(ROOT / "Index") and not path.resolve().is_relative_to(ROOT / "Index/prediction_experiments"):
        raise ValueError("Candidate destination must be an isolated experiment")
    qualification = metadata.get("qualification", "UNVALIDATED_BATCH_B_SMOKE")
    if qualification not in {"UNVALIDATED_BATCH_B_SMOKE", "UNCONFIRMED_BATCH_C_DEVELOPMENT"}:
        raise ValueError("Unsupported research qualification")
    path.mkdir(parents=True, exist_ok=False)
    actual = {**metadata, "family": model.family, "version": BUNDLE_VERSION, "qualification": qualification,
              "publication_enabled": False, "promotion_allowed": False,
              "dependencies": dependencies(), "python": sys.version.split()[0], "estimator": model.metadata}
    joblib.dump(model, path / "model.joblib", compress=3)
    write_json(path / "manifest.json", actual)
    complete(path)


def predict_candidate(path, rows, *, feature_contract_id):
    """Load a trusted explicitly named bundle and enforce its saved inference contract."""
    import joblib
    import numpy as np
    path = Path(path)
    artifacts = verify_complete(path, required={"model.joblib", "manifest.json"})
    if set(artifacts) != {"model.joblib", "manifest.json"}:
        raise ValueError("Unexpected candidate artifacts")
    metadata = read_json(path / "manifest.json")
    if (metadata.get("version") != BUNDLE_VERSION or metadata.get("publication_enabled") is not False
            or metadata.get("promotion_allowed") is not False
            or metadata.get("feature_contract_id") != feature_contract_id
            or metadata.get("dependencies") != dependencies()
            or metadata.get("python") != sys.version.split()[0]):
        raise ValueError("Candidate feature/environment/isolation contract mismatch")
    if not rows or any(r.get("feature_contract_id") != feature_contract_id for r in rows):
        raise ValueError("Inference rows do not match candidate feature contract")
    names = metadata["names"]
    if any(len(r["values"]) != len(names) or not re.fullmatch(r"[0-9a-f]{64}", r.get("snapshot_id", "")) for r in rows):
        raise ValueError("Invalid candidate feature vector or snapshot identity")
    matrix = np.asarray([[np.nan if v is None else v for v in r["values"]] for r in rows], dtype=float)
    model = joblib.load(path / "model.joblib")
    if (model.metadata != metadata["estimator"] or model.market != metadata.get("market")
            or model.family != metadata.get("family") or list(model.preprocessor.names) != metadata.get("names")):
        raise ValueError("Serialized estimator metadata mismatch")
    return model.predict(matrix, names=names)


def predict_strategy(path, rows, *, feature_contract_id, statistical_rows, baseline_contract_id):
    """Apply the frozen Batch C strategy to explicitly equivalent baseline rows.

    predict_candidate is the estimator-only inference/replay API. This API
    additionally enforces the saved statistical reconstruction contract and
    fixture identity before applying the frozen convex weight. Neither API
    retrieves live profiles or makes a public prediction.
    """
    import numpy as np
    path = Path(path)
    verify_complete(path, required={"model.joblib", "manifest.json"})
    metadata = read_json(path / "manifest.json")
    if (metadata.get("qualification") != "UNCONFIRMED_BATCH_C_DEVELOPMENT"
            or metadata.get("baseline_contract_id") != baseline_contract_id
            or metadata.get("strategy") not in {"statistical", "selected_ml", "selected_blend"}):
        raise ValueError("Frozen strategy/baseline contract mismatch")
    weight = metadata.get("weight_ml")
    if type(weight) not in (int, float) or not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Invalid frozen convex weight")
    if (metadata["strategy"] == "statistical" and weight != 0
            or metadata["strategy"] == "selected_ml" and weight != 1):
        raise ValueError("Frozen strategy disagrees with its weight")
    lookup = {}
    for baseline in statistical_rows:
        key = baseline.get("fixture_id")
        if key in lookup:
            raise ValueError("Duplicate statistical baseline fixture")
        lookup[key] = baseline
    ids = [row["fixture"]["fixture_id"] for row in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(lookup):
        raise ValueError("Frozen strategy requires identical baseline/inference fixtures")
    values = []
    for row in rows:
        baseline = lookup[row["fixture"]["fixture_id"]]
        if (baseline.get("snapshot_id") != row["snapshot_id"]
                or baseline.get("feature_contract_id") != feature_contract_id
                or baseline.get("market") != metadata["market"]):
            raise ValueError("Statistical baseline snapshot/market differs from candidate inputs")
        value = baseline.get("statistical")
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError("Invalid statistical baseline prediction")
        values.append(value)
    ml = predict_candidate(path, rows, feature_contract_id=feature_contract_id)
    return (1 - weight) * np.asarray(values, dtype=float) + weight * ml
