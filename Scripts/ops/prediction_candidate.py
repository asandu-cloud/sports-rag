#!/usr/bin/env python3
"""Explicit, isolated candidate fitting and inference. No promotion command.

This is an input/IO safety runner, not Phase 3's model benchmark. A fit alone
does not qualify a model, supply a validated R², or enable a production blend.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.ops.prediction_features import _hash_file, _write, experiment_directory, replay_dataset
from Scripts.rag_ingest.core.candidate_features import FEATURE_IMPLEMENTATION, candidate_predict
from Scripts.rag_ingest.core.model_features import MARKETS, SCHEMA_VERSION, digest, utc


def fit_candidate(*, root: Path, dataset: Path, name: str, market: str, train_through: datetime) -> Path:
    """Fit only an explicitly bounded training subset; preserve future rows.

    No tuning, CV, calibration, ROI or promotion is performed. The complete
    source dataset is verified first; fitted imputation sees training rows only.
    """
    import joblib
    import numpy as np
    import sklearn
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cutoff = utc(train_through)
    if market not in MARKETS:
        raise ValueError("Unsupported candidate market")
    replay_dataset(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text())
    schema = json.loads((dataset / "feature-schema.json").read_text())
    inputs = json.loads((dataset / "inputs.json").read_text())
    if manifest["source_hashes"]["Scripts/rag_ingest/core/model_features.py"] != FEATURE_IMPLEMENTATION or schema["version"] != SCHEMA_VERSION:
        raise ValueError("Re-export features with the current feature implementation before fitting")
    known = {row["fixture_id"]: row for row in inputs["history"]}
    rows = []
    excluded = {"outside_training_cutoff": 0, "unavailable_label_or_features": 0, "label_observed_later": 0}
    with (dataset / "features.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            completed = utc(row["fixture"]["kickoff"]) + timedelta(hours=manifest["policy"]["assumed_completion_hours"])
            if completed >= cutoff:
                excluded["outside_training_cutoff"] += 1
                continue
            if not row["feature_eligible"][market] or row["labels"][market] is None:
                excluded["unavailable_label_or_features"] += 1
                continue
            observation = known[row["fixture"]["fixture_id"]].get("observed_at")
            if manifest["availability"] == "observed" and (not observation or utc(observation) > cutoff):
                excluded["label_observed_later"] += 1
                continue
            rows.append(row)
    if len(rows) < 5:
        raise ValueError("Fewer than five eligible training rows; inspect coverage/cutoff first")
    # Five is a software smoke-test minimum, not statistical qualification.
    x = np.asarray([[float("nan") if v is None else v for v in row["values"]] for row in rows], dtype=float)
    y = np.asarray([row["labels"][market] for row in rows], dtype=float)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=50.0)),
    ])
    target = experiment_directory(root, name)
    model.fit(x, y)
    (target / "candidate").mkdir()
    joblib.dump(model, target / "candidate/model.joblib")
    contract = {"feature_version": schema["version"], "names": schema["names"], "policy_id": digest(manifest["policy"]),
                "registry_id": digest(inputs["competitions"]), "implementation_id": FEATURE_IMPLEMENTATION}
    _write(target / "manifest.json", {"kind": "unvalidated_candidate", "market": market, "feature_contract": contract,
                                     "train_through": cutoff.isoformat(), "trained_rows": len(rows), "exclusions": excluded,
                                     "training_fixture_ids": [row["fixture"]["fixture_id"] for row in rows],
                                     "training_snapshot_ids": [row["snapshot_id"] for row in rows],
                                     "source_dataset_manifest_hash": _hash_file(dataset / "manifest.json"),
                                     "source_dataset_artifacts": json.loads((dataset / "COMPLETE.json").read_text()),
                                     "source_availability": manifest["availability"],
                                     "created_at": datetime.now(timezone.utc).isoformat(),
                                     "dependencies": {"python": sys.version.split()[0], "numpy": np.__version__,
                                                      "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
                                     "runner_source_hash": _hash_file(Path(__file__)),
                                     "estimator": "median-imputer / standard-scaler / Ridge(alpha=50)",
                                     "qualification": "UNVALIDATED: chronological benchmarks still required",
                                     "publication_enabled": False, "promotion_allowed": False})
    _write(target / "COMPLETE.json", {key: _hash_file(target / key) for key in ("candidate/model.joblib", "manifest.json")})
    return target


def predict_candidate_bundle(candidate: Path, snapshot: dict) -> dict:
    """Load only an explicitly named trusted local candidate; never active models."""
    import joblib

    artifacts = json.loads((candidate / "COMPLETE.json").read_text())
    if set(artifacts) != {"candidate/model.joblib", "manifest.json"}:
        raise ValueError("Invalid candidate bundle manifest")
    for key, checksum in artifacts.items():
        if _hash_file(candidate / key) != checksum:
            raise ValueError("Candidate artifact checksum mismatch")
    metadata = json.loads((candidate / "manifest.json").read_text())
    if metadata.get("kind") != "unvalidated_candidate" or metadata.get("promotion_allowed") is not False:
        raise ValueError("Not an isolated, unvalidated candidate")
    model = joblib.load(candidate / "candidate/model.joblib")
    return {"market": metadata["market"], "value": candidate_predict(model, snapshot, feature_contract=metadata["feature_contract"]),
            "snapshot_id": snapshot["snapshot_id"], "qualification": metadata["qualification"], "published": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--experiment", required=True)
    train.add_argument("--market", choices=MARKETS, required=True)
    train.add_argument("--train-through", type=datetime.fromisoformat, required=True)
    predict = sub.add_parser("predict")
    predict.add_argument("--candidate", type=Path, required=True)
    predict.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        print(fit_candidate(root=ROOT, dataset=args.dataset, name=args.experiment, market=args.market, train_through=args.train_through))
    else:
        print(json.dumps(predict_candidate_bundle(args.candidate, json.loads(args.snapshot.read_text())), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
