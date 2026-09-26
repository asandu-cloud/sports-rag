"""Batch B: one predeclared development smoke fold, no model selection or release."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from Scripts.rag_ingest.core.model_features import digest
from .artifacts import (ROOT, ALLOWED_DATASET_FILES, complete, dependencies, load_baselines,
                        new_experiment, predict_candidate, read_json, save_candidate,
                        sha, source_hashes, write_json)
from .data import DevelopmentDataset, family_gate
from .estimators import DEFAULT_PARAMS, FAMILIES, INTERACTION_BASES, fit_estimator
from .isolation import offline_guard
from .metrics import evaluate_predictions

VERSION = "phase3-benchmark-smoke.v1"
MARKETS = ("goals", "corners", "sot")


@contextmanager
def runtime_environment(directory):
    values = {name: "2" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT")}
    values.update(MPLCONFIGDIR=str(Path(directory) / ".runtime/matplotlib"))
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield values
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def environment_report():
    """Require an isolated venv with the resolved lock, including native imports."""
    prefix = Path(sys.prefix).resolve()
    if prefix == Path(sys.base_prefix).resolve() or prefix == (ROOT / ".venv").resolve():
        raise ValueError("Use an isolated research virtual environment, not the production/system Python")
    config = prefix / "pyvenv.cfg"
    if not config.exists() or "include-system-site-packages = false" not in config.read_text().lower():
        raise ValueError("Research environment must not inherit production/system site packages")
    lock = ROOT / "requirements-phase3.lock.txt"
    expected = {}
    for line in lock.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, version = line.split("==")
            expected[name] = version
    actual = {name: importlib.metadata.version(name) for name in expected}
    if expected != actual:
        raise ValueError("Research dependencies do not match requirements-phase3.lock.txt")
    for name in ("numpy", "scipy", "sklearn", "lightgbm", "xgboost", "catboost", "joblib"):
        importlib.import_module(name)
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "executable": sys.executable, "prefix": str(prefix), "dependencies": actual,
            "dependency_lock_sha256": sha(lock), "cpu_threads": 2, "seed": 42}


def _matrix(rows):
    return np.asarray([[np.nan if value is None else value for value in row["values"]]
                       for row in rows], dtype=float)


def _metrics(targets, predictions):
    result = evaluate_predictions(targets, predictions)
    # Per-fixture forecast/target evidence is saved separately, not duplicated
    # several times inside aggregate reports.
    result.pop("per_fixture", None)
    return result


def run_smoke(*, root, dataset, baselines, name, fold_id="development-01",
              markets=MARKETS, lookback_days=None, half_life_days=365):
    """Fit every agreed family on one fold; all results remain unqualified.

    Failures are retained per task with non-success run status. A model cannot
    silently disappear, remove evaluation fixtures or lower a support gate.
    No outer-fold selection, OOF blend learning or holdout opening occurs here.
    """
    root, dataset, baselines = Path(root).resolve(), Path(dataset).resolve(), Path(baselines).resolve()
    markets = tuple(markets)
    if not markets or len(set(markets)) != len(markets) or not set(markets).issubset(MARKETS):
        raise ValueError("Choose distinct eligible Phase 3 markets; cards are excluded")
    if lookback_days not in (None, 730, 1460) or half_life_days not in (None, 180, 365, 730):
        raise ValueError("History and decay must use the predeclared Phase 3 options")
    target = new_experiment(root, name)
    started = time.monotonic()
    try:
        with runtime_environment(target) as runtime:
            environment = environment_report()
            hashes = source_hashes()
            baseline_files = read_json(baselines / "COMPLETE.json")
            # Restrict every open under the dataset to development inputs. The
            # sidecar may contain only previously prepared development baselines.
            reads = [dataset / path for path in ALLOWED_DATASET_FILES]
            reads += [baselines / "COMPLETE.json", *(baselines / p for p in baseline_files)]
            with offline_guard(root=root, output=target, readable_files=reads):
                data = DevelopmentDataset(dataset)
                baseline_manifest, baseline_rows = load_baselines(baselines, data)
                selections = {market: data.select_fold(market, fold_id, lookback_days=lookback_days,
                                                       half_life_days=half_life_days) for market in markets}
                specification = {"version": VERSION, "purpose": "software_smoke_not_model_selection",
                    "dataset_id": data.manifest["dataset_id"], "dataset_artifacts": data.artifact_hashes,
                    "feature_contract_id": data.manifest["feature_contract_id"],
                    "fold_id": fold_id, "markets": list(markets), "families": list(FAMILIES),
                    "parameters": DEFAULT_PARAMS, "lookback_days": lookback_days,
                    "half_life_days": half_life_days, "seed": 42, "cpu_threads": 2,
                    "linear_interactions": INTERACTION_BASES,
                    "weighted_median_tie": "lower_observed_value", "all_missing_imputation": 0,
                    "source_hashes": hashes, "baseline_manifest_sha256": sha(baselines / "manifest.json"),
                    "baseline_versions": baseline_manifest["baseline"],
                    "cohorts": {market: selected[3] for market, selected in selections.items()},
                    "qualification": "UNVALIDATED_BATCH_B_SMOKE", "publication_enabled": False,
                    "promotion_allowed": False, "confirmation_opened": False,
                    "final_system_test_opened": False, "hyperparameter_selection": False,
                    "blend_learning": "Batch C only; no fitted blend in this smoke",
                    "metrics": "equal-fixture RMSE/MAE/bias/R2; common uncalibrated Poisson diagnostics",
                    "historical_betting_metrics": "unavailable_no_authentic_quote_dataset"}
                write_json(target / "specification.json", specification)
                write_json(target / "environment.json", {**environment, "runtime": runtime})
                spec_id = digest(specification)
                reports, failures = {}, []
                prediction_path = target / "predictions.jsonl"
                with prediction_path.open("x") as prediction_file:
                    for market, (train, validation, weights, support) in selections.items():
                        x, vx = _matrix(train), _matrix(validation)
                        y = np.asarray([row["target"] for row in train], dtype=float)
                        vy = np.asarray([row["target"] for row in validation], dtype=float)
                        reports[market] = {"support": support, "methods": {}}

                        def record(method, values, *, raw=None):
                            values = np.asarray(values, dtype=float)
                            if values.shape != vy.shape:
                                raise ValueError("A method returned a different evaluation cohort")
                            scores = _metrics(vy, values)
                            for index, row in enumerate(validation):
                                evidence = {"market": market, "method": method, "fold_id": fold_id,
                                    "fixture_id": row["fixture"]["fixture_id"], "kickoff": row["fixture"]["kickoff"],
                                    "competition": row["fixture"]["competition"], "season": row["fixture"]["season"],
                                    "snapshot_id": row["snapshot_id"], "target": float(vy[index]),
                                    "prediction": float(values[index]), "specification_id": spec_id,
                                    "qualification": "UNVALIDATED_BATCH_B_SMOKE"}
                                if raw is not None:
                                    evidence["raw_prediction"] = float(raw[index])
                                prediction_file.write(json.dumps(evidence, sort_keys=True, allow_nan=False) + "\n")
                            return scores

                        for method in ("league_average", "statistical"):
                            try:
                                values = [baseline_rows[(row["fixture"]["fixture_id"], market)][method]
                                          for row in validation]
                                reports[market]["methods"][method] = {"status": "complete", "metrics": record(method, values)}
                            except Exception as exc:
                                error = {"status": "failed", "type": type(exc).__name__, "message": str(exc)}
                                reports[market]["methods"][method] = error
                                failures.append({"market": market, "method": method, **error})
                        for family in FAMILIES:
                            gate = family_gate(support, family)
                            if not gate["qualified"]:
                                result = {"status": "insufficient_support", "gate": gate}
                                reports[market]["methods"][family] = result
                                failures.append({"market": market, "method": family, **result})
                                continue
                            print(f"Smoke {market}/{family}: {len(train)} train, {len(validation)} validation", flush=True)
                            task_started = time.monotonic()
                            try:
                                model = fit_estimator(family, x, y, weights, data.schema["names"], market)
                                prediction = model.predict_with_diagnostics(vx, names=data.schema["names"])
                                candidate = target / "candidates" / market / family
                                save_candidate(candidate, model, metadata={
                                    "dataset_id": data.manifest["dataset_id"], "market": market,
                                    "feature_contract_id": data.manifest["feature_contract_id"],
                                    "names": data.schema["names"], "feature_schema": data.schema,
                                    "specification_id": spec_id, "source_hashes": hashes,
                                    "fold_id": fold_id, "fit_cutoff": support["fit_cutoff"],
                                    "training_fixture_ids": [r["fixture"]["fixture_id"] for r in train],
                                    "training_snapshot_ids": [r["snapshot_id"] for r in train],
                                    "training_labels_sha256": digest(y.tolist()),
                                    "training_weights_sha256": digest(weights.tolist()),
                                    "support": support, "lookback_days": lookback_days,
                                    "half_life_days": half_life_days,
                                    "availability": data.manifest["availability"],
                                    "supported_scope": "unqualified_development_smoke_only"})
                                restored = predict_candidate(candidate, validation,
                                                feature_contract_id=data.manifest["feature_contract_id"])
                                if not np.array_equal(restored, prediction["values"]):
                                    raise RuntimeError("Saved candidate predictions differ from fitted estimator")
                                scores = record(family, restored, raw=prediction["raw_values"])
                                reports[market]["methods"][family] = {"status": "complete", "metrics": scores,
                                    "gate": gate, "seconds": time.monotonic() - task_started,
                                    "candidate": str(candidate.relative_to(target)),
                                    "clipped_count": prediction["clipped_count"], "serialization_parity": True}
                            except Exception as exc:
                                error = {"status": "failed", "type": type(exc).__name__, "message": str(exc)}
                                reports[market]["methods"][family] = error
                                failures.append({"market": market, "method": family, **error})
                    prediction_file.flush()
                status = "complete" if not failures else "incomplete"
                write_json(target / "report.json", {"version": VERSION, "status": status,
                    "qualification": "software_validation_only_not_evidence_of_improvement",
                    "specification_id": spec_id, "markets": reports, "failures": failures,
                    "seconds": time.monotonic() - started,
                    "confirmation_opened": False, "final_system_test_opened": False,
                    "publication_enabled": False, "promotion_allowed": False})
            if source_hashes() != hashes:
                raise RuntimeError("Research implementation changed during run; completion withheld")
            if not failures:
                complete(target)
            else:
                write_json(target / "INCOMPLETE.json", {"failures": failures})
        return target
    except Exception as exc:
        write_json(target / "FAILED.json", {"type": type(exc).__name__, "message": str(exc),
                                            "created_at": datetime.now(timezone.utc).isoformat()})
        raise
