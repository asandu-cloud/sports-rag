"""Bounded Batch C extension; no reserved targets, live I/O or promotion."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta
import json
from pathlib import Path
import re

import numpy as np

from Scripts.rag_ingest.core.model_features import digest, utc
from .artifacts import (ALLOWED_DATASET_FILES, load_baselines, new_experiment, read_json,
                        sha, source_hashes, verify_complete, write_json)
from .checkpoints import (Checkpoints, RunBudgetReached, atomic_write_json, complete_atomic,
                          experiment_lock, verify_committed_tasks)
from .data import DevelopmentDataset, family_gate, recency_weights
from .extension_selection import KINDS, choose_architecture, blend_from_oof
from .forward import _input_paths, _safe_resume, prediction_rows, write_jsonl
from .isolation import offline_guard
from .metrics import numerical_metrics
from .runner import MARKETS, _matrix, environment_report, runtime_environment

VERSION = "phase3-bounded-extension.v1"
QUALIFICATION = "UNCONFIRMED_BATCH_C_EXTENSION"
PROTOCOL = "docs/phase3-extension-protocol-2026-09-27.md"
LOOKBACK, HALF_LIFE, MAX_FITS = 1460, 365, 78
FAMILIES = dict(zip(KINDS, ("ridge", "xgboost", "poisson", "catboost")))


def _components(kind):
    if kind not in KINDS:
        raise ValueError("Unknown extension architecture")
    return ("home", "away") if kind.startswith("team_") else ("total",)


def _old_paths(path):
    """Allow only metadata and saved outer predictions, never original models."""
    manifest = read_json(path / "COMPLETE.json")
    result = read_json(path / "RESULT.json")
    report = result.get("report_directory", "")
    if not re.fullmatch(r"reports-\d{4}", report):
        raise ValueError("Invalid Batch C result pointer")
    names = ["RESULT.json", "specification.json", f"{report}/outer-predictions.jsonl"]
    for name in names:
        item = path / name
        if item.resolve() != item or sha(item) != manifest.get(name):
            raise ValueError("Batch C comparator checksum mismatch")
    return [path / "COMPLETE.json", *(path / n for n in names)]


def _load_old(path, data):
    paths = _old_paths(path)
    spec = read_json(path / "specification.json")
    if (spec.get("dataset_id") != data.manifest["dataset_id"]
            or spec.get("feature_contract_id") != data.manifest["feature_contract_id"]
            or spec.get("confirmation_opened") is not False
            or spec.get("publication_enabled") is not False):
        raise ValueError("Batch C comparator dataset/isolation mismatch")
    result = {}
    with paths[-1].open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["method"] not in {"selected_ml", "selected_blend"}:
                continue
            key = (row["market"], row["fold_id"], row["method"], row["fixture_id"])
            if key in result:
                raise ValueError("Duplicate Batch C comparator prediction")
            result[key] = row
    return result


def _common(rows, baselines, market):
    kept, excluded = [], []
    for row in rows:
        fid = row["fixture"]["fixture_id"]
        parts = [row["team_labels"][side][market] for side in ("home", "away")]
        if any(type(v) not in (int, float) or not np.isfinite(v) or v < 0 or v != int(v) for v in parts):
            raise ValueError("Eligible fixture lacks verified team counts")
        if sum(parts) != row["target"]:
            raise ValueError("Team targets do not sum to the frozen total")
        baseline = baselines.get((fid, market))
        if baseline is None:
            excluded.append({"fixture_id": fid, "reason": "missing_statistical_baseline"})
            continue
        if (baseline["fixture_id"] != fid or baseline["market"] != market
                or baseline["snapshot_id"] != row["snapshot_id"]
                or baseline["feature_contract_id"] != row["feature_contract_id"]):
            raise ValueError("Baseline feature/fixture/market identity mismatch")
        value = baseline.get("statistical")
        if type(value) not in (float, int) or not np.isfinite(value) or value <= 0:
            excluded.append({"fixture_id": fid, "reason": "nonpositive_or_invalid_statistical_baseline"})
            continue
        average = baseline.get("league_average")
        if type(average) not in (float, int) or not np.isfinite(average) or average < 0:
            raise ValueError("Invalid league-average comparator")
        kept.append(row)
    return kept, excluded


def _statistical(rows, baselines, market):
    return np.asarray([baselines[(r["fixture"]["fixture_id"], market)]["statistical"] for r in rows])


def _support(train, validation, cutoff):
    weights, report = recency_weights(train, cutoff, HALF_LIFE)
    return weights, {**report, "fit_cutoff": cutoff, "lookback_days": LOOKBACK,
        "training_unique_fixtures": len(train), "validation_unique_fixtures": len(validation),
        "training_membership_sha256": digest([r["fixture"]["fixture_id"] for r in train]),
        "validation_membership_sha256": digest([r["fixture"]["fixture_id"] for r in validation])}


def _component_task(checkpoints, *, kind, side, market, fold, train, validation,
                    weights, support, names, baselines, baseline_id, save_model=False):
    from .extension_models import fit_component
    if side not in _components(kind):
        raise ValueError("Wrong component for architecture")
    gate = family_gate(support, FAMILIES[kind])
    if not gate["qualified"]:
        return {"status": "insufficient_support", "gate": gate}
    y = np.asarray([r["target"] if side == "total" else r["team_labels"][side][market] for r in train], dtype=float)
    anchor = _statistical(train, baselines, market)
    v_anchor = _statistical(validation, baselines, market)
    evidence = {"fixture_ids": [r["fixture"]["fixture_id"] for r in train],
                "snapshot_ids": [r["snapshot_id"] for r in train], "targets": y.tolist(),
                "weights": weights.tolist(), "statistical": anchor.tolist(),
                "label_available_at": [r["label_available_at"] for r in train]}
    if (not train or any(utc(r["label_available_at"]) >= utc(fold["fit_cutoff"]) for r in train)
            or any(utc(r["fixture"]["kickoff"]) < utc(fold["fit_cutoff"]) - timedelta(days=LOOKBACK) for r in train)):
        raise ValueError("Training labels/window cross the fit cutoff")
    contract = {"kind": kind, "side": side, "market": market, "fold_id": fold["fold_id"],
                "fit_cutoff": fold["fit_cutoff"], "baseline_contract_id": baseline_id,
                "names": list(names), "training_sha256": digest(evidence), "support": support,
                "validation_snapshots": [r["snapshot_id"] for r in validation],
                "validation_statistical_sha256": digest(v_anchor.tolist()), "save_model": save_model}

    def compute(attempt):
        from .estimators import EstimatorFitError
        write_json(attempt / "training.json", evidence)
        write_json(attempt / "validation.json", {"fixture_ids": [r["fixture"]["fixture_id"] for r in validation],
            "snapshot_ids": contract["validation_snapshots"], "statistical": v_anchor.tolist()})
        try:
            model = fit_component(kind, _matrix(train), y, weights, names, market,
                                  statistical=anchor if side == "total" else None)
            prediction = model.predict_with_diagnostics(_matrix(validation), names=names,
                                  statistical=v_anchor if side == "total" else None)
        except EstimatorFitError as exc:
            return {"status": "failed", "failure": {"type": type(exc).__name__, "message": str(exc)},
                    "predictions": np.asarray([]), "raw_predictions": np.asarray([])}
        diagnostics = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in prediction.items()
                       if k not in {"values", "raw_values"}}
        write_json(attempt / "diagnostics.json", diagnostics)
        result = {"status": "complete", "kind": kind, "side": side,
                  "predictions": prediction["values"], "raw_predictions": prediction["raw_values"],
                  "clipped_count": prediction["clipped_count"], "estimator": model.metadata,
                  "training_max_label_available_at": max(r["label_available_at"] for r in train)}
        if save_model:
            import joblib
            joblib.dump(model, attempt / "component.joblib", compress=3)
            restored = joblib.load(attempt / "component.joblib")
            replay = restored.predict_with_diagnostics(_matrix(validation), names=names,
                                                       statistical=v_anchor if side == "total" else None)
            if any(not np.array_equal(replay[k], prediction[k]) for k in ("values", "raw_values")):
                raise RuntimeError("Component serialization replay mismatch")
            result["component_path"] = str((attempt / "component.joblib").relative_to(checkpoints.path.parent))
            result["serialization_parity"] = True
        return result
    result = checkpoints.get(contract, compute)
    if result["status"] == "complete" and len(result["predictions"]) != len(validation):
        raise ValueError("Component checkpoint prediction cohort mismatch")
    return result


def _rows(rows, values, market, fold, method, names, **extra):
    predictions = prediction_rows(rows, values, market=market, fold=fold, method=method, names=names)
    return [{**r, "qualification": QUALIFICATION, "label_available_at": source["label_available_at"], **extra}
            for r, source in zip(predictions, rows, strict=True)]


def freeze_strategy(rows, *, cutoff):
    """Final strategy selection also respects the pre-confirmation label cutoff."""
    comparisons = []
    for order, method in enumerate(("statistical", "selected_blend", "selected_ml")):
        original = [r for r in rows if r["method"] == method]
        if any(utc(r["kickoff"]) >= utc(cutoff) for r in original):
            raise ValueError("Final strategy received a held-out forecast")
        selected = sorted((r for r in original if utc(r["label_available_at"]) < utc(cutoff)),
                          key=lambda r: (utc(r["kickoff"]), r["fixture_id"]))
        ids = [r["fixture_id"] for r in selected]
        if not selected or len(set(ids)) != len(ids):
            raise ValueError("Final strategy requires nonempty unique fixture cohorts")
        comparisons.append({"method": method, "tie_order": order, "membership_sha256": digest(ids),
            "targets_sha256": digest([r["target"] for r in selected]),
            "snapshots_sha256": digest([r["snapshot_id"] for r in selected]),
            "late_label_exclusions": sorted(r["fixture_id"] for r in original if utc(r["label_available_at"]) >= utc(cutoff)),
            "metrics": numerical_metrics([r["target"] for r in selected], [r["prediction"] for r in selected])})
    for key in ("membership_sha256", "targets_sha256", "snapshots_sha256"):
        if len({r[key] for r in comparisons}) != 1:
            raise ValueError("Final strategy cohorts, targets or snapshots differ")
    return {"method": min(comparisons, key=lambda r: (r["metrics"]["rmse"], r["tie_order"]))["method"],
            "fit_cutoff": cutoff, "comparisons": comparisons}


def _old_values(old, rows, market, fold, method, names):
    references = _rows(rows, np.zeros(len(rows)), market, fold, method, names)
    fields = ("fixture_id", "snapshot_id", "target", "fold_id", "market", "competition", "season",
              "source_class", "round_group", "season_stage", "support_band", "missingness_band", "forecast_stage")
    result = []
    for reference in references:
        value = old.get((market, fold["fold_id"], method, reference["fixture_id"]))
        if (value is None or any(value[k] != reference[k] for k in fields)
                or utc(value["kickoff"]) != utc(reference["kickoff"])):
            raise ValueError("Batch C comparison must have identical snapshots, targets and slices")
        if type(value["prediction"]) not in (int, float) or not np.isfinite(value["prediction"]) or value["prediction"] < 0:
            raise ValueError("Invalid saved Batch C prediction")
        result.append(value["prediction"])
    return np.asarray(result)


def _execute(data, baselines, baseline_contract, old, target, checkpoints, spec_id):
    from .evaluation import summarize, paired_comparison
    from .extension_bundle import save_bundle, predict_bundle
    names, folds = data.schema["names"], data.folds
    baseline_id = digest(baseline_contract)
    raw, adaptive, outer, coverage, choices, tasks = defaultdict(list), defaultdict(list), [], [], {}, []
    validations = {}
    for market in MARKETS:
        choices[market] = {}
        for index, fold in enumerate(folds):
            train0, val0, _, original_support = data.select_fold(market, fold["fold_id"],
                                                  lookback_days=LOOKBACK, half_life_days=HALF_LIFE)
            train, excluded_train = _common(train0, baselines, market)
            validation, excluded_val = _common(val0, baselines, market)
            weights, support = _support(train, validation, fold["fit_cutoff"])
            item = {"market": market, "fold_id": fold["fold_id"], "original_support": original_support,
                    "support": support, "training_exclusions": excluded_train, "validation_exclusions": excluded_val}
            coverage.append(item)
            # Diagnose unsupported cohorts before spending any fits for this fold.
            gates = {kind: family_gate(support, FAMILIES[kind]) for kind in KINDS}
            if any(not g["qualified"] for g in gates.values()):
                write_json(target / f"unsupported-{market}-{fold['fold_id']}.json", {**item, "gates": gates})
                raise ValueError("Insufficient declared extension cohort; no automatic scope expansion")
            validations[(market, fold["fold_id"])] = validation
            choice = choose_architecture(raw[market], prior_fold_ids=[f["fold_id"] for f in folds[:index]],
                                         cutoff=fold["fit_cutoff"])
            blend = blend_from_oof(adaptive[market], choices[market], cutoff=fold["fit_cutoff"])
            choices[market][fold["fold_id"]] = choice
            choice_report = {"choice": choice, "blend": blend}
            item["selection"] = choice_report
            statistical = _statistical(validation, baselines, market)
            outputs, new_records = {}, {}
            for kind in KINDS:
                components = []
                for side in _components(kind):
                    result = _component_task(checkpoints, kind=kind, side=side, market=market, fold=fold,
                        train=train, validation=validation, weights=weights, support=support, names=names,
                        baselines=baselines, baseline_id=baseline_id)
                    tasks.append({"market": market, "fold_id": fold["fold_id"], "kind": kind, "side": side,
                                  "checkpoint_id": result.get("checkpoint_id"), "status": result["status"]})
                    if result["status"] != "complete":
                        write_json(target / f"failed-{market}-{fold['fold_id']}-{kind}-{side}.json", tasks[-1])
                        raise RuntimeError("Extension component failed; complete comparison withheld")
                    components.append(result)
                values = sum((r["predictions"] for r in components), np.zeros(len(validation)))
                outputs[kind] = values
                records = []
                for row, prediction, stat in zip(validation, values, statistical, strict=True):
                    records.append({"fixture_id": row["fixture"]["fixture_id"], "snapshot_id": row["snapshot_id"],
                        "market": market, "fold_id": fold["fold_id"], "kind": kind, "target": row["target"],
                        "prediction": float(prediction), "statistical": float(stat), "kickoff": row["fixture"]["kickoff"],
                        "label_available_at": row["label_available_at"], "fit_cutoff": fold["fit_cutoff"],
                        "training_max_label_available_at": max(r["training_max_label_available_at"] for r in components),
                        "component_checkpoints": [r["checkpoint_id"] for r in components]})
                new_records[kind] = records
                if index >= 2:
                    outer.extend(_rows(validation, values, market, fold, kind, names))
                print(f"Extension {market}/{fold['fold_id']}/{kind}: {checkpoints.created} new component fits", flush=True)
            selected = outputs[choice["kind"]]
            if index >= 2:
                methods = {"league_average": np.asarray([baselines[(r["fixture"]["fixture_id"], market)]["league_average"] for r in validation]),
                           "statistical": statistical, "selected_ml": selected,
                           "selected_blend": (1 - blend["weight_ml"]) * statistical + blend["weight_ml"] * selected}
                for previous in ("selected_ml", "selected_blend"):
                    methods["batch_c_" + previous] = _old_values(old, validation, market, fold, previous, names)
                for method, values in methods.items():
                    outer.extend(_rows(validation, values, market, fold, method, names))
            for kind in KINDS:
                raw[market].extend(new_records[kind])
            adaptive[market].extend({**r, "choice_id": choice["choice_id"]} for r in new_records[choice["kind"]])

    report = target / f"reports-{len(list(target.glob('reports-*'))) + 1:04d}"
    report.mkdir()
    frozen = {}
    cutoff = data.splits["boundaries"]["phase3_confirmation_start"]
    for market in MARKETS:
        choice = choose_architecture(raw[market], prior_fold_ids=[f["fold_id"] for f in folds], cutoff=cutoff)
        blend = blend_from_oof(adaptive[market], choices[market], cutoff=cutoff)
        strategy_choice = freeze_strategy([r for r in outer if r["market"] == market], cutoff=cutoff)
        strategy, comparison = strategy_choice["method"], strategy_choice["comparisons"]
        weight = 0.0 if strategy == "statistical" else 1.0 if strategy == "selected_ml" else blend["weight_ml"]
        proposal = {"kind": choice["kind"], "strategy": strategy, "weight_ml": weight,
                    "selection": choice, "blend": blend, "strategy_comparison": comparison,
                    "qualification": QUALIFICATION, "publication_enabled": False, "promotion_allowed": False}
        frozen[market] = proposal
        outer.extend({**r, "method": "frozen_strategy"} for r in list(outer)
                     if r["market"] == market and r["method"] == strategy)
        if weight == 0:
            proposal["status"] = "retain_statistical"
            continue
        training0 = [{**r, "target": r["labels"][market]} for r in data.rows
            if r["market_eligibility"][market]["eligible"] and utc(r["label_available_at"]) < utc(cutoff)
            and utc(r["fixture"]["kickoff"]) >= utc(cutoff) - timedelta(days=LOOKBACK)]
        training, excluded = _common(training0, baselines, market)
        replay = validations[(market, folds[-1]["fold_id"])]
        weights, support = _support(training, replay, cutoff)
        components, component_results = {}, []
        for side in _components(choice["kind"]):
            result = _component_task(checkpoints, kind=choice["kind"], side=side, market=market,
                fold={"fold_id": "final-development-refit", "fit_cutoff": cutoff}, train=training, validation=replay,
                weights=weights, support=support, names=names, baselines=baselines, baseline_id=baseline_id, save_model=True)
            if result["status"] != "complete":
                raise RuntimeError("Final extension refit unsupported or failed")
            import joblib
            components[side] = joblib.load(target / result["component_path"])
            component_results.append(result)
        candidate = report / f"candidate-{market}"
        metadata = {"kind": choice["kind"], "market": market, "strategy": strategy, "weight_ml": weight,
            "qualification": QUALIFICATION, "names": names, "feature_contract_id": data.manifest["feature_contract_id"],
            "dataset_id": data.manifest["dataset_id"], "baseline_contract_id": baseline_id,
            "baseline_contract": baseline_contract, "specification_id": spec_id, "source_hashes": source_hashes(),
            "fit_cutoff": cutoff, "lookback_days": LOOKBACK, "half_life_days": HALF_LIFE,
            "support": support, "component_checkpoints": [r["checkpoint_id"] for r in component_results],
            "training_exclusions": excluded, "interpretation": "later_fixed_refit_of_exploratory_adaptive_proposal"}
        save_bundle(candidate, components, metadata=metadata)
        restored = predict_bundle(candidate, replay, feature_contract_id=data.manifest["feature_contract_id"],
            statistical_rows=[baselines[(r["fixture"]["fixture_id"], market)] for r in replay], baseline_contract_id=baseline_id)
        expected = (1 - weight) * _statistical(replay, baselines, market) + weight * sum(
            (r["predictions"] for r in component_results), np.zeros(len(replay)))
        if not np.array_equal(restored, expected):
            raise RuntimeError("Frozen extension strategy serialization replay mismatch")
        proposal.update(status="frozen_unconfirmed", candidate=str(candidate.relative_to(target)),
                        support=support, serialization_parity=True, final_components=len(components))

    write_jsonl(report / "outer-predictions.jsonl", outer)
    write_jsonl(report / "architecture-oof.jsonl", [r for m in MARKETS for r in raw[m]])
    write_jsonl(report / "adaptive-oof.jsonl", [r for m in MARKETS for r in adaptive[m]])
    write_json(report / "coverage.json", coverage)
    write_json(report / "choices.json", choices)
    write_json(report / "tasks.json", tasks)
    write_json(report / "frozen-proposals.json", frozen)
    print("Extension fits finished; computing paired metrics and uncertainty", flush=True)
    write_json(report / "metrics.json", summarize(outer))
    uncertainty = {}
    for method, baseline in (("frozen_strategy", "league_average"), ("frozen_strategy", "statistical"),
                             ("frozen_strategy", "batch_c_selected_blend")):
        for weeks in (1, 2):
            result = paired_comparison(outer, method, baseline, block_weeks=weeks)
            result["role"] = "six_primary_descriptive" if baseline != "batch_c_selected_blend" else "secondary_descriptive"
            result["caveat"] = "exploratory_after_Batch_C;not_independent_confirmation"
            uncertainty[f"{method}_vs_{baseline}_{weeks}w"] = result
    write_json(report / "uncertainty.json", uncertainty)
    write_json(report / "review.json", {"status": "complete", "qualification": QUALIFICATION,
        "component_development_fits": len(tasks), "architecture_fold_evaluations": 48,
        "resources": checkpoints.statistics(), "frozen_proposals": frozen,
        "outer_prediction_rows": len(outer), "adaptive_oof_rows": sum(map(len, adaptive.values())),
        "publication_enabled": False, "promotion_allowed": False,
        "confirmation_opened": False, "calibration_opened": False, "final_system_test_opened": False,
        "limits": ["Extension motivated by inspected Batch C; outer development is exploratory.",
                  "Statistical comparator is a historical core reconstruction, not full live output.",
                  "Assumed-final historical availability; no authentic betting quote/ROI evaluation.",
                  "Only fixed1460/365 reference features; no eight-year/history or probability conclusion.",
                  "Team means add without asserting independence or doubling unique-fixture sample size.",
                  "Final fixed architecture is a later refit, not the adaptive OOF model sequence."]})
    complete_atomic(report)
    return {"status": "complete", "report_directory": report.name, "specification_id": spec_id,
            "qualification": QUALIFICATION}


def run_extension(*, root, dataset, baselines, batch_c, name, resume=False):
    root, dataset, baselines, batch_c = map(lambda p: Path(p).resolve(), (root, dataset, baselines, batch_c))
    target = _safe_resume(root, name) if resume else new_experiment(root, name)
    with experiment_lock(target), runtime_environment(target):
        environment = environment_report()
        reads = [dataset / p for p in ALLOWED_DATASET_FILES] + _input_paths(baselines) + _old_paths(batch_c)
        with offline_guard(root=root, output=target, readable_files=reads):
            data = DevelopmentDataset(dataset)
            baseline_manifest, baseline_rows = load_baselines(baselines, data)
            old = _load_old(batch_c, data)
            if len(data.folds) != 4 or [f["fold_id"] for f in data.folds] != [f"development-{i:02d}" for i in range(1, 5)]:
                raise ValueError("Extension requires the declared four development folds")
            spec = {"version": VERSION, "dataset_id": data.manifest["dataset_id"],
                    "dataset_artifacts": data.artifact_hashes, "feature_contract_id": data.manifest["feature_contract_id"],
                    "baseline_completion_sha256": sha(baselines / "COMPLETE.json"),
                    "batch_c_completion_sha256": sha(batch_c / "COMPLETE.json"),
                    "protocol_sha256": sha(root / PROTOCOL), "source_hashes": source_hashes(),
                    "environment": {k: environment[k] for k in ("python", "platform", "machine", "dependencies", "dependency_lock_sha256")},
                    "kinds": list(KINDS), "folds": data.folds, "lookback_days": LOOKBACK, "half_life_days": HALF_LIFE,
                    "maximum_fit_attempts": MAX_FITS, "expected_development_components": 72,
                    "qualification": QUALIFICATION, "publication_enabled": False, "promotion_allowed": False,
                    "confirmation_opened": False, "calibration_opened": False, "final_system_test_opened": False}
            spec_id = digest(spec)
            if resume:
                if read_json(target / "specification.json") != spec:
                    raise ValueError("Extension resume source/data/protocol/environment changed")
                if (target / "COMPLETE.json").exists():
                    verify_complete(target)
                    return target
            else:
                write_json(target / "specification.json", spec)
                write_json(target / "environment.json", environment)
            checkpoints = Checkpoints(target / "tasks", spec_id, max_fits=MAX_FITS)
            number = len(list(target.glob("invocation-*.json"))) + 1
            try:
                if (target / "RESULT.json").exists():
                    result = read_json(target / "RESULT.json")
                    if result.get("specification_id") != spec_id or not re.fullmatch(r"reports-\d{4}", result.get("report_directory", "")):
                        raise ValueError("Invalid extension result pointer")
                    verify_complete(target / result["report_directory"])
                    if verify_committed_tasks(target / "tasks", spec_id)["incomplete_tasks"]:
                        raise ValueError("Committed extension result has incomplete tasks")
                    complete_atomic(target)
                    return target
                result = _execute(data, baseline_rows, baseline_manifest["baseline"], old, target, checkpoints, spec_id)
                if source_hashes() != spec["source_hashes"] or sha(root / PROTOCOL) != spec["protocol_sha256"]:
                    raise RuntimeError("Extension code/protocol changed during run")
                write_json(target / f"invocation-{number:04d}.json", {"status": "complete", **checkpoints.statistics()})
                atomic_write_json(target / "RESULT.json", result)
                complete_atomic(target)
            except RunBudgetReached as exc:
                write_json(target / f"invocation-{number:04d}.json", {"status": "paused", "reason": str(exc), **checkpoints.statistics()})
                print(f"Extension stopped safely at budget: {exc}", flush=True)
            except Exception as exc:
                write_json(target / f"invocation-{number:04d}.json", {"status": "failed", "type": type(exc).__name__,
                           "message": str(exc), **checkpoints.statistics()})
                raise
    return target
