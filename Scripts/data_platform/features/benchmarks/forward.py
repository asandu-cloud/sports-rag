"""Batch C chronological development; reserved outcomes never enter this runner."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

import numpy as np

from Scripts.rag_ingest.core.model_features import digest, utc
from .artifacts import (ALLOWED_DATASET_FILES, complete, load_baselines, new_experiment,
                        predict_candidate, predict_strategy, read_json, save_candidate, sha, source_hashes,
                        verify_complete, write_json)
from .checkpoints import (Checkpoints, RunBudgetReached, experiment_lock, atomic_write_json,
                          complete_atomic, verify_committed_tasks)
from .data import DevelopmentDataset, family_gate, recency_weights
from .estimators import DEFAULT_PARAMS, FAMILIES, EstimatorFitError, fit_estimator
from .isolation import offline_guard
from .metrics import numerical_metrics
from .runner import MARKETS, _matrix, environment_report, runtime_environment
from .selection import recipes, choose_recipe, fit_blend, tie_key
from .variants import VARIANTS, PROFILE_VARIANTS, load_variants, transform_features

VERSION = "phase3-forward-development.v1"
QUALIFICATION = "UNCONFIRMED_BATCH_C_DEVELOPMENT"


def _input_paths(directory):
    directory = Path(directory)
    manifest = read_json(directory / "COMPLETE.json")
    paths = [directory / "COMPLETE.json"]
    for name in manifest:
        relative = Path(name)
        item = directory / relative
        if relative.is_absolute() or ".." in relative.parts or item.is_symlink() or not item.resolve().is_relative_to(directory):
            raise ValueError("Unsafe sidecar artifact path")
        paths.append(item)
    return paths


def _safe_resume(root, name):
    # Identical path rules to new_experiment; resumption never adopts a symlink.
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", name):
        raise ValueError("Use a simple experiment name")
    path = root / "Index/prediction_experiments" / name
    if path.resolve() != path or not path.is_dir():
        raise ValueError("Resume requires an existing isolated experiment")
    return path


def specification(data, baseline_path, variant_path, environment):
    return {"version": VERSION, "dataset_id": data.manifest["dataset_id"],
            "dataset_artifacts": data.artifact_hashes,
            "feature_contract_id": data.manifest["feature_contract_id"],
            "baseline_completion_sha256": sha(baseline_path / "COMPLETE.json"),
            "variant_completion_sha256": sha(variant_path / "COMPLETE.json"),
            "source_hashes": source_hashes(), "dependency_lock_sha256": environment["dependency_lock_sha256"],
            "environment": {key: environment[key] for key in ("python", "platform", "machine", "dependencies")},
            "protocol_sha256": sha(Path(__file__).resolve().parents[4] /
                                   "docs/phase3-forward-development-protocol-2026-09-26.md"),
            "recipes": recipes(), "folds": data.folds,
            "outer_fold_ids": [fold["fold_id"] for fold in data.folds[2:]],
            "seed": 42, "cpu_threads": 2, "primary_metric": "equal_fixture_rmse",
            "minimum_inner_folds": 2, "warmup": "fold1_fixed_defaults_all365;fold2_one_earlier_block",
            "blend": "analytic_convex_sse_on_earlier_selection_honest_adaptive_oof",
            "variants": list(VARIANTS), "feature_comparison_maximum_fits": 84,
            "feature_variant_selection": "diagnostic_only_reference_policy_retained",
            "bootstrap": {"resamples": 2000, "seed": 42, "block_weeks": [1, 2],
                          "familywise_comparisons": 6},
            "budgets": {"max_seconds_per_invocation": 28800, "max_fits_per_experiment": 2500},
            "qualification": QUALIFICATION, "publication_enabled": False, "promotion_allowed": False,
            "confirmation_opened": False, "calibration_opened": False, "final_system_test_opened": False,
            "historical_roi": "unavailable_no_authentic_timestamped_quote_dataset"}


def _support(train, validation, cutoff, half_life, base=None):
    weights, weighting = recency_weights(train, cutoff, half_life)
    support = {**(base or {}), **weighting, "fit_cutoff": cutoff,
               "training_unique_fixtures": len(train), "validation_unique_fixtures": len(validation),
               "training_membership_sha256": digest([r["fixture"]["fixture_id"] for r in train]),
               "validation_membership_sha256": digest([r["fixture"]["fixture_id"] for r in validation])}
    return weights, support


def _fit_task(checkpoints, *, market, fold, recipe, train, validation, weights,
              support, names, view, bundle_metadata=None):
    gate = family_gate(support, recipe["family"])
    if not gate["qualified"]:
        return {"status": "insufficient_support", "gate": gate, "support": support,
                "recipe_id": recipe["recipe_id"], "fold_id": fold["fold_id"]}
    y = np.asarray([row["target"] for row in train], dtype=float)
    contract = {"market": market, "fold_id": fold["fold_id"], "fit_cutoff": fold["fit_cutoff"],
                "family": recipe["family"], "params": recipe["params"], "view": view,
                "names": list(names), "training_membership_sha256": support["training_membership_sha256"],
                "validation_membership_sha256": support["validation_membership_sha256"],
                "training_labels_sha256": digest(y.tolist()), "weights_sha256": digest(weights.tolist()),
                "validation_snapshot_sha256": digest([r["snapshot_id"] for r in validation]),
                "bundle_metadata": bundle_metadata}
    # Snapshot IDs bind values to immutable validated sidecar contracts. No
    # current profiles, archives or database are available to this operation.
    def compute(attempt):
        try:
            model = fit_estimator(recipe["family"], _matrix(train), y, weights, names, market,
                                  params=recipe["params"])
            prediction = model.predict_with_diagnostics(_matrix(validation), names=names)
        except EstimatorFitError as exc:
            return {"status": "failed", "failure": {"type": type(exc).__name__, "message": str(exc)},
                    "predictions": np.asarray([], dtype=float), "raw_predictions": np.asarray([], dtype=float),
                    "training_max_label_available_at": max(r["label_available_at"] for r in train)}
        result = {"status": "complete", "predictions": prediction["values"],
                  "raw_predictions": prediction["raw_values"], "clipped_count": prediction["clipped_count"],
                  "training_max_label_available_at": max(r["label_available_at"] for r in train),
                  "estimator": model.metadata}
        if bundle_metadata is not None:
            candidate = attempt / "candidate"
            save_candidate(candidate, model, metadata={**bundle_metadata, "market": market,
                           "names": list(names), "qualification": QUALIFICATION,
                           "training_fixture_ids": [r["fixture"]["fixture_id"] for r in train],
                           "training_snapshot_ids": [r["snapshot_id"] for r in train],
                           "training_labels_sha256": contract["training_labels_sha256"],
                           "training_weights_sha256": contract["weights_sha256"], "support": support})
            restored = predict_candidate(candidate, validation,
                                         feature_contract_id=bundle_metadata["feature_contract_id"])
            if not np.array_equal(restored, prediction["values"]):
                raise RuntimeError("Final development bundle failed exact serialization replay")
            result["candidate"] = str(candidate.relative_to(checkpoints.path.parent))
            result["serialization_parity"] = True
        return result
    result = checkpoints.get(contract, compute)
    if result["status"] == "complete" and len(result["predictions"]) != len(validation):
        raise ValueError("Checkpoint prediction cohort length mismatch")
    return {**result, "support": support, "gate": gate, "recipe_id": recipe["recipe_id"],
            "fold_id": fold["fold_id"], "fit_cutoff": fold["fit_cutoff"]}


def selection_records(grid, validations, *, market, prior_folds, receiving_cutoff):
    """Rescore prior predictions using only labels available at selection time."""
    records = []
    for fold in prior_folds:
        fold_id = fold["fold_id"]
        rows = validations[(market, fold_id)]
        mask = np.asarray([utc(r["label_available_at"]) < utc(receiving_cutoff) for r in rows])
        available = [r for r, keep in zip(rows, mask) if keep]
        targets = np.asarray([r["target"] for r in available], dtype=float)
        for recipe in recipes():
            result = grid[(market, fold_id, recipe["recipe_id"])]
            record = {"recipe_id": recipe["recipe_id"], "fold_id": fold_id, "status": result["status"],
                      "fit_cutoff": fold["fit_cutoff"], "n": len(available),
                      "validation_membership_sha256": digest([r["fixture"]["fixture_id"] for r in available]),
                      "targets_sha256": digest(targets.tolist()),
                      "max_label_available_at": max((r["label_available_at"] for r in available), default=None),
                      "support": {**result["support"], "validation_unique_fixtures": len(available)}}
            if result.get("failure"):
                record["failure"] = result["failure"]
            if result["status"] == "complete":
                record.update(rmse=numerical_metrics(targets, result["predictions"][mask])["rmse"],
                              training_max_label_available_at=result["training_max_label_available_at"])
            records.append(record)
    return records


def seed_choice(fold, family=None):
    family = family or "ridge"
    recipe = next(r for r in recipes() if r["family"] == family and r["params"] == DEFAULT_PARAMS[family]
                  and r["lookback_days"] is None and r["half_life_days"] == 365)
    return {"status": "selected", "recipe": recipe, "recipe_id": recipe["recipe_id"],
            "selection_fold_ids": [], "selection_cutoff": fold["fit_cutoff"],
            "policy": "fixed_predeclared_seed_no_tuning"}


def choose_before(grid, validations, *, market, folds, index, family=None):
    fold = folds[index]
    if index == 0:
        return seed_choice(fold, family)
    prior = folds[:index]
    records = selection_records(grid, validations, market=market, prior_folds=prior,
                                receiving_cutoff=fold["fit_cutoff"])
    return choose_recipe(records, [f["fold_id"] for f in prior], family=family,
                         min_folds=min(2, index), fit_cutoff=fold["fit_cutoff"])


def earlier_blend(oof, *, cutoff, minimum_folds=2):
    rows = [r for r in oof if utc(r["label_available_at"]) < utc(cutoff)]
    if len({r["fold_id"] for r in rows}) < minimum_folds:
        return {"weight_ml": 0.0, "status": "warmup_insufficient_oof", "n": len(rows),
                "fit_cutoff": cutoff, "oof_fixture_ids": []}
    for row in rows:
        if (utc(row["training_max_label_available_at"]) >= utc(row["fit_cutoff"])
                or utc(row["fit_cutoff"]) > utc(row["kickoff"])
                or utc(row["selection_cutoff"]) != utc(row["fit_cutoff"])
                or any(utc(end) > utc(row["fit_cutoff"]) for end in row["selection_validation_ends"])):
            raise ValueError("Blend OOF prediction was not selected/trained strictly forward")
    ids = [r["fixture_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate fixture in blend training")
    fitted = fit_blend([r["target"] for r in rows], [r["statistical"] for r in rows],
                       [r["ml"] for r in rows])
    return {**fitted, "status": "fitted", "fit_cutoff": cutoff,
            "oof_fixture_ids": ids, "oof_prediction_sha256": digest(rows),
            "oof_fold_ids": sorted({r["fold_id"] for r in rows}),
            "interpretation": "adaptive_selection_honest_oof;final_recipe_is_a_later_refit"}


def prediction_rows(rows, values, *, market, fold, method, names, recipe_id=None, weight_ml=None):
    numeric = [i for i, name in enumerate(names) if not name.endswith("__missing")]
    pos = {name: i for i, name in enumerate(names)}
    result = []
    for row, value in zip(rows, values, strict=True):
        fixture = row["fixture"]
        count = min(row["values"][pos[side + "_current_matches"]] or 0 for side in ("home", "away"))
        missing = sum(row["values"][i] is None for i in numeric) / len(numeric)
        support = min(row["support"][market][s]["count"] for s in ("home", "away"))
        result.append({"fixture_id": fixture["fixture_id"], "kickoff": fixture["kickoff"],
                       "competition": fixture["competition"], "season": fixture["season"],
                       "market": market, "fold_id": fold["fold_id"], "method": method,
                       "target": row["target"], "prediction": float(value), "snapshot_id": row["snapshot_id"],
                       "source_class": row["source_class"], "round_group": row["round_group"],
                       "season_stage": "0-7" if count < 8 else "8-19" if count < 20 else "20+",
                       "support_band": "5-9" if support < 10 else "10+",
                       "missingness_band": "0" if missing == 0 else "0-10%" if missing <= .1 else
                                           "10-25%" if missing <= .25 else ">25%",
                       "forecast_stage": row["forecast_stage"], "recipe_id": recipe_id,
                       "weight_ml": weight_ml, "qualification": QUALIFICATION})
    return result


def write_jsonl(path, rows):
    with Path(path).open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def run_development(*, root, dataset, baselines, variants, name, resume=False):
    """Run/resume the fixed Batch C procedure; never open reserved label stores."""
    root, dataset = Path(root).resolve(), Path(dataset).resolve()
    baselines, variants = Path(baselines).resolve(), Path(variants).resolve()
    target = _safe_resume(root, name) if resume else new_experiment(root, name)
    with experiment_lock(target), runtime_environment(target):
        environment = environment_report()
        reads = [dataset / p for p in ALLOWED_DATASET_FILES] + _input_paths(baselines) + _input_paths(variants)
        with offline_guard(root=root, output=target, readable_files=reads):
            data = DevelopmentDataset(dataset)
            baseline_manifest, baseline_rows = load_baselines(baselines, data)
            variant_manifest, variant_rows = load_variants(variants, data)
            if set(variant_manifest["variants"]) != set(PROFILE_VARIANTS):
                raise ValueError("Batch C requires all three frozen rebuilt profile variants")
            if len(data.folds) != 4:
                raise ValueError("This frozen protocol requires exactly four development blocks")
            spec = specification(data, baselines, variants, environment)
            spec_id = digest(spec)
            if resume:
                if read_json(target / "specification.json") != spec:
                    raise ValueError("Resume specification/source/data/environment changed")
                if (target / "COMPLETE.json").exists():
                    verify_complete(target)
                    return target
            else:
                write_json(target / "specification.json", spec)
                write_json(target / "environment.json", environment)
            checkpoints = Checkpoints(target / "tasks", spec_id)
            invocation = len(list(target.glob("invocation-*.json"))) + 1
            try:
                if (target / "RESULT.json").exists():
                    result = read_json(target / "RESULT.json")
                    report_name = result.get("report_directory", "")
                    if (not __import__("re").fullmatch(r"reports-\d{4}", report_name)
                            or result.get("specification_id") != spec_id):
                        raise ValueError("Invalid committed result pointer")
                    verify_complete(target / report_name)
                    committed = verify_committed_tasks(target / "tasks", spec_id)
                    if committed["incomplete_tasks"]:
                        raise ValueError("Committed result has unfinished task checkpoints")
                    complete_atomic(target)
                    return target
                result = _execute(data, baseline_rows, baseline_manifest["baseline"], variant_rows,
                                  variant_manifest, target, checkpoints, spec_id)
                if source_hashes() != spec["source_hashes"]:
                    raise RuntimeError("Source changed during development; completion withheld")
                write_json(target / f"invocation-{invocation:04d}.json",
                           {"status": "complete", **checkpoints.statistics()})
                # All report generation happens in a fresh attempt. The pointer
                # is published only after each report and bundle is checksummed.
                atomic_write_json(target / "RESULT.json", result)
                complete_atomic(target)
            except RunBudgetReached as exc:
                write_json(target / f"invocation-{invocation:04d}.json",
                           {"status": "paused", "reason": str(exc), **checkpoints.statistics()})
                print(f"Development paused safely: {exc}", flush=True)
            except Exception as exc:
                write_json(target / f"invocation-{invocation:04d}.json",
                           {"status": "failed", "type": type(exc).__name__, "message": str(exc),
                            **checkpoints.statistics()})
                raise
    return target


def _execute(data, baselines, baseline_contract, variants, variant_manifest, target, checkpoints, spec_id):
    from .evaluation import summarize, paired_comparison

    folds, names = data.folds, data.schema["names"]
    grid, validations, choices, outer, oof = {}, {}, {}, [], defaultdict(list)
    score_records = []
    for market in MARKETS:
        choices[market] = {}
        for index, fold in enumerate(folds):
            fid = fold["fold_id"]
            selections = {}
            for recipe_index, recipe in enumerate(recipes()):
                pair = (recipe["lookback_days"], recipe["half_life_days"])
                if pair not in selections:
                    selections[pair] = data.select_fold(market, fid, lookback_days=pair[0], half_life_days=pair[1])
                train, validation, weights, support = selections[pair]
                validations[(market, fid)] = validation
                result = _fit_task(checkpoints, market=market, fold=fold, recipe=recipe,
                                   train=train, validation=validation, weights=weights, support=support,
                                   names=names, view=data.manifest["feature_contract_id"])
                grid[(market, fid, recipe["recipe_id"])] = result
                if recipe_index % 26 == 0:
                    print(f"Forward {market}/{fid}: recipe {recipe_index + 1}/156; "
                          f"{checkpoints.created} new fits, {checkpoints.reused} reused", flush=True)
            fold_choices = {}
            for family in (*FAMILIES, "selected_ml"):
                choice = choose_before(grid, validations, market=market, folds=folds, index=index,
                                       family=None if family == "selected_ml" else family)
                fold_choices[family] = choice
                if choice["status"] != "selected":
                    raise RuntimeError(f"No supported forward recipe for {market}/{fid}/{family}")
                fitted = grid[(market, fid, choice["recipe_id"])]
                if fitted["status"] != "complete":
                    raise RuntimeError(f"Selected forward recipe is unsupported for {market}/{fid}/{family}")
                if index >= 2:
                    outer.extend(prediction_rows(validation, fitted["predictions"], market=market, fold=fold,
                                                 method=family, names=names, recipe_id=choice["recipe_id"]))
            choices[market][fid] = fold_choices
            selection = fold_choices["selected_ml"]
            selected = grid[(market, fid, selection["recipe_id"])]
            statistical = np.asarray([baselines[(r["fixture"]["fixture_id"], market)]["statistical"]
                                      for r in validation])
            blend = earlier_blend(oof[market], cutoff=fold["fit_cutoff"])
            fold_choices["blend"] = blend
            if index >= 2:
                for method in ("league_average", "statistical"):
                    values = [baselines[(r["fixture"]["fixture_id"], market)][method] for r in validation]
                    outer.extend(prediction_rows(validation, values, market=market, fold=fold,
                                                 method=method, names=names))
                w = blend["weight_ml"]
                outer.extend(prediction_rows(validation, (1 - w) * statistical + w * selected["predictions"],
                                             market=market, fold=fold, method="selected_blend", names=names,
                                             recipe_id=selection["recipe_id"], weight_ml=w))
            selected_folds = set(selection["selection_fold_ids"])
            ends = [f["validation_end"] for f in folds if f["fold_id"] in selected_folds]
            for row, stat, ml in zip(validation, statistical, selected["predictions"], strict=True):
                oof[market].append({"fixture_id": row["fixture"]["fixture_id"], "market": market,
                    "fold_id": fid, "recipe_id": selection["recipe_id"], "kickoff": row["fixture"]["kickoff"],
                    "label_available_at": row["label_available_at"], "fit_cutoff": fold["fit_cutoff"],
                    "training_max_label_available_at": selected["training_max_label_available_at"],
                    "selection_cutoff": selection["selection_cutoff"], "selection_validation_ends": ends,
                    "selection_records_sha256": selection.get("selection_records_sha256"),
                    "selection_max_label_available_at": selection.get("selection_max_label_available_at"),
                    "selection_choice_id": selection.get("choice_id"),
                    "selection_fold_ids": selection["selection_fold_ids"], "target": row["target"],
                    "statistical": float(stat), "ml": float(ml)})
        score_records.extend(selection_records(grid, validations, market=market, prior_folds=folds,
                             receiving_cutoff=data.splits["boundaries"]["phase3_confirmation_start"]))
    print("Main grid complete; building paired history and feature comparisons", flush=True)
    history = _history_comparisons(grid, validations, choices, folds, names)
    ablations, coverage = _feature_comparisons(data, variants, choices, checkpoints)
    frozen = _freeze_candidates(data, grid, validations, choices, outer, oof, checkpoints, spec_id,
                                baselines, baseline_contract)
    selected_rows = []
    for market in MARKETS:
        strategy = frozen[market].get("strategy", "statistical")
        selected_rows.extend({**row, "method": "frozen_strategy"} for row in outer
                             if row["market"] == market and row["method"] == strategy)
    outer.extend(selected_rows)
    # Reports are separate attempts so an interruption during expensive metrics
    # does not overwrite earlier partial evidence on continuation.
    number = len(list(target.glob("reports-*"))) + 1
    report_dir = target / f"reports-{number:04d}"
    report_dir.mkdir()
    write_jsonl(report_dir / "outer-predictions.jsonl", outer)
    write_jsonl(report_dir / "adaptive-oof-predictions.jsonl", [r for m in MARKETS for r in oof[m]])
    write_json(report_dir / "recipe-scores.json", score_records)
    write_json(report_dir / "chronological-choices.json", choices)
    write_json(report_dir / "frozen-recipes.json", frozen)
    write_json(report_dir / "history-comparisons.json", history)
    write_json(report_dir / "feature-coverage.json", coverage)
    write_jsonl(report_dir / "feature-predictions.jsonl", [r for rows in ablations.values() for r in rows])
    print("Computing declared metrics and grouped uncertainty", flush=True)
    write_json(report_dir / "metrics.json", summarize(outer))
    comparisons = {}
    for method in ("frozen_strategy", "selected_ml", "selected_blend"):
        for baseline in ("league_average", "statistical"):
            for block in (1, 2):
                print(f"Uncertainty {method}/{baseline}: {block}-week blocks", flush=True)
                comparison = paired_comparison(outer, method, baseline, block_weeks=block)
                comparison["role"] = "six_primary_comparisons" if method == "frozen_strategy" else "secondary_descriptive"
                comparison["selection_caveat"] = "development intervals after selection; no confirmation qualification"
                comparisons[f"{method}_vs_{baseline}_{block}w"] = comparison
    write_json(report_dir / "uncertainty.json", comparisons)
    feature_reports = {}
    for variant, rows in ablations.items():
        if rows:
            feature_reports[variant] = {"metrics": summarize(rows),
                "paired": paired_comparison(rows, variant, "paired_reference"),
                "two_week": paired_comparison(rows, variant, "paired_reference", block_weeks=2)}
        else:
            feature_reports[variant] = {"status": "insufficient_support"}
    write_json(report_dir / "feature-comparisons.json", feature_reports)
    status_counts = defaultdict(int)
    for r in grid.values():
        status_counts[r["status"]] += 1
    write_json(report_dir / "review.json", {
        "version": VERSION, "qualification": QUALIFICATION, "status": "complete",
        "main_grid_slots": len(grid), "main_grid_statuses": dict(status_counts),
        "outer_prediction_rows": len(outer), "adaptive_oof_rows": sum(map(len, oof.values())),
        "resources": checkpoints.statistics(), "frozen_candidates": frozen,
        "limitations": [
            "Assumed-final retrospective input availability, not immutable original forecast snapshots.",
            "Only two outer development blocks; selection-aware confirmation remains unopened.",
            "Four-year and all-history windows may be identical at these early cutoffs.",
            "Longer profile variants change only early-season priors and are conditional diagnostics.",
            "Only one reconstructed forecast stage; no prospective lineup/amendment acceptance.",
            "Common Poisson diagnostics are uncalibrated; no historical betting-performance claims.",
            "Final recipe refit differs from adaptive earlier OOF model selection; confirmation is required."],
        "publication_enabled": False, "promotion_allowed": False,
        "confirmation_opened": False, "calibration_opened": False, "final_system_test_opened": False})
    complete_atomic(report_dir)
    return {"status": "complete", "report_directory": report_dir.name,
            "specification_id": spec_id, "qualification": QUALIFICATION}


def _history_comparisons(grid, validations, choices, folds, names):
    results = {}
    catalog = recipes()
    for market in MARKETS:
        rows_by_method, fit_evidence = defaultdict(list), []
        for fold in folds[2:]:
            fid = fold["fold_id"]
            selected = choices[market][fid]["selected_ml"]["recipe"]
            validation = validations[(market, fid)]
            for lookback in (730, 1460, None):
                for decay_label, half_life in (("no_decay", None), ("selected_decay", selected["half_life_days"])):
                    recipe = next(r for r in catalog if r["family"] == selected["family"]
                                  and r["params"] == selected["params"] and r["lookback_days"] == lookback
                                  and r["half_life_days"] == half_life)
                    fitted = grid[(market, fid, recipe["recipe_id"])]
                    method = f"history_{lookback or 'all'}_{decay_label}"
                    evidence = {"fold_id": fid, "method": method, "recipe": recipe,
                                "status": fitted["status"], "support": fitted["support"],
                                "checkpoint_id": fitted.get("checkpoint_id")}
                    fit_evidence.append(evidence)
                    if fitted["status"] == "complete":
                        rows_by_method[method].extend(prediction_rows(
                            validation, fitted["predictions"], market=market, fold=fold,
                            method=method, names=names, recipe_id=recipe["recipe_id"]))
        methods = {}
        for method, rows in rows_by_method.items():
            methods[method] = {"coverage_specific_metrics": numerical_metrics([r["target"] for r in rows],
                                                           [r["prediction"] for r in rows]),
                               "folds": sorted({r["fold_id"] for r in rows}),
                               "membership_sha256": digest([r["fixture_id"] for r in rows])}
        identical = []
        for fold in folds[2:]:
            items = [r for r in fit_evidence if r["fold_id"] == fold["fold_id"] and r["status"] == "complete"]
            for i, first in enumerate(items):
                for second in items[i + 1:]:
                    if first["checkpoint_id"] == second["checkpoint_id"]:
                        identical.append({"fold_id": fold["fold_id"],
                                          "methods": [first["method"], second["method"]]})
        expected_methods = {f"history_{window or 'all'}_{decay}" for window in (730, 1460, None)
                            for decay in ("no_decay", "selected_decay")}
        mappings = {method: {r["fixture_id"]: r for r in rows_by_method.get(method, [])}
                    for method in expected_methods}
        common_ids = set.intersection(*(set(rows) for rows in mappings.values()))
        paired = {"status": "complete" if common_ids else "insufficient_common_coverage",
                  "n": len(common_ids), "membership_sha256": digest(sorted(common_ids)), "methods": {}}
        for method, mapping in sorted(mappings.items()):
            rows = [mapping[fid] for fid in sorted(common_ids)]
            if rows:
                paired["methods"][method] = numerical_metrics(
                    [r["target"] for r in rows], [r["prediction"] for r in rows])
                paired["folds"] = sorted({r["fold_id"] for r in rows})
        results[market] = {"methods": methods, "paired_common_cohort": paired, "fits": fit_evidence,
                           "identical_fits": identical,
                           "interpretation": "paired metrics use only common fixtures; coverage-specific scores are separate"}
    return results


def _variant_rows(rows, market, variant, variants, names):
    if variant in PROFILE_VARIANTS:
        kept, changed = [], []
        for row in rows:
            item = variants[(row["fixture"]["fixture_id"], variant)]
            common = item["common_eligible"][market]
            if not common:
                continue
            kept.append(row)
            changed.append({**row, "values": item["values"], "snapshot_id": item["snapshot_id"],
                            "feature_contract_id": item["feature_contract_id"],
                            "support": item["support"]})
        if not changed:
            return kept, changed, names, {"variant": variant, "status": "no_common_rows"}
        return kept, changed, names, {"variant": variant,
                                      "feature_contract_id": changed[0]["feature_contract_id"]}
    changed, reduced, contract = transform_features(rows, names, variant)
    return list(rows), changed, reduced, contract


def _feature_comparisons(data, variants, choices, checkpoints):
    predictions, coverage = {name: [] for name in VARIANTS}, []
    for market in MARKETS:
        for fold in data.folds[2:]:
            selected = choices[market][fold["fold_id"]]["selected_ml"]["recipe"]
            train, validation, _, original_support = data.select_fold(
                market, fold["fold_id"], lookback_days=selected["lookback_days"],
                half_life_days=selected["half_life_days"])
            for variant in VARIANTS:
                ref_train, variant_train, variant_names, contract = _variant_rows(
                    train, market, variant, variants, data.schema["names"])
                ref_validation, variant_validation, vnames, vcontract = _variant_rows(
                    validation, market, variant, variants, data.schema["names"])
                if variant_train and variant_validation and (variant_names != vnames or contract != vcontract):
                    raise ValueError("Training/inference feature variant contracts differ")
                weights, support = _support(ref_train, ref_validation, fold["fit_cutoff"],
                                            selected["half_life_days"], original_support)
                item = {"market": market, "fold_id": fold["fold_id"], "variant": variant,
                        "recipe": selected, "original_training": len(train),
                        "original_validation": len(validation), "common_training": len(ref_train),
                        "common_validation": len(ref_validation), "support": support,
                        "variant_contract": contract, "status": "insufficient_support"}
                gate = family_gate(support, selected["family"])
                item["gate"] = gate
                if gate["qualified"]:
                    reference = _fit_task(checkpoints, market=market, fold=fold, recipe=selected,
                        train=ref_train, validation=ref_validation, weights=weights, support=support,
                        names=data.schema["names"], view=data.manifest["feature_contract_id"])
                    changed = _fit_task(checkpoints, market=market, fold=fold, recipe=selected,
                        train=variant_train, validation=variant_validation, weights=weights, support=support,
                        names=variant_names, view=contract)
                    # Metadata/slicing uses the same reference rows for both
                    # methods: a changed support feature cannot move a fixture
                    # into a different slice and make the paired report invalid.
                    for method, fitted in (("paired_reference", reference), (variant, changed)):
                        predictions[variant].extend(prediction_rows(
                            ref_validation, fitted["predictions"], market=market, fold=fold,
                            method=method, names=data.schema["names"], recipe_id=selected["recipe_id"]))
                    item.update(status="complete", reference_checkpoint=reference["checkpoint_id"],
                                variant_checkpoint=changed["checkpoint_id"],
                                unchanged_training_vectors=sum(a["values"] == b["values"]
                                    for a, b in zip(ref_train, variant_train)) if variant in PROFILE_VARIANTS else None,
                                unchanged_validation_vectors=sum(a["values"] == b["values"]
                                    for a, b in zip(ref_validation, variant_validation)) if variant in PROFILE_VARIANTS else None)
                coverage.append(item)
                print(f"Feature {market}/{fold['fold_id']}/{variant}: {item['status']}; "
                      f"{len(ref_train)} train, {len(ref_validation)} validation", flush=True)
    return predictions, coverage


def final_training_rows(data, market, recipe):
    cutoff = data.splits["boundaries"]["phase3_confirmation_start"]
    earliest = utc(cutoff) - timedelta(days=recipe["lookback_days"]) if recipe["lookback_days"] else None
    rows = [{**row, "target": row["labels"][market]} for row in data.rows
            if row["market_eligibility"][market]["eligible"]
            and utc(row["label_available_at"]) < utc(cutoff)
            and (earliest is None or utc(row["fixture"]["kickoff"]) >= earliest)]
    return rows


def _freeze_candidates(data, grid, validations, choices, outer, oof, checkpoints, spec_id,
                       baselines, baseline_contract):
    result = {}
    cutoff = data.splits["boundaries"]["phase3_confirmation_start"]
    for market in MARKETS:
        scores = selection_records(grid, validations, market=market, prior_folds=data.folds,
                                   receiving_cutoff=cutoff)
        chosen = choose_recipe(scores, [f["fold_id"] for f in data.folds], fit_cutoff=cutoff)
        if chosen["status"] != "selected":
            result[market] = {"status": "retain_statistical", "reason": "insufficient_development_evidence",
                              "selection": chosen, "weight_ml": 0.0}
            continue
        blend = earlier_blend(oof[market], cutoff=cutoff)
        comparisons = []
        for method, weight_order in (("statistical", 0), ("selected_blend", 1), ("selected_ml", 2)):
            rows = [row for row in outer if row["market"] == market and row["method"] == method]
            metrics = numerical_metrics([r["target"] for r in rows], [r["prediction"] for r in rows])
            comparisons.append({"method": method, "metrics": metrics, "tie_order": weight_order,
                                "membership_sha256": digest([r["fixture_id"] for r in rows])})
        if len({c["membership_sha256"] for c in comparisons}) != 1:
            raise ValueError("Final strategy comparison has mismatched cohorts")
        comparisons.sort(key=lambda item: (item["metrics"]["rmse"], item["tie_order"]))
        method = comparisons[0]["method"]
        weight = 0.0 if method == "statistical" else 1.0 if method == "selected_ml" else blend["weight_ml"]
        recipe = chosen["recipe"]
        training = final_training_rows(data, market, recipe)
        # These development features are used only for serialization parity.
        # No final-refit in-sample predictions are presented as OOF metrics.
        replay = validations[(market, data.folds[-1]["fold_id"])]
        weights, support = _support(training, replay, cutoff, recipe["half_life_days"])
        fold = {"fold_id": "final-development-refit", "fit_cutoff": cutoff}
        metadata = {"dataset_id": data.manifest["dataset_id"],
                    "feature_contract_id": data.manifest["feature_contract_id"],
                    "feature_schema": data.schema, "specification_id": spec_id,
                    "source_hashes": source_hashes(), "fit_cutoff": cutoff,
                    "fold_id": fold["fold_id"], "recipe": recipe,
                    "lookback_days": recipe["lookback_days"], "half_life_days": recipe["half_life_days"],
                    "availability": data.manifest["availability"],
                    "supported_scope": "development_only_no_league_qualification",
                    "strategy": method, "weight_ml": weight, "blend_provenance": blend,
                    "baseline_contract_id": digest(baseline_contract), "baseline_contract": baseline_contract,
                    "inference_api": "artifacts.predict_strategy; predict_candidate is estimator-only",
                    "selection_provenance": chosen, "supported_markets": [market],
                    "confirmation_opened": False, "publication_enabled": False, "promotion_allowed": False}
        fitted = _fit_task(checkpoints, market=market, fold=fold, recipe=recipe,
                          train=training, validation=replay, weights=weights, support=support,
                          names=data.schema["names"], view=data.manifest["feature_contract_id"],
                          bundle_metadata=metadata)
        if fitted["status"] != "complete":
            raise RuntimeError("Final development refit lacks required training support")
        baseline_replay = [baselines[(r["fixture"]["fixture_id"], market)] for r in replay]
        strategy_predictions = predict_strategy(checkpoints.path.parent / fitted["candidate"], replay,
            feature_contract_id=data.manifest["feature_contract_id"], statistical_rows=baseline_replay,
            baseline_contract_id=digest(baseline_contract))
        expected = (1 - weight) * np.asarray([r["statistical"] for r in baseline_replay]) + weight * fitted["predictions"]
        if not np.array_equal(strategy_predictions, expected):
            raise RuntimeError("Frozen system strategy replay differs from its blend contract")
        result[market] = {"status": "frozen_for_confirmation", "strategy": method,
                          "weight_ml": weight, "recipe": recipe, "candidate": fitted["candidate"],
                          "selection": chosen, "blend": blend, "outer_strategy_comparison": comparisons,
                          "serialization_parity": True, "qualification": QUALIFICATION,
                          "strategy_replay_parity": True, "baseline_contract_id": digest(baseline_contract),
                          "fit_cutoff": cutoff, "support": support, "publication_enabled": False,
                          "promotion_allowed": False}
        print(f"Frozen development recipe for {market}; confirmation remains unopened", flush=True)
    return result
