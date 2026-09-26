"""Batch D: frozen, one-time numerical confirmation with no fitting or promotion."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

from Scripts.rag_ingest.core.model_features import digest, utc
from . import confirmation_data as cd
from .artifacts import (ROOT, ALLOWED_DATASET_FILES, new_experiment, read_json, sha,
                       source_hashes, verify_complete, predict_strategy)
from .baselines import baseline_metadata
from .checkpoints import atomic_write_json, complete_atomic, experiment_lock
from .data import DevelopmentDataset, MARKETS, COMPETITIONS, family_gate
from .evaluation import paired_comparison, summarize
from .extension_bundle import predict_bundle
from .forward import prediction_rows, write_jsonl
from .isolation import offline_guard
from .runner import runtime_environment, environment_report

VERSION = "phase3-confirmation.v1"
PROTOCOL = "docs/phase3-confirmation-protocol-2026-09-27.md"
RULES = {
    "primary": "equal_fixture_rmse", "minimum_relative_improvement": .01,
    "minimum_confirmation_fixtures": 1000, "minimum_calendar_weeks": 20,
    "minimum_slice_fixtures": 200, "minimum_slice_match_dates": 20,
    "slice_maximum_deterioration_upper_95": .05,
    "familywise_confidence": .95, "primary_comparisons": 6,
    "resamples": 20000, "seed": 42, "primary_block_weeks": 1,
    "sensitivity_block_weeks": 2, "sensitivity_role": "reported_robustness_not_an_additional_gate",
    "baseline_methods": ["statistical", "league_average"],
    "cohort": "all_frozen_eligible_confirmation_fixtures_no_silent_intersections",
    "selection": "lower_development_outer_rmse_of_two_frozen_proposals_ties_original_batch_c",
    "fitting": "none_models_preprocessors_blends_fixed_at_confirmation_start",
    "feature_updates": "strict_prematch_assumed_final_kickoff_plus_3h_current_and_previous_season",
    "betting_metrics": "unavailable_no_authentic_historical_quote_cohort",
}


def _safe(path):
    path = Path(path).absolute()
    if path.resolve() != path:
        raise ValueError("Symlinks are not allowed in confirmation inputs")
    return path


def _children(directory):
    """Read completion names only; do not follow arbitrary paths from a manifest."""
    directory = _safe(directory)
    manifest = read_json(directory / "COMPLETE.json")
    files = [directory / "COMPLETE.json"]
    for name in manifest:
        DevelopmentDataset._validate_name(name)
        path = _safe(directory / name)
        if not path.is_relative_to(directory):
            raise ValueError("Artifact escapes input directory")
        files.append(path)
    return files


def _development_proposals(path, data, *, extension):
    verify_complete(path, required={"RESULT.json", "specification.json"})
    result = read_json(path / "RESULT.json")
    report = result.get("report_directory", "")
    if result.get("status") != "complete" or not re.fullmatch(r"reports-\d{4}", report):
        raise ValueError("Development experiment is incomplete")
    frozen_name = "frozen-proposals.json" if extension else "frozen-recipes.json"
    frozen = read_json(path / report / frozen_name)
    metrics = read_json(path / report / "metrics.json")
    proposals = {}
    for market in MARKETS:
        item = frozen[market]
        if not item.get("serialization_parity") or item.get("strategy") not in {"selected_ml", "selected_blend"}:
            raise ValueError("Expected a frozen, replayed nonzero development proposal")
        relative = item["candidate"]
        DevelopmentDataset._validate_name(relative)
        bundle = _safe(path / relative)
        verify_complete(bundle, required={"manifest.json"})
        manifest = read_json(bundle / "manifest.json")
        if (manifest.get("dataset_id") != data.manifest["dataset_id"]
                or manifest.get("feature_contract_id") != data.manifest["feature_contract_id"]
                or utc(manifest["fit_cutoff"]) != data.boundaries["phase3_confirmation_start"]
                or manifest.get("market") != market or manifest.get("weight_ml") != item["weight_ml"]
                or manifest.get("strategy") != item["strategy"]
                or manifest.get("publication_enabled") is not False or manifest.get("promotion_allowed") is not False):
            raise ValueError("Development candidate contract differs from dataset/frozen proposal")
        family = manifest.get("family") or {"team_poisson": "poisson", "team_catboost": "catboost",
            "residual_ridge": "ridge", "offset_xgboost": "xgboost"}.get(manifest.get("kind"))
        if not family_gate(manifest["support"], family)["qualified"]:
            raise ValueError("Frozen candidate lacks training/development support")
        score = metrics["by_market"][market]["methods"][item["strategy"]]["overall"]
        proposals[market] = {"origin": "extension" if extension else "batch_c", "source_bundle": str(bundle),
            "source_complete_sha256": sha(bundle / "COMPLETE.json"), "weight_ml": item["weight_ml"],
            "strategy": item["strategy"], "kind": manifest.get("kind", manifest.get("family")),
            "rmse": score["rmse"], "outer_support": score["support"],
            "training_support": manifest["support"], "baseline_contract_id": manifest["baseline_contract_id"]}
    return proposals


def choose_proposals(original, extension):
    chosen = {}
    for market in MARKETS:
        a, b = original[market], extension[market]
        if a["outer_support"] != b["outer_support"]:
            raise ValueError("Development proposal comparisons need identical cohorts")
        if a["baseline_contract_id"] != b["baseline_contract_id"]:
            raise ValueError("Development baseline contracts differ")
        chosen[market] = a if a["rmse"] <= b["rmse"] else b
    return chosen


def inspection_claim(registry, spec, experiment):
    """Durable external claim: a new experiment/spec cannot reset consumed data."""
    registry = _safe(registry)
    registry.mkdir(parents=True, exist_ok=True)
    path = registry / "OPENED.json"
    identity = {"version": VERSION, "dataset_id": spec["dataset_id"], "partition": cd.PARTITION,
                "specification_id": digest(spec), "experiment": str(_safe(experiment)),
                "status": "consumed_for_model_evaluation", "later_partitions_opened": []}
    if path.exists():
        actual = read_json(_safe(path))
        if {k: actual.get(k) for k in identity} != identity:
            raise ValueError("Confirmation already consumed by another experiment/specification")
        return actual
    event = {**identity, "opened_at": datetime.now(timezone.utc).isoformat()}
    atomic_write_json(path, event)
    return event


def verify_source_amendment(target, saved, current):
    """Allow a documented implementation repair, never a changed test recipe.

    The original freeze/claim stays authoritative. A separate immutable record
    and copies bind repaired source bytes; all non-source specification fields
    must remain exactly equal, including models, protocol, inputs and gates.
    """
    if {k: v for k, v in current.items() if k != "source_hashes"} != {
            k: v for k, v in saved.items() if k != "source_hashes"}:
        raise ValueError("Frozen confirmation specification changed; cannot rerun as untouched")
    directory = target / "correctness-amendment-0001"
    verify_complete(directory, required={"amendment.json"})
    record = read_json(directory / "amendment.json")
    if (record.get("original_specification_id") != digest(saved)
            or record.get("original_source_hashes") != saved["source_hashes"]
            or record.get("replacement_source_hashes") != current["source_hashes"]
            or record.get("predictions_generated_before_repair") is not False
            or record.get("labels_scored_before_repair") is not False
            or record.get("unchanged_candidates_cohorts_and_rules") is not True
            or not isinstance(record.get("reason"), str) or not record["reason"].strip()):
        raise ValueError("Missing or inconsistent correctness amendment")
    changed = {name for name, value in current["source_hashes"].items()
               if value != saved["source_hashes"].get(name)}
    if changed != set(record.get("changed_sources", [])) or not changed:
        raise ValueError("Correctness amendment changed-source mismatch")
    for name in changed:
        if sha(directory / "source" / name) != current["source_hashes"][name]:
            raise ValueError("Correctness amendment source archive mismatch")
    return {"directory": directory.name, "complete_sha256": sha(directory / "COMPLETE.json"),
            "reason": record["reason"], "original_specification_id": digest(saved)}


def _stage(target, name, function):
    """Reuse completed immutable stages; retain incomplete attempts for diagnosis."""
    pointer = target / (name + ".json")
    if pointer.exists():
        saved = read_json(pointer)
        if not re.fullmatch(re.escape(name) + r"-\d{4}", saved.get("directory", "")):
            raise ValueError("Unsafe stage pointer")
        path = target / saved["directory"]
        verify_complete(path)
        if sha(path / "COMPLETE.json") != saved["complete_sha256"]:
            raise ValueError("Stage completion changed")
        return path
    for orphan in sorted(target.glob(name + "-????")):
        if (orphan / "COMPLETE.json").exists():
            verify_complete(orphan)
            atomic_write_json(pointer, {"directory": orphan.name, "complete_sha256": sha(orphan / "COMPLETE.json")})
            return orphan
    path = target / f"{name}-{len(list(target.glob(name + '-????'))) + 1:04d}"
    path.mkdir()
    function(path)
    complete_atomic(path)
    atomic_write_json(pointer, {"directory": path.name, "complete_sha256": sha(path / "COMPLETE.json")})
    return path


def decision(uncertainty):
    """Apply the predeclared gates without choosing/reweighting any predictor."""
    result = {"version": VERSION, "by_market": {}, "publication_enabled": False, "promotion_allowed": False}
    for market in MARKETS:
        comparisons = {base: uncertainty[f"{base}_1w"]["by_market"][market]["overall"]
                       for base in RULES["baseline_methods"]}
        support = comparisons["statistical"]["support"]
        if any(c["support"] != support for c in comparisons.values()):
            raise ValueError("Qualification comparison cohorts differ")
        reasons = []
        if support["unique_fixtures"] < 1000 or support["calendar_weeks"] < 20:
            reasons.append("insufficient_confirmation_support")
        for base, c in comparisons.items():
            if c["point_improvement_at_least_one_percent"] is not True:
                reasons.append(base + ":below_one_percent")
            if c["interval_status"] != "available" or c["improvement_interval_excludes_zero"] is not True:
                reasons.append(base + ":positive_improvement_not_established")
        enough = support["unique_fixtures"] >= 1000 and support["calendar_weeks"] >= 20
        intervals = all(c["interval_status"] == "available" for c in comparisons.values())
        harmful = any(c.get("improvement_interval") and c["improvement_interval"]["upper"] < 0
                      for c in comparisons.values())
        status = ("qualified_pooled_numerical" if not reasons else "inconclusive" if not enough or not intervals
                  else "harmful_against_at_least_one_baseline" if harmful else "no_qualified_improvement")
        slices = uncertainty["statistical_1w"]["by_market"][market]["slices"]
        leagues = {}
        for league in COMPETITIONS:
            item = slices["competition"].get(league)
            why = []
            if item is None:
                why.append("no_eligible_confirmation_fixtures")
            else:
                if item["support"]["unique_fixtures"] < 200 or item["support"]["match_dates"] < 20:
                    why.append("insufficient_league_support")
                if item["interval_status"] != "available":
                    why.append("insufficient_league_precision")
                if item["slice_deterioration_below_five_percent"] is not True:
                    why.append("deterioration_below_five_percent_not_established")
            if reasons:
                why.append("pooled_candidate_not_qualified")
            leagues[league] = {"qualified": not why, "reasons": why,
                "support": None if item is None else item["support"],
                "deterioration_upper_95": None if item is None else item["deterioration_upper_95"]}
        # Other supported slices are disclosed as diagnostics, with the same
        # material-regression threshold; league qualification remains explicit.
        regressions = {field: {key: item["deterioration_upper_95"] for key, item in values.items()
            if item["interval_status"] == "available" and item["slice_deterioration_below_five_percent"] is not True}
            for field, values in slices.items() if field != "competition"}
        slice_qualification = {field: {key: {"qualified": not reasons and item["interval_status"] == "available"
                    and item["slice_deterioration_below_five_percent"] is True,
                "support": item["support"], "interval_status": item["interval_status"],
                "deterioration_upper_95": item["deterioration_upper_95"],
                "unavailable_reasons": item.get("unavailable_reasons", [])}
                for key, item in values.items()} for field, values in slices.items() if field != "competition"}
        result["by_market"][market] = {"status": status, "reasons": reasons, "support": support,
            "comparisons": comparisons, "league_qualification": leagues,
            "other_slice_qualification": slice_qualification,
            "supported_other_slice_regression_flags": regressions,
            "two_week_sensitivity": {base: uncertainty[f"{base}_2w"]["by_market"][market]["overall"]
                                      for base in RULES["baseline_methods"]}}
    return result


def _predictions(rows, baselines, target, spec):
    lookup = {(r["fixture_id"], r["market"]): r for r in baselines}
    if len(lookup) != len(baselines):
        raise ValueError("Duplicate confirmation baseline")
    output = []
    for market in MARKETS:
        selected = [r for r in rows if r["market_eligibility"][market]["eligible"]]
        stats = [lookup[(r["fixture"]["fixture_id"], market)] for r in selected]
        proposal = spec["candidates"][market]
        infer = predict_bundle if proposal["origin"] == "extension" else predict_strategy
        values = infer(target / "freeze" / ("candidate-" + market), selected,
            feature_contract_id=spec["feature_contract_id"], statistical_rows=stats,
            baseline_contract_id=spec["baseline_contract_id"])
        for method, predictions in (("candidate", values), ("statistical", [r["statistical"] for r in stats]),
                                     ("league_average", [r["league_average"] for r in stats])):
            # Reuse precisely the development slice definitions. No observed
            # target is supplied; remove the placeholder before persistence.
            records = prediction_rows([{**r, "target": None} for r in selected], predictions,
                market=market, fold={"fold_id": cd.PARTITION}, method=method, names=spec["feature_names"],
                weight_ml=proposal["weight_ml"] if method == "candidate" else 0)
            for record in records:
                record.pop("target")
                record["qualification"] = "FROZEN_CONFIRMATION_PREDICTION"
                from .data import _finite
                if not _finite(record["prediction"]) or record["prediction"] < 0:
                    raise ValueError("Invalid confirmation prediction; no fixture may be dropped")
            output.extend(records)
    expected = {(r["fixture"]["fixture_id"], m, method) for r in rows for m in MARKETS
                if r["market_eligibility"][m]["eligible"] for method in ("candidate", "statistical", "league_average")}
    if {(r["fixture_id"], r["market"], r["method"]) for r in output} != expected or len(output) != len(expected):
        raise ValueError("Confirmation prediction coverage changed")
    return output


def run_confirmation(*, root, dataset, baselines, batch_c, extension, name, resume=False):
    root = _safe(root)
    dataset, baselines, batch_c, extension = map(_safe, (dataset, baselines, batch_c, extension))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", name):
        raise ValueError("Use a simple confirmation experiment name")
    target = _safe(root / "Index/prediction_experiments" / name) if resume else new_experiment(root, name)
    if not target.is_dir():
        raise ValueError("Missing confirmation experiment")
    with experiment_lock(target), runtime_environment(target):
        environment = environment_report()
        # Before the claim only development artifacts and outcome-free split
        # metadata can be opened. Raw completion hashes bind later stores.
        reads = [dataset / file for file in ALLOWED_DATASET_FILES]
        reads += _children(baselines) + _children(batch_c) + _children(extension)
        with offline_guard(root=root, output=target, readable_files=reads):
            data = DevelopmentDataset(dataset)
            if (data.boundaries["phase3_confirmation_start"] != utc("2024-01-01T00:00:00Z")
                    or data.boundaries["calibration_start"] != utc("2025-01-01T00:00:00Z")):
                raise ValueError("This frozen Batch D protocol reserves exactly the 2024 confirmation period")
            verify_complete(baselines, required={"manifest.json"})
            base_manifest = read_json(baselines / "manifest.json")
            baseline = baseline_metadata(ROOT)
            if (base_manifest["baseline"] != baseline or base_manifest["dataset_id"] != data.manifest["dataset_id"]):
                raise ValueError("Statistical reconstruction source/dataset changed")
            original = _development_proposals(batch_c, data, extension=False)
            extended = _development_proposals(extension, data, extension=True)
            chosen = choose_proposals(original, extended)
            if any(p["baseline_contract_id"] != digest(baseline) for p in chosen.values()):
                raise ValueError("Candidate and comparator contracts differ")
            completed = read_json(dataset / "COMPLETE.json")
            spec = {"version": VERSION, "rules": RULES, "dataset_id": data.manifest["dataset_id"],
                "dataset": str(dataset), "dataset_complete_sha256": sha(dataset / "COMPLETE.json"),
                "feature_contract_id": data.manifest["feature_contract_id"], "feature_names": data.schema["names"],
                "boundaries": data.splits["boundaries"], "split_sha256": sha(dataset / "splits.json"),
                "lockbox_policy_sha256": sha(dataset / "lockbox-policy.json"),
                "protected_inputs": {n: completed[n] for n in (cd.FEATURES, cd.LABELS, cd.HISTORY)},
                "candidates": chosen, "development_comparison": {"batch_c": original, "extension": extended},
                "source_hashes": source_hashes(), "protocol_sha256": sha(ROOT / PROTOCOL),
                "environment": environment, "baseline_contract": baseline, "baseline_contract_id": digest(baseline),
                "provenance": {str(p): sha(p / "COMPLETE.json") for p in (baselines, batch_c, extension)},
                "training_performed": False, "publication_enabled": False, "promotion_allowed": False}
            freeze = target / "freeze"
            runtime_sources = spec["source_hashes"]
            amendment = None
            if freeze.exists():
                verify_complete(freeze)
                saved = read_json(freeze / "specification.json")
                if saved != spec:
                    amendment = verify_source_amendment(target, saved, spec)
                    spec = saved
            else:
                freeze.mkdir()
                atomic_write_json(freeze / "specification.json", spec)
                for market, proposal in chosen.items():
                    shutil.copytree(proposal["source_bundle"], freeze / ("candidate-" + market))
                for relative in ("manifest.json", "feature-schema.json", "splits.json", "lockbox-policy.json", "COMPLETE.json"):
                    destination = freeze / "dataset" / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(dataset / relative, destination)
                for relative in (*spec["source_hashes"], PROTOCOL):
                    destination = freeze / "source" / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / relative, destination)
                complete_atomic(freeze)
        # This external record precedes ANY read of confirmation feature/label
        # stores or outcome-rich history. An interrupted run still consumes it.
        # Bind the calendar period as well as dataset identity: re-exporting
        # or copying data must not manufacture a fresh untouched 2024 test.
        registry = root / "Index/prediction_experiments/phase3-confirmation-inspections/phase3_confirmation_20240101_20250101"
        event = inspection_claim(registry, spec, target)
        if amendment is not None:
            amended_event = registry / "IMPLEMENTATION-AMENDMENT-0001.json"
            if amended_event.exists():
                if read_json(amended_event) != amendment:
                    raise ValueError("Permanent amendment record differs")
            else:
                atomic_write_json(amended_event, amendment)
        with offline_guard(root=root, output=target):
            if not (target / "inspection.json").exists():
                atomic_write_json(target / "inspection.json", event)
            elif read_json(target / "inspection.json") != event:
                raise ValueError("Experiment inspection differs from permanent registry")
            if (target / "COMPLETE.json").exists():
                verify_complete(target)
                return target

        def prepare(path):
            with offline_guard(root=root, output=target, readable_files=[dataset / cd.FEATURES, dataset / cd.HISTORY]):
                print("Opening claimed confirmation features; reconstructing pre-match inputs", flush=True)
                rows = cd.feature_rows(cd.checked_read(dataset, cd.FEATURES, spec["protected_inputs"][cd.FEATURES]), data)
                history, competitions, audit = cd.history_before(
                    cd.checked_read(dataset, cd.HISTORY, spec["protected_inputs"][cd.HISTORY]).decode("utf-8"),
                    data.boundaries["calibration_start"])
                baseline_rows = cd.prepare_rows(rows, history, competitions, scoring=baseline["scoring"],
                                                 end=data.boundaries["calibration_start"])
                write_jsonl(path / "features.jsonl", rows)
                write_jsonl(path / "baselines.jsonl", baseline_rows)
                atomic_write_json(path / "coverage.json", cd.coverage(rows))
                atomic_write_json(path / "history-access.json", audit)

        with offline_guard(root=root, output=target):
            inputs = _stage(target, "inputs", prepare)
            rows = [json.loads(s) for s in (inputs / "features.jsonl").read_text().splitlines()]
            baseline_rows = [json.loads(s) for s in (inputs / "baselines.jsonl").read_text().splitlines()]

            def predict(path):
                print("Applying frozen candidates; no model or blend fitting", flush=True)
                predictions = _predictions(rows, baseline_rows, target, spec)
                write_jsonl(path / "predictions.jsonl", predictions)
                atomic_write_json(path / "manifest.json", {"specification_id": digest(spec),
                    "inputs_complete_sha256": sha(inputs / "COMPLETE.json"), "rows": len(predictions),
                    "confirmation_label_store_opened": False, "training_performed": False})

            predictions = _stage(target, "predictions", predict)

            def evaluate(path):
                with offline_guard(root=root, output=target, readable_files=[dataset / cd.LABELS]):
                    # The prediction COMPLETE marker already exists here.
                    labels = cd.labels_by_fixture(cd.checked_read(dataset, cd.LABELS, spec["protected_inputs"][cd.LABELS]), rows)
                    forecasts = [json.loads(s) for s in (predictions / "predictions.jsonl").read_text().splitlines()]
                    scored = [{**r, "target": labels[r["fixture_id"]][r["market"]]} for r in forecasts]
                    write_jsonl(path / "scored-predictions.jsonl", scored)
                    atomic_write_json(path / "metrics.json", summarize(scored))
                    uncertainty = {}
                    for base in RULES["baseline_methods"]:
                        for weeks in (1, 2):
                            print(f"Confirmation uncertainty: {base}, {weeks}-week blocks", flush=True)
                            uncertainty[f"{base}_{weeks}w"] = paired_comparison(scored, "candidate", base,
                                resamples=RULES["resamples"], seed=RULES["seed"], block_weeks=weeks)
                    atomic_write_json(path / "uncertainty.json", uncertainty)
                    atomic_write_json(path / "decision.json", decision(uncertainty))
                    atomic_write_json(path / "manifest.json", {"predictions_complete_sha256": sha(predictions / "COMPLETE.json"),
                        "specification_id": digest(spec), "confirmation_status": "consumed",
                        "calibration_opened": False, "final_system_test_opened": False, "prospective_reserve_opened": False})

            evaluation = _stage(target, "evaluation", evaluate)
            if source_hashes() != runtime_sources or sha(ROOT / PROTOCOL) != spec["protocol_sha256"]:
                raise RuntimeError("Confirmation source changed during execution")
            atomic_write_json(target / "RESULT.json", {"version": VERSION, "status": "complete",
                "specification_id": digest(spec), "evaluation_directory": evaluation.name,
                "implementation_amendment": amendment, "execution_source_hashes": runtime_sources,
                "inspection_registry": str(registry / "OPENED.json"), "confirmation_status": "consumed",
                "reserved_partitions": ["calibration", "final_system_test", "prospective_reserve"],
                "training_performed": False, "publication_enabled": False, "promotion_allowed": False})
            complete_atomic(target)
    return target
