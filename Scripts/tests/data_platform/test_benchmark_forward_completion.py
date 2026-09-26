"""Synthetic final-integration cases; no real datasets, fitting or providers."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from Scripts.data_platform.features.benchmarks import checkpoints, forward
from Scripts.data_platform.features.benchmarks.artifacts import read_json, verify_complete
from Scripts.data_platform.features.benchmarks.selection import recipes
from Scripts.rag_ingest.core.model_features import digest


def fold(number):
    year, month = 2022 + (number - 1) // 2, 1 if number % 2 else 7
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end = datetime(year + (month == 7), 7 if month == 1 else 1, 1, tzinfo=timezone.utc)
    return {"fold_id": f"development-{number:02d}", "fit_cutoff": start.isoformat(),
            "validation_start": start.isoformat(), "validation_end": end.isoformat()}


def fake_completed_result(tmp_path, monkeypatch):
    root = tmp_path / "project"
    target = root / "Index/prediction_experiments/resumable"
    target.mkdir(parents=True)
    baselines, variants = root / "baselines", root / "variants"
    for item in (baselines, variants):
        item.mkdir()
        checkpoints.atomic_write_json(item / "COMPLETE.json", {})
    spec = {"version": forward.VERSION, "source_hashes": {"synthetic-test": "frozen"}}
    specification_id = digest(spec)
    checkpoints.atomic_write_json(target / "specification.json", spec)
    store = checkpoints.Checkpoints(target / "tasks", specification_id)
    store.get({"synthetic_task": True}, lambda _: {"status": "complete", "predictions": np.array([1.])})
    report = target / "reports-0001"
    report.mkdir()
    checkpoints.atomic_write_json(report / "review.json", {"synthetic_test": True, "status": "complete"})
    checkpoints.complete_atomic(report)
    checkpoints.atomic_write_json(target / "RESULT.json", {"status": "complete", "report_directory": report.name,
                                                           "specification_id": specification_id})
    monkeypatch.setattr(forward, "environment_report", lambda: {"synthetic_test": True})
    monkeypatch.setattr(forward, "DevelopmentDataset", lambda _: SimpleNamespace(folds=[fold(i) for i in range(1, 5)]))
    monkeypatch.setattr(forward, "load_baselines", lambda *_: ({"baseline": "synthetic"}, {}))
    monkeypatch.setattr(forward, "load_variants", lambda *_: ({"variants": list(forward.PROFILE_VARIANTS)}, {}))
    monkeypatch.setattr(forward, "specification", lambda *_: spec)
    monkeypatch.setattr(forward, "source_hashes", lambda: spec["source_hashes"])
    monkeypatch.setattr(forward, "_execute", lambda *args, **kwargs: pytest.fail("Committed reports must never rerun experiments"))
    monkeypatch.setattr(forward, "fit_estimator", lambda *args, **kwargs: pytest.fail("Committed result must not refit"))
    arguments = {"root": root, "dataset": root / "unused-synthetic-dataset", "baselines": baselines,
                 "variants": variants, "name": "resumable", "resume": True}
    return target, arguments


def test_result_committed_before_top_completion_recovers_without_refitting(tmp_path, monkeypatch):
    target, arguments = fake_completed_result(tmp_path, monkeypatch)
    before = {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}
    assert not (target / "COMPLETE.json").exists()
    assert forward.run_development(**arguments) == target
    assert (target / "COMPLETE.json").exists()
    manifest = verify_complete(target)
    assert "RESULT.json" in manifest and "reports-0001/COMPLETE.json" in manifest
    assert len(list((target / "tasks").glob("*/attempt-*"))) == 1
    assert list(target.glob("invocation-*.json")) == []
    for relative, content in before.items():
        assert (target / relative).read_bytes() == content
    completion = (target / "COMPLETE.json").read_bytes()
    assert forward.run_development(**arguments) == target
    assert (target / "COMPLETE.json").read_bytes() == completion


@pytest.mark.parametrize("corruption", ["report", "unfinished_task", "result_specification", "result_path"])
def test_result_resume_rejects_invalid_committed_evidence_before_retraining(tmp_path, monkeypatch, corruption):
    target, arguments = fake_completed_result(tmp_path, monkeypatch)
    if corruption == "report":
        (target / "reports-0001/review.json").write_text('{"changed": true}')
    elif corruption == "unfinished_task":
        (target / "tasks" / ("e" * 64)).mkdir()
    else:
        result = read_json(target / "RESULT.json")
        result["specification_id" if corruption == "result_specification" else "report_directory"] = (
            "different" if corruption == "result_specification" else "../outside")
        (target / "RESULT.json").write_text(__import__("json").dumps(result))
    with pytest.raises(ValueError):
        forward.run_development(**arguments)
    assert not (target / "COMPLETE.json").exists()
    assert len(list((target / "tasks").glob("*/attempt-*"))) == 1


def history_inputs():
    folds = [fold(i) for i in range(1, 5)]
    selected = next(r for r in recipes() if r["family"] == "ridge" and r["params"] == {"alpha": 50}
                    and r["lookback_days"] is None and r["half_life_days"] == 365)
    choices = {"goals": {f["fold_id"]: {"selected_ml": {"recipe": selected}} for f in folds}}
    names = ["home_current_matches", "away_current_matches", "missing_rate",
             "home_current_matches__missing", "away_current_matches__missing", "missing_rate__missing"]
    validations, grid = {}, {}
    for index, f in enumerate(folds[2:], 3):
        kickoff = datetime.fromisoformat(f["fit_cutoff"]) + timedelta(days=1)
        rows = [{"fixture": {"fixture_id": index * 100 + i, "kickoff": kickoff.isoformat(), "competition": "EPL", "season": 2023},
                 "target": float(i), "snapshot_id": digest([index, i]), "source_class": "verified_local_reconstruction",
                 "round_group": "domestic_regular", "forecast_stage": "reconstructed_immediately_before_kickoff",
                 "values": [10., 10., None, 0., 0., 1.], "support": {"goals": {"home": {"count": 10}, "away": {"count": 10}}}}
                for i in range(3 if index == 3 else 2)]
        validations[("goals", f["fold_id"])] = rows
        for recipe in recipes():
            if recipe["family"] != "ridge" or recipe["params"] != {"alpha": 50} or recipe["half_life_days"] not in (None, 365):
                continue
            grid[("goals", f["fold_id"], recipe["recipe_id"])] = {
                "status": "complete", "support": {"synthetic_test": True},
                "checkpoint_id": digest([f["fold_id"], recipe["recipe_id"]]),
                "predictions": np.asarray([row["target"] + (100 if index == 3 else 0) for row in rows])}
    return folds, selected, choices, names, validations, grid


@pytest.mark.parametrize("missing_all_folds", [False, True])
def test_history_length_aggregate_uses_common_fixtures_when_one_window_is_unsupported(monkeypatch, missing_all_folds):
    monkeypatch.setattr(forward, "MARKETS", ("goals",))
    folds, selected, choices, names, validations, grid = history_inputs()
    limited = next(r for r in recipes() if r["family"] == "ridge" and r["params"] == selected["params"]
                   and r["lookback_days"] == 730 and r["half_life_days"] is None)
    for f in folds[2:] if missing_all_folds else folds[2:3]:
        grid[("goals", f["fold_id"], limited["recipe_id"])]["status"] = "insufficient_support"
    result = forward._history_comparisons(grid, validations, choices, folds, names)["goals"]
    assert len(result["fits"]) == 12
    assert sum(item["status"] == "insufficient_support" for item in result["fits"]) == (2 if missing_all_folds else 1)
    paired = result["paired_common_cohort"]
    if missing_all_folds:
        assert paired["status"] == "insufficient_common_coverage" and paired["n"] == 0 and paired["methods"] == {}
        assert "history_730_no_decay" not in result["methods"]
    else:
        assert paired["status"] == "complete" and paired["n"] == 2
        assert paired["folds"] == ["development-04"]
        assert len(paired["methods"]) == 6
        assert all(score["n"] == 2 and score["rmse"] == 0 for score in paired["methods"].values())
        assert result["methods"]["history_all_no_decay"]["coverage_specific_metrics"]["n"] == 5
        assert result["methods"]["history_all_no_decay"]["coverage_specific_metrics"]["rmse"] > 0
        assert result["methods"]["history_730_no_decay"]["coverage_specific_metrics"]["n"] == 2


def fit_inputs():
    f = fold(1)
    cutoff = datetime.fromisoformat(f["fit_cutoff"])
    def row(fid, kickoff):
        return {"fixture": {"fixture_id": fid, "kickoff": kickoff.isoformat(), "competition": "EPL"},
                "target": 1., "values": [1.], "snapshot_id": digest(fid),
                "label_available_at": (kickoff + timedelta(hours=3)).isoformat()}
    train = [row(i, cutoff - timedelta(days=30)) for i in range(1, 2001)]
    validation = [row(i, cutoff + timedelta(days=1)) for i in range(2001, 2501)]
    weights, support = forward._support(train, validation, f["fit_cutoff"], None)
    return {"market": "goals", "fold": f, "recipe": recipes()[0], "train": train,
            "validation": validation, "weights": weights, "support": support, "names": ["synthetic_feature"], "view": "synthetic-view"}


def test_expected_estimator_failure_is_committed_visible_and_not_refitted(tmp_path, monkeypatch):
    arguments = fit_inputs()
    def fail(*args, **kwargs):
        raise forward.EstimatorFitError("synthetic convergence failure")
    monkeypatch.setattr(forward, "fit_estimator", fail)
    first = checkpoints.Checkpoints(tmp_path / "tasks", "synthetic-specification")
    result = forward._fit_task(first, **arguments)
    assert result["status"] == "failed" and result["gate"]["qualified"]
    assert result["failure"] == {"type": "EstimatorFitError", "message": "synthetic convergence failure"}
    assert result["predictions"].size == 0
    checkpoint = tmp_path / "tasks" / result["checkpoint_id"]
    assert (checkpoint / "SUCCESS.json").exists()  # successful recording of the failed recipe
    monkeypatch.setattr(forward, "fit_estimator", lambda *a, **kw: pytest.fail("known failed recipe must not refit automatically"))
    resumed = checkpoints.Checkpoints(tmp_path / "tasks", "synthetic-specification")
    saved = forward._fit_task(resumed, **arguments)
    assert saved["status"] == "failed" and saved["failure"] == result["failure"]
    assert resumed.created == 0 and resumed.reused == 1
    monkeypatch.setattr(forward, "recipes", lambda: [arguments["recipe"]])
    records = forward.selection_records(
        {("goals", arguments["fold"]["fold_id"], arguments["recipe"]["recipe_id"]): saved},
        {("goals", arguments["fold"]["fold_id"]): arguments["validation"]},
        market="goals", prior_folds=[arguments["fold"]], receiving_cutoff=fold(2)["fit_cutoff"])
    assert len(records) == 1 and records[0]["status"] == "failed"
    assert records[0]["failure"] == result["failure"] and records[0]["n"] == 500
    assert "rmse" not in records[0]


def test_unexpected_fit_integrity_error_stops_instead_of_becoming_an_omitted_recipe(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("synthetic feature contract mismatch")
    monkeypatch.setattr(forward, "fit_estimator", fail)
    store = checkpoints.Checkpoints(tmp_path / "tasks", "synthetic-specification")
    with pytest.raises(ValueError, match="contract mismatch"):
        forward._fit_task(store, **fit_inputs())
    assert not list((tmp_path / "tasks").glob("*/SUCCESS.json"))
    assert len(list((tmp_path / "tasks").glob("*/attempt-*/FAILED.json"))) == 1
