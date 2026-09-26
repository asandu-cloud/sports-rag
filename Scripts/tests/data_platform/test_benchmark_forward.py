"""Boundaries for full forward selection, blends and immutable continuation."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from Scripts.data_platform.features.benchmarks import checkpoints, forward
from Scripts.data_platform.features.benchmarks.artifacts import read_json
from Scripts.data_platform.features.benchmarks.selection import recipes
from Scripts.rag_ingest.core.model_features import digest


def _computed(attempt):
    return {"status": "complete", "predictions": np.array([1.0, 2.0]), "raw_predictions": [-1.0, 2.0]}


def test_checkpoint_resume_never_refits_and_rejects_corruption_before_loading(tmp_path, monkeypatch):
    first = checkpoints.Checkpoints(tmp_path / "tasks", "frozen")
    result = first.get({"task": 1}, _computed)
    second = checkpoints.Checkpoints(tmp_path / "tasks", "frozen")
    restored = second.get({"task": 1}, lambda _: pytest.fail("completed fit rerun"))
    assert np.array_equal(result["predictions"], restored["predictions"])
    assert result["checkpoint_id"] == restored["checkpoint_id"]
    assert second.previous_fits == 1 and second.created == 0
    array_file = next((tmp_path / "tasks").glob("*/attempt-*/predictions.npz"))
    with array_file.open("ab") as handle:
        handle.write(b"corruption")
    third = checkpoints.Checkpoints(tmp_path / "tasks", "frozen")
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("unverified arrays opened"))
    with pytest.raises(ValueError, match="checksum"):
        third.get({"task": 1}, _computed)


def test_partial_checkpoint_is_preserved_and_retried_separately(tmp_path):
    first = checkpoints.Checkpoints(tmp_path / "tasks", "frozen")
    def fail(attempt):
        (attempt / "partial.txt").write_text("preserve")
        raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError, match="interrupted"):
        first.get({"task": 1}, fail)
    second = checkpoints.Checkpoints(tmp_path / "tasks", "frozen")
    second.get({"task": 1}, _computed)
    assert len(list((tmp_path / "tasks").glob("*/attempt-*"))) == 2
    assert next((tmp_path / "tasks").glob("*/attempt-0001/partial.txt")).read_text() == "preserve"
    assert len(list((tmp_path / "tasks").glob("*/attempt-0001/FAILED.json"))) == 1


def test_changed_contract_or_spec_cannot_reuse_existing_fit(tmp_path):
    fits = []
    def compute(attempt):
        fits.append(attempt)
        return _computed(attempt)
    store = checkpoints.Checkpoints(tmp_path / "tasks", "a")
    store.get({"weight": 1}, compute)
    store.get({"weight": 1}, compute)
    store.get({"weight": 2}, compute)
    changed = checkpoints.Checkpoints(tmp_path / "tasks", "b")
    changed.get({"weight": 1}, compute)
    assert len(fits) == 3


def test_fit_budget_persists_across_process_restarts(tmp_path):
    first = checkpoints.Checkpoints(tmp_path / "tasks", "a", max_fits=1)
    first.get({"task": 1}, _computed)
    second = checkpoints.Checkpoints(tmp_path / "tasks", "a", max_fits=1)
    second.get({"task": 1}, lambda _: pytest.fail("cached task consumes budget"))
    with pytest.raises(checkpoints.RunBudgetReached, match="fit budget"):
        second.get({"task": 2}, _computed)


def test_checkpoint_pointer_traversal_and_concurrent_run_rejected(tmp_path):
    store = checkpoints.Checkpoints(tmp_path / "tasks", "a")
    result = store.get({"task": 1}, _computed)
    (tmp_path / "tasks" / result["checkpoint_id"] / "SUCCESS.json").write_text('{"attempt":"../other"}')
    with pytest.raises(ValueError, match="pointer"):
        checkpoints.Checkpoints(tmp_path / "tasks", "a").get({"task": 1}, _computed)
    with checkpoints.experiment_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another process"):
            with checkpoints.experiment_lock(tmp_path):
                pytest.fail("concurrent experiment entered")


def _fold(n):
    year, month = 2022 + (n - 1) // 2, 1 if n % 2 else 7
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end = datetime(year + (month == 7), 7 if month == 1 else 1, 1, tzinfo=timezone.utc)
    return {"fold_id": f"development-{n:02d}", "fit_cutoff": start.isoformat(),
            "validation_start": start.isoformat(), "validation_end": end.isoformat()}


def _grid():
    folds = [_fold(i) for i in range(1, 5)]
    grid, validations = {}, {}
    for fold in folds:
        cutoff = datetime.fromisoformat(fold["fit_cutoff"])
        rows = [{"fixture": {"fixture_id": int(fold["fold_id"][-2:]) * 10000 + i,
                             "kickoff": (cutoff + timedelta(hours=i * 4)).isoformat()},
                 "target": 2.0, "label_available_at": (cutoff + timedelta(hours=i * 4 + 3)).isoformat()}
                for i in range(501)]
        validations[("goals", fold["fold_id"])] = rows
        for index, recipe in enumerate(recipes()):
            grid[("goals", fold["fold_id"], recipe["recipe_id"])] = {
                "status": "complete", "predictions": np.repeat(2.0 + index / 1000, len(rows)),
                "support": {"training_unique_fixtures": 6000, "effective_sample_size": 4000,
                            "validation_unique_fixtures": len(rows)},
                "training_max_label_available_at": (cutoff - timedelta(hours=1)).isoformat()}
    return folds, grid, validations


def test_outer_outcomes_cannot_change_recipe_chosen_for_that_outer_fold():
    folds, grid, rows = _grid()
    first = forward.choose_before(grid, rows, market="goals", folds=folds, index=2)
    for row in rows[("goals", folds[2]["fold_id"])]:
        row["target"] = 5000.0
    for recipe in recipes():
        grid[("goals", folds[2]["fold_id"], recipe["recipe_id"])]["predictions"] *= 500
    second = forward.choose_before(grid, rows, market="goals", folds=folds, index=2)
    assert first == second
    assert first["selection_fold_ids"] == ["development-01", "development-02"]


def test_boundary_late_labels_do_not_enter_tuning_score():
    folds, grid, rows = _grid()
    cutoff = folds[2]["fit_cutoff"]
    late = rows[("goals", folds[1]["fold_id"])][-1]
    late["label_available_at"] = cutoff
    late["target"] = 100000
    records = forward.selection_records(grid, rows, market="goals", prior_folds=folds[:2],
                                        receiving_cutoff=cutoff)
    last = [r for r in records if r["fold_id"] == folds[1]["fold_id"]]
    assert all(r["n"] == 500 and r["support"]["validation_unique_fixtures"] == 500 for r in last)
    assert max(r["rmse"] for r in last) < 1


def _oof_rows():
    return [{"fixture_id": i + 1, "fold_id": f"development-{i + 1:02d}",
             "kickoff": f"2022-0{1 + i * 6}-10T12:00:00+00:00",
             "label_available_at": f"2022-0{1 + i * 6}-10T15:00:00+00:00",
             "fit_cutoff": f"2022-0{1 + i * 6}-01T00:00:00+00:00",
             "selection_cutoff": f"2022-0{1 + i * 6}-01T00:00:00+00:00",
             "training_max_label_available_at": f"2021-12-31T20:00:00+00:00",
             "selection_validation_ends": [] if i == 0 else ["2022-07-01T00:00:00+00:00"],
             "target": 3, "statistical": 2, "ml": 4}
            for i in range(2)]


def test_blend_uses_only_prior_available_oof_and_zero_is_valid():
    rows = _oof_rows()
    fitted = forward.earlier_blend(rows, cutoff="2023-01-01T00:00:00+00:00")
    assert fitted["weight_ml"] == .5
    future = {**rows[-1], "fixture_id": 3, "target": 100000,
              "label_available_at": "2023-01-01T00:00:00+00:00"}
    assert forward.earlier_blend(rows + [future], cutoff="2023-01-01T00:00:00+00:00") == fitted
    for row in rows:
        row["target"] = 2
    assert forward.earlier_blend(rows, cutoff="2023-01-01T00:00:00+00:00")["weight_ml"] == 0


@pytest.mark.parametrize("mutation", ["duplicate", "late_training", "future_selection"])
def test_blend_rejects_invalid_oof_provenance(mutation):
    rows = _oof_rows()
    if mutation == "duplicate":
        rows[1]["fixture_id"] = rows[0]["fixture_id"]
    elif mutation == "late_training":
        rows[0]["training_max_label_available_at"] = rows[0]["fit_cutoff"]
    else:
        rows[0]["selection_validation_ends"] = ["2023-01-01T00:00:00+00:00"]
    with pytest.raises(ValueError):
        forward.earlier_blend(rows, cutoff="2023-01-01T00:00:00+00:00")


def test_final_refit_excludes_late_labels_ineligible_rows_and_old_lookback():
    def row(fid, kickoff, label, eligible=True):
        return {"fixture": {"fixture_id": fid, "kickoff": kickoff}, "label_available_at": label,
                "market_eligibility": {"goals": {"eligible": eligible}}, "labels": {"goals": 2}}
    data = SimpleNamespace(splits={"boundaries": {"phase3_confirmation_start": "2024-01-01T00:00:00+00:00"}},
        rows=[row(1, "2023-12-30T12:00:00+00:00", "2023-12-30T15:00:00+00:00"),
              row(2, "2023-12-31T23:00:00+00:00", "2024-01-01T02:00:00+00:00"),
              row(3, "2023-12-29T12:00:00+00:00", "2023-12-29T15:00:00+00:00", False),
              row(4, "2020-01-01T12:00:00+00:00", "2020-01-01T15:00:00+00:00")])
    actual = forward.final_training_rows(data, "goals", {"lookback_days": 730})
    assert [r["fixture"]["fixture_id"] for r in actual] == [1]

