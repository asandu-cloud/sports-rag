"""Synthetic orchestration contracts; no real data or estimator fits."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import numpy as np
import pytest

from Scripts.data_platform.features.benchmarks import checkpoints, evaluation, extension, extension_models
from Scripts.data_platform.features.benchmarks import extension_bundle
from Scripts.data_platform.features.benchmarks.artifacts import read_json, verify_complete
from Scripts.rag_ingest.core.model_features import digest

NAMES = ["home_current_matches", "away_current_matches", "optional",
         "home_current_matches__missing", "away_current_matches__missing", "optional__missing"]
FEATURE_ID = digest("synthetic-feature-contract")


def fold(number):
    year, month = 2022 + (number - 1) // 2, 1 if number % 2 else 7
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end = datetime(year + (month == 7), 7 if month == 1 else 1, 1, tzinfo=timezone.utc)
    return {"fold_id": f"development-{number:02d}", "fit_cutoff": start.isoformat(),
            "validation_start": start.isoformat(), "validation_end": end.isoformat()}


def row(fid, kickoff, *, home=1, away=2):
    kickoff = datetime.fromisoformat(kickoff) if isinstance(kickoff, str) else kickoff
    return {"fixture": {"fixture_id": fid, "kickoff": kickoff.isoformat(), "competition": "EPL", "season": kickoff.year},
        "target": float(home + away), "labels": {market: float(home + away) for market in extension.MARKETS},
        "team_labels": {side: {market: float(value) for market in extension.MARKETS} for side, value in (("home", home), ("away", away))},
        "label_available_at": (kickoff + timedelta(hours=3)).isoformat(), "values": [5., 5., None, 0., 0., 1.],
        "snapshot_id": digest(fid), "feature_contract_id": FEATURE_ID,
        "market_eligibility": {market: {"eligible": True} for market in extension.MARKETS},
        "support": {market: {side: {"count": 5} for side in ("home", "away")} for market in extension.MARKETS},
        "source_class": "verified_local_reconstruction", "round_group": "domestic_regular",
        "forecast_stage": "reconstructed_immediately_before_kickoff"}


def baselines(rows):
    return {(r["fixture"]["fixture_id"], market): {"fixture_id": r["fixture"]["fixture_id"], "market": market,
        "snapshot_id": r["snapshot_id"], "feature_contract_id": FEATURE_ID,
        "statistical": 5., "league_average": 4.} for r in rows for market in extension.MARKETS}


class SyntheticComponent:
    """One supplied-array estimator stand-in; deliberately no learned model."""
    def __init__(self, kind, y):
        self.kind, self.mean = kind, float(np.mean(y))
        self.metadata = {"synthetic_test": True, "kind": kind}

    def predict_with_diagnostics(self, X, *, names, statistical=None):
        assert (statistical is not None) == (self.kind in extension_models.ANCHORED)
        values = np.full(len(X), self.mean + (1. if self.kind in extension_models.ANCHORED else 0.))
        return {"values": values, "raw_values": values.copy(), "clipped_count": 0,
                "raw_correction": values - statistical if statistical is not None else np.zeros(len(X))}

    def predict(self, X, *, names, statistical=None):
        return self.predict_with_diagnostics(X, names=names, statistical=statistical)["values"]


def synthetic_fit(calls):
    def fit(kind, X, y, weights, names, market, *, statistical=None):
        calls.append({"kind": kind, "market": market, "target": np.asarray(y).copy(),
                      "statistical": None if statistical is None else np.asarray(statistical).copy()})
        return SyntheticComponent(kind, y)
    return fit


def component_inputs():
    f = fold(1)
    cutoff = datetime.fromisoformat(f["fit_cutoff"])
    train = [row(1, cutoff - timedelta(days=30), home=1, away=2), row(2, cutoff - timedelta(days=20), home=2, away=4)]
    validation = [row(3, cutoff + timedelta(days=1))]
    weights, support = extension._support(train, validation, f["fit_cutoff"])
    return {"market": "goals", "fold": f, "train": train, "validation": validation, "weights": weights,
            "support": support, "names": NAMES, "baselines": baselines(train + validation), "baseline_id": digest("baseline")}


@pytest.mark.parametrize("kind,side,expected", [("residual_ridge", "total", [3., 6.]), ("offset_xgboost", "total", [3., 6.]),
    ("team_poisson", "home", [1., 2.]), ("team_poisson", "away", [2., 4.]),
    ("team_catboost", "home", [1., 2.]), ("team_catboost", "away", [2., 4.])])
def test_component_routes_counts_and_anchors_then_reuses_one_checkpoint(tmp_path, monkeypatch, kind, side, expected):
    # Numerical support gates have independent tests; this tiny fixture targets
    # orchestration and never fits a numerical estimator.
    monkeypatch.setattr(extension, "family_gate", lambda *a: {"qualified": True})
    calls = []
    monkeypatch.setattr(extension_models, "fit_component", synthetic_fit(calls))
    inputs = component_inputs()
    store = checkpoints.Checkpoints(tmp_path / "tasks", "synthetic", max_fits=extension.MAX_FITS)
    first = extension._component_task(store, kind=kind, side=side, **inputs)
    resumed = checkpoints.Checkpoints(tmp_path / "tasks", "synthetic", max_fits=extension.MAX_FITS)
    second = extension._component_task(resumed, kind=kind, side=side, **inputs)
    assert len(calls) == 1 and calls[0]["target"].tolist() == expected
    assert calls[0]["statistical"] is None if side != "total" else calls[0]["statistical"].tolist() == [5., 5.]
    assert first["status"] == second["status"] == "complete"
    assert np.array_equal(first["predictions"], second["predictions"])
    assert resumed.created == 0 and resumed.reused == 1
    attempt = next((tmp_path / "tasks").glob("*/attempt-*"))
    assert read_json(attempt / "training.json")["targets"] == expected
    assert read_json(attempt / "validation.json")["fixture_ids"] == [3]
    assert len(list((tmp_path / "tasks").glob("*/attempt-*"))) == 1


def test_component_rejects_wrong_side_and_late_labels_before_estimator_call(tmp_path, monkeypatch):
    monkeypatch.setattr(extension, "family_gate", lambda *a: {"qualified": True})
    monkeypatch.setattr(extension_models, "fit_component", lambda *a, **k: pytest.fail("invalid training cannot fit"))
    inputs = component_inputs()
    store = checkpoints.Checkpoints(tmp_path / "tasks", "synthetic")
    with pytest.raises(ValueError, match="Wrong component"):
        extension._component_task(store, kind="team_poisson", side="total", **inputs)
    inputs["train"][0]["label_available_at"] = inputs["fold"]["fit_cutoff"]
    with pytest.raises(ValueError, match="cutoff"):
        extension._component_task(store, kind="residual_ridge", side="total", **inputs)
    assert store.created == 0


def test_common_cohort_excludes_missing_or_nonpositive_baselines_for_every_architecture():
    rows = [row(i, datetime(2022, 1, i, tzinfo=timezone.utc)) for i in range(1, 5)]
    source = baselines(rows)
    del source[(2, "goals")]
    source[(3, "goals")]["statistical"] = 0.
    source[(4, "goals")]["statistical"] = float("nan")
    kept, excluded = extension._common(rows, source, "goals")
    assert [r["fixture"]["fixture_id"] for r in kept] == [1]
    assert excluded == [{"fixture_id": 2, "reason": "missing_statistical_baseline"},
                        {"fixture_id": 3, "reason": "nonpositive_or_invalid_statistical_baseline"},
                        {"fixture_id": 4, "reason": "nonpositive_or_invalid_statistical_baseline"}]


@pytest.mark.parametrize("change", ["team_null", "team_fraction", "team_total", "baseline_snapshot", "baseline_market", "baseline_contract"])
def test_common_cohort_integrity_errors_cannot_be_hidden_as_coverage_exclusions(change):
    rows = [row(1, datetime(2022, 1, 1, tzinfo=timezone.utc))]
    source = baselines(rows)
    if change.startswith("team_"):
        rows[0]["team_labels"]["home"]["goals"] = {"team_null": None, "team_fraction": .5, "team_total": 4}[change]
    else:
        source[(1, "goals")][{"baseline_snapshot": "snapshot_id", "baseline_market": "market", "baseline_contract": "feature_contract_id"}[change]] = "different"
    with pytest.raises(ValueError):
        extension._common(rows, source, "goals")


@pytest.mark.parametrize("field", ["snapshot_id", "target", "competition", "season", "forecast_stage", "missingness_band", "kickoff"])
def test_saved_batch_c_comparator_requires_identical_fixture_evidence(field):
    f = fold(3)
    rows = [row(1, datetime.fromisoformat(f["fit_cutoff"]) + timedelta(days=1))]
    reference = extension._rows(rows, [2.5], "goals", f, "selected_ml", NAMES)[0]
    old = {("goals", f["fold_id"], "selected_ml", 1): reference}
    assert extension._old_values(old, rows, "goals", f, "selected_ml", NAMES).tolist() == [2.5]
    reference[field] = "2023-01-03T00:00:00Z" if field == "kickoff" else 999 if field in {"target", "season"} else "different"
    with pytest.raises(ValueError, match="identical"):
        extension._old_values(old, rows, "goals", f, "selected_ml", NAMES)


class ForbiddenLabels(dict):
    def __getitem__(self, key):
        pytest.fail("Reserved-period targets must not be read")


class SyntheticDataset:
    def __init__(self):
        self.folds = [fold(i) for i in range(1, 5)]
        self.schema = {"names": NAMES}
        self.manifest = {"feature_contract_id": FEATURE_ID, "dataset_id": digest("synthetic-dataset")}
        self.splits = {"boundaries": {"phase3_confirmation_start": "2024-01-01T00:00:00+00:00"}}
        self.rows = [row(1, datetime(2021, 10, 1, tzinfo=timezone.utc)), row(2, datetime(2021, 11, 1, tzinfo=timezone.utc))]
        self.validation = {}
        for index, f in enumerate(self.folds, 1):
            start = datetime.fromisoformat(f["fit_cutoff"])
            rows = [row(index * 100 + offset, start + timedelta(days=offset)) for offset in (1, 2)]
            self.rows.extend(rows)
            self.validation[f["fold_id"]] = rows
        self.regular_rows = list(self.rows)
        for index, year in enumerate((2024, 2025, 2026), 1):
            reserved = row(9000 + index, datetime(year, 1, 2, tzinfo=timezone.utc))
            reserved["labels"] = reserved["team_labels"] = ForbiddenLabels()
            self.rows.append(reserved)

    def select_fold(self, market, fold_id, *, lookback_days, half_life_days):
        assert (lookback_days, half_life_days) == (1460, 365)
        f = next(f for f in self.folds if f["fold_id"] == fold_id)
        cutoff = datetime.fromisoformat(f["fit_cutoff"])
        train = [{**r, "target": r["labels"][market]} for r in self.rows
                 if datetime.fromisoformat(r["label_available_at"]) < cutoff]
        validation = [{**r, "target": r["labels"][market]} for r in self.validation[fold_id]]
        weights, support = extension._support(train, validation, f["fit_cutoff"])
        return train, validation, weights, support


def test_complete_execution_routes_72_development_and_six_team_refits_without_reserved_labels(tmp_path, monkeypatch):
    data = SyntheticDataset()
    source = baselines(data.regular_rows)
    calls = []
    monkeypatch.setattr(extension_models, "fit_component", synthetic_fit(calls))
    monkeypatch.setattr(extension, "family_gate", lambda *a: {"qualified": True})
    monkeypatch.setattr(extension, "source_hashes", lambda: {"synthetic_test": "frozen"})
    monkeypatch.setattr(evaluation, "summarize", lambda rows: {"synthetic_test": True, "rows": len(rows)})
    monkeypatch.setattr(evaluation, "paired_comparison", lambda *a, **k: {"synthetic_test": True})
    bundles = {}
    def save(path, components, metadata):
        bundles[str(path)] = (components, metadata)
        path.mkdir()
        checkpoints.atomic_write_json(path / "synthetic-bundle.json", metadata)
    def predict(path, rows, *, feature_contract_id, statistical_rows, baseline_contract_id):
        components, metadata = bundles[str(path)]
        assert feature_contract_id == FEATURE_ID and baseline_contract_id == metadata["baseline_contract_id"]
        assert {r["fixture_id"] for r in statistical_rows} == {r["fixture"]["fixture_id"] for r in rows}
        means = sum((np.full(len(rows), c.mean) for c in components.values()), np.zeros(len(rows)))
        return (1 - metadata["weight_ml"]) * np.asarray([r["statistical"] for r in statistical_rows]) + metadata["weight_ml"] * means
    monkeypatch.setattr(extension_bundle, "save_bundle", save)
    monkeypatch.setattr(extension_bundle, "predict_bundle", predict)
    old = {}
    for market in extension.MARKETS:
        for f in data.folds[2:]:
            for method in ("selected_ml", "selected_blend"):
                for reference in extension._rows(data.validation[f["fold_id"]], [4., 4.], market, f, method, NAMES):
                    old[(market, f["fold_id"], method, reference["fixture_id"])] = reference
    target = tmp_path / "extension"
    target.mkdir()
    store = checkpoints.Checkpoints(target / "tasks", "synthetic-specification", max_fits=extension.MAX_FITS)
    result = extension._execute(data, source, {"synthetic_baseline": True}, old, target, store, "synthetic-specification")
    assert extension.MAX_FITS == 78 and store.created == len(calls) == 78
    assert Counter(c["kind"] for c in calls) == {"residual_ridge": 12, "offset_xgboost": 12, "team_poisson": 30, "team_catboost": 24}
    assert all(np.all(call["target"] == (3. if call["kind"] in extension_models.ANCHORED else call["target"][0])) for call in calls)
    report = target / result["report_directory"]
    verify_complete(report)
    tasks = read_json(report / "tasks.json")
    assert len(tasks) == 72 and all(t["status"] == "complete" for t in tasks)
    choices = read_json(report / "choices.json")
    proposals = read_json(report / "frozen-proposals.json")
    for market in extension.MARKETS:
        assert choices[market]["development-01"]["kind"] == "residual_ridge"
        assert choices[market]["development-02"]["kind"] == "team_poisson"
        assert choices[market]["development-03"]["prior_fold_ids"] == ["development-01", "development-02"]
        assert proposals[market]["final_components"] == 2 and proposals[market]["serialization_parity"] is True
        assert proposals[market]["publication_enabled"] is False and proposals[market]["promotion_allowed"] is False
    for _, metadata in bundles.values():
        assert metadata["fit_cutoff"] == "2024-01-01T00:00:00+00:00"
        assert metadata["support"]["training_unique_fixtures"] == len(data.regular_rows)
    for artifact in ("architecture-oof.jsonl", "adaptive-oof.jsonl", "outer-predictions.jsonl"):
        records = [json.loads(line) for line in (report / artifact).read_text().splitlines()]
        assert all(r["fixture_id"] < 9000 for r in records)
    assert checkpoints.verify_committed_tasks(target / "tasks", "synthetic-specification")["committed_tasks"] == 78
