"""Runner boundaries and immutable artifacts; never touch real project state."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import socket
import sqlite3

import numpy as np
import pytest

from Scripts.data_platform.features.benchmarks import artifacts, estimators, runner
from Scripts.data_platform.features.benchmarks.isolation import offline_guard
from Scripts.tests.data_platform.test_benchmark_data import synthetic, save
from Scripts.tests.data_platform.test_benchmark_estimators import _training


def test_offline_guard_blocks_protected_io_in_main_and_worker_threads(tmp_path):
    root = tmp_path / "project"
    output = root / "Index/prediction_experiments/test"
    output.mkdir(parents=True)
    protected = root / "Index/protected.json"
    protected.write_text("preserved")
    research = root / "Index/research.json"
    research.write_text("permitted")
    with ThreadPoolExecutor(max_workers=1) as executor:
        with offline_guard(root=root, output=output, readable_files=[research]):
            assert research.read_text() == "permitted"
            (output / "result.json").write_text("result")
            with pytest.raises(RuntimeError, match="protected data"):
                protected.read_text()
            with pytest.raises(RuntimeError, match="protected data"):
                executor.submit(protected.read_text).result()
            with pytest.raises(RuntimeError, match="socket"):
                executor.submit(socket.create_connection, ("127.0.0.1", 9)).result()
            with pytest.raises(RuntimeError, match="sqlite3"):
                sqlite3.connect(":memory:")
            with pytest.raises(RuntimeError, match="write outside"):
                protected.write_text("wrong")
    assert protected.read_text() == "preserved"
    assert (output / "result.json").read_text() == "result"


def test_nested_guards_restore_prior_policy_even_after_failure(tmp_path):
    output = tmp_path / "Index/prediction_experiments/test"
    output.mkdir(parents=True)
    first, second = tmp_path / "Index/a", tmp_path / "Index/b"
    first.write_text("a"); second.write_text("b")
    with offline_guard(root=tmp_path, output=output, readable_files=[first]):
        with pytest.raises(RuntimeError):
            with offline_guard(root=tmp_path, output=output, readable_files=[second]):
                assert second.read_text() == "b"
                first.read_text()
        assert first.read_text() == "a"
        with pytest.raises(RuntimeError):
            second.read_text()
    assert second.read_text() == "b"


def test_top_level_manifest_binds_nested_completion_and_rejects_escapes(tmp_path):
    child = tmp_path / "candidate"
    child.mkdir()
    (child / "model").write_text("fitted")
    artifacts.complete(child)
    artifacts.complete(tmp_path)
    assert "candidate/COMPLETE.json" in artifacts.verify_complete(tmp_path)
    (child / "COMPLETE.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        artifacts.verify_complete(tmp_path)
    (tmp_path / "COMPLETE.json").write_text(json.dumps({"../secret": "0" * 64}))
    with pytest.raises(ValueError, match="Unsafe"):
        artifacts.verify_complete(tmp_path)


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "dependencies", lambda: {"test-environment": "fixed"})
    x, y, weights, names = _training(count=80)
    model = estimators.fit_estimator("ridge", x, y, weights, names, "goals")
    rows = [{"snapshot_id": f"{i + 1:064x}", "feature_contract_id": "a" * 64,
             "values": [None if np.isnan(v) else float(v) for v in row]} for i, row in enumerate(x[:4])]
    metadata = {"market": "goals", "names": list(names), "feature_contract_id": "a" * 64}
    location = tmp_path / "candidate"
    artifacts.save_candidate(location, model, metadata=metadata)
    return location, model, rows, metadata


def test_bundle_saved_transform_reuse_contract_and_missingness(candidate):
    path, model, rows, metadata = candidate
    before = deepcopy(model.metadata)
    actual = artifacts.predict_candidate(path, rows, feature_contract_id="a" * 64)
    expected = model.predict(np.asarray([[np.nan if v is None else v for v in r["values"]] for r in rows]),
                             names=metadata["names"])
    assert np.array_equal(actual, expected)
    assert model.metadata == before
    with pytest.raises(ValueError, match="contract"):
        artifacts.predict_candidate(path, rows, feature_contract_id="b" * 64)
    broken = deepcopy(rows)
    broken[0]["values"] = broken[0]["values"][:-1]
    with pytest.raises(ValueError, match="vector"):
        artifacts.predict_candidate(path, broken, feature_contract_id="a" * 64)


def test_mislabeled_fitted_model_rejected_on_save_and_load(candidate, tmp_path):
    path, model, rows, metadata = candidate
    with pytest.raises(ValueError, match="identity"):
        artifacts.save_candidate(tmp_path / "wrong", model, metadata={**metadata, "market": "corners"})
    stored = artifacts.read_json(path / "manifest.json")
    stored["market"] = "corners"
    (path / "manifest.json").write_text(json.dumps(stored))
    completion = artifacts.read_json(path / "COMPLETE.json")
    completion["manifest.json"] = artifacts.sha(path / "manifest.json")
    (path / "COMPLETE.json").write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="metadata mismatch"):
        artifacts.predict_candidate(path, rows, feature_contract_id="a" * 64)


def test_checksum_failure_precedes_pickle_loading(candidate, monkeypatch):
    path, _, rows, _ = candidate
    with (path / "model.joblib").open("ab") as handle:
        handle.write(b"changed")
    import joblib
    monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("corrupt pickle must never load"))
    with pytest.raises(ValueError, match="checksum"):
        artifacts.predict_candidate(path, rows, feature_contract_id="a" * 64)


def test_new_experiments_cannot_overwrite_or_escape(tmp_path):
    made = artifacts.new_experiment(tmp_path, "first")
    with pytest.raises(FileExistsError):
        artifacts.new_experiment(tmp_path, "first")
    with pytest.raises(ValueError):
        artifacts.new_experiment(tmp_path, "../escape")
    assert made.exists()


def _runner_inputs(tmp_path):
    root = tmp_path / "project"
    dataset = root / "Index/prediction_experiments/dataset"
    dataset.parent.mkdir(parents=True)
    docs = synthetic()
    save(dataset, docs)
    sidecar = dataset.parent / "baselines"
    sidecar.mkdir()
    manifest = docs["manifest.json"]
    records = [{"fixture_id": row["fixture"]["fixture_id"], "market": market,
                "snapshot_id": row["snapshot_id"], "feature_contract_id": manifest["feature_contract_id"],
                "league_average": 2.0, "statistical": 2.1}
               for row in docs["development/features.jsonl"] for market in runner.MARKETS
               if row["market_eligibility"][market]["eligible"]]
    artifacts.write_json(sidecar / "manifest.json", {"version": artifacts.BASELINE_VERSION,
        "scope": "development_only", "dataset_id": manifest["dataset_id"],
        "feature_contract_id": manifest["feature_contract_id"], "publication_enabled": False,
        "rows": len(records), "baseline": {"statistical_version": "synthetic_test"}})
    artifacts.write_json(sidecar / "preparation.json", {"synthetic_test": True})
    (sidecar / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    artifacts.complete(sidecar)
    return root, dataset, sidecar


def test_runner_reports_every_insufficient_family_without_relaxing_gates(tmp_path, monkeypatch):
    root, dataset, sidecar = _runner_inputs(tmp_path)
    monkeypatch.setattr(runner, "environment_report", lambda: {"synthetic_test": True})
    monkeypatch.setattr(runner, "fit_estimator", lambda *a, **k: pytest.fail("insufficient cohort cannot fit"))
    path = runner.run_smoke(root=root, dataset=dataset, baselines=sidecar, name="smoke", markets=["goals"])
    report = artifacts.read_json(path / "report.json")
    assert report["status"] == "incomplete"
    assert not (path / "COMPLETE.json").exists()
    assert report["confirmation_opened"] is False
    assert len(report["failures"]) == len(runner.FAMILIES)
    methods = report["markets"]["goals"]["methods"]
    assert all(methods[family]["status"] == "insufficient_support" for family in runner.FAMILIES)
    assert methods["statistical"]["status"] == "complete"


def test_runner_keeps_failed_models_visible_and_production_files_intact(tmp_path, monkeypatch):
    root, dataset, sidecar = _runner_inputs(tmp_path)
    active = root / "Index/ml_models/active.bin"
    active.parent.mkdir()
    active.write_bytes(b"preserve")
    monkeypatch.setattr(runner, "environment_report", lambda: {"synthetic_test": True})
    # Tiny synthetic integration only: real qualification rules are exercised
    # above and in test_benchmark_data. No actual estimator is fitted here.
    monkeypatch.setattr(runner, "family_gate", lambda *a: {"qualified": True})
    def fail(*args, **kwargs):
        raise estimators.EstimatorFitError("deliberate fit failure")
    monkeypatch.setattr(runner, "fit_estimator", fail)
    path = runner.run_smoke(root=root, dataset=dataset, baselines=sidecar, name="failed", markets=["goals"])
    report = artifacts.read_json(path / "report.json")
    assert len(report["failures"]) == 5
    assert all(item["status"] == "failed" for item in report["failures"])
    assert active.read_bytes() == b"preserve"
    assert not (path / "COMPLETE.json").exists()

