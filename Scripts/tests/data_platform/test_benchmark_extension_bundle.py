"""Synthetic trusted-bundle parity, isolation and equivalent-input checks."""
from copy import deepcopy
import json

import numpy as np
import pytest

from Scripts.data_platform.features.benchmarks import artifacts
from Scripts.data_platform.features.benchmarks import extension_bundle as bundles
from Scripts.data_platform.features.benchmarks import extension_models as models
from Scripts.rag_ingest.core.model_features import digest
from Scripts.tests.data_platform.test_benchmark_extension_models import fitted_components


def _inputs(fitted_components, kind):
    fitted, values, _, _, names, statistical = fitted_components
    model = fitted[kind]
    components = {"total": model} if kind in models.ANCHORED else {"home": model, "away": model}
    metadata = {"kind": kind, "market": "goals", "names": list(names),
                "feature_contract_id": model.metadata["feature_contract_id"],
                "baseline_contract": {"version": "synthetic-core.v1"},
                "baseline_contract_id": digest({"version": "synthetic-core.v1"}), "weight_ml": .25}
    rows = [{"fixture": {"fixture_id": i + 1}, "snapshot_id": f"{i + 1:064x}",
             "feature_contract_id": metadata["feature_contract_id"],
             "values": [None if np.isnan(value) else float(value) for value in row]}
            for i, row in enumerate(values[:5])]
    baselines = [{"fixture_id": row["fixture"]["fixture_id"], "snapshot_id": row["snapshot_id"],
                  "feature_contract_id": metadata["feature_contract_id"], "market": "goals",
                  "statistical": float(value)} for row, value in zip(rows, statistical)]
    return components, metadata, rows, baselines


def _predict(path, metadata, rows, baselines, **changes):
    kwargs = {"feature_contract_id": metadata["feature_contract_id"], "statistical_rows": baselines,
              "baseline_contract_id": metadata["baseline_contract_id"], **changes}
    return bundles.predict_bundle(path, rows, **kwargs)


def _change_manifest(path, mutate):
    manifest = artifacts.read_json(path / "manifest.json")
    mutate(manifest)
    (path / "manifest.json").write_text(json.dumps(manifest))
    complete = artifacts.read_json(path / "COMPLETE.json")
    complete["manifest.json"] = artifacts.sha(path / "manifest.json")
    (path / "COMPLETE.json").write_text(json.dumps(complete))


@pytest.mark.parametrize("kind", models.KINDS)
@pytest.mark.parametrize("weight", [0., .25, 1.])
def test_all_architectures_exact_frozen_strategy_replay(fitted_components, tmp_path, kind, weight):
    components, metadata, rows, baselines = _inputs(fitted_components, kind)
    metadata["weight_ml"] = weight
    path = bundles.save_bundle(tmp_path / "bundle", components, metadata)
    actual = _predict(path, metadata, rows, list(reversed(baselines)))
    _, values, _, _, names, statistical = fitted_components
    predicted = sum(model.predict(values[:5], names=names,
                    **({"statistical": statistical[:5]} if kind in models.ANCHORED else {}))
                    for model in components.values())
    assert np.array_equal(actual, (1 - weight) * statistical[:5] + weight * predicted)
    saved = artifacts.read_json(path / "manifest.json")
    assert saved["qualification"] == bundles.QUALIFICATION
    assert saved["publication_enabled"] is False and saved["promotion_allowed"] is False
    assert saved["dependencies"] == artifacts.dependencies()
    assert saved["source_hashes"] and set(saved["components"]) == set(components)
    with pytest.raises(FileExistsError):
        bundles.save_bundle(path, components, metadata)


@pytest.mark.parametrize("kind", models.KINDS)
def test_wrong_component_set_or_kind_rejected_before_directory_creation(fitted_components, tmp_path, kind):
    components, metadata, _, _ = _inputs(fitted_components, kind)
    path = tmp_path / "invalid"
    with pytest.raises(ValueError, match="component set"):
        bundles.save_bundle(path, {"other": next(iter(components.values()))}, metadata)
    other_kind = "team_poisson" if kind in models.ANCHORED else "residual_ridge"
    wrong = {name: fitted_components[0][other_kind] for name in components}
    with pytest.raises(ValueError, match="component identity"):
        bundles.save_bundle(path, wrong, metadata)
    assert not path.exists()


@pytest.mark.parametrize("field,value", [
    ("weight_ml", True), ("weight_ml", -1), ("weight_ml", 1.1), ("weight_ml", float("nan")),
    ("market", "cards"), ("qualification", "QUALIFIED"), ("publication_enabled", True),
    ("promotion_allowed", True), ("feature_contract_id", "b" * 64), ("baseline_contract_id", "b" * 64),
])
def test_invalid_saved_contract_is_rejected(fitted_components, tmp_path, field, value):
    components, metadata, _, _ = _inputs(fitted_components, "residual_ridge")
    with pytest.raises(ValueError):
        bundles.save_bundle(tmp_path / "bad", components, {**metadata, field: value})


@pytest.mark.parametrize("field,value", [
    ("version", "wrong"), ("qualification", "QUALIFIED"), ("python", "0.0"),
    ("dependencies", {}), ("promotion_allowed", True), ("combination", "multiply_probabilities"),
])
def test_manifest_contract_failure_precedes_unpickling(fitted_components, tmp_path, monkeypatch, field, value):
    components, metadata, rows, baselines = _inputs(fitted_components, "residual_ridge")
    path = bundles.save_bundle(tmp_path / "bundle", components, metadata)
    _change_manifest(path, lambda m: m.update({field: value}))
    import joblib
    monkeypatch.setattr(joblib, "load", lambda *a, **kw: pytest.fail("Invalid bundle must never unpickle"))
    with pytest.raises(ValueError):
        _predict(path, metadata, rows, baselines)


def test_checksum_failure_precedes_unpickling(fitted_components, tmp_path, monkeypatch):
    components, metadata, rows, baselines = _inputs(fitted_components, "residual_ridge")
    path = bundles.save_bundle(tmp_path / "bundle", components, metadata)
    with (path / "components.joblib").open("ab") as handle:
        handle.write(b"corrupt")
    import joblib
    monkeypatch.setattr(joblib, "load", lambda *a, **kw: pytest.fail("Corrupt bundle must never unpickle"))
    with pytest.raises(ValueError, match="checksum"):
        _predict(path, metadata, rows, baselines)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "extra", "snapshot", "market", "contract", "negative", "nan"])
def test_inequivalent_baselines_rejected_before_unpickling(fitted_components, tmp_path, monkeypatch, mutation):
    components, metadata, rows, baselines = _inputs(fitted_components, "residual_ridge")
    path = bundles.save_bundle(tmp_path / "bundle", components, metadata)
    if mutation == "duplicate":
        baselines.append(baselines[0])
    elif mutation == "missing":
        baselines.pop()
    elif mutation == "extra":
        baselines.append({**baselines[0], "fixture_id": 99})
    else:
        field, value = {"snapshot": ("snapshot_id", "f" * 64), "market": ("market", "corners"),
                        "contract": ("feature_contract_id", "f" * 64), "negative": ("statistical", -1.),
                        "nan": ("statistical", float("nan"))}[mutation]
        baselines[0][field] = value
    import joblib
    monkeypatch.setattr(joblib, "load", lambda *a, **kw: pytest.fail("Invalid baseline must never unpickle"))
    with pytest.raises(ValueError):
        _predict(path, metadata, rows, baselines)


@pytest.mark.parametrize("mutation", ["duplicate_fixture", "duplicate_snapshot", "bad_id", "bad_snapshot", "contract", "names", "shape", "bool", "infinite", "missingness"])
def test_invalid_rows_rejected_before_unpickling(fitted_components, tmp_path, monkeypatch, mutation):
    components, metadata, rows, baselines = _inputs(fitted_components, "residual_ridge")
    path = bundles.save_bundle(tmp_path / "bundle", components, metadata)
    if mutation == "duplicate_fixture":
        rows[1]["fixture"]["fixture_id"] = rows[0]["fixture"]["fixture_id"]
    elif mutation == "duplicate_snapshot":
        rows[1]["snapshot_id"] = rows[0]["snapshot_id"]
    elif mutation == "bad_id":
        rows[0]["fixture"]["fixture_id"] = True
    elif mutation == "bad_snapshot":
        rows[0]["snapshot_id"] = "not-a-snapshot"
    elif mutation == "contract":
        rows[0]["feature_contract_id"] = "f" * 64
    elif mutation == "names":
        rows[0]["names"] = list(reversed(metadata["names"]))
    elif mutation == "shape":
        rows[0]["values"].pop()
    elif mutation == "bool":
        rows[0]["values"][0] = True
    elif mutation == "infinite":
        rows[0]["values"][0] = float("inf")
    else:
        index = metadata["names"].index("home_xg_pm__missing")
        rows[0]["values"][index] = 0.
    import joblib
    monkeypatch.setattr(joblib, "load", lambda *a, **kw: pytest.fail("Invalid rows must never unpickle"))
    with pytest.raises(ValueError):
        _predict(path, metadata, rows, baselines)


def test_zero_offset_is_rejected_but_zero_residual_anchor_allowed(fitted_components, tmp_path):
    for kind in models.ANCHORED:
        components, metadata, rows, baselines = _inputs(fitted_components, kind)
        path = bundles.save_bundle(tmp_path / kind, components, metadata)
        baselines[0]["statistical"] = 0.
        if kind == "offset_xgboost":
            with pytest.raises(ValueError, match="nonpositive offset"):
                _predict(path, metadata, rows, baselines)
        else:
            assert np.isfinite(_predict(path, metadata, rows, baselines)).all()


def test_serialized_component_metadata_and_contract_ids_are_bound(fitted_components, tmp_path):
    components, metadata, rows, baselines = _inputs(fitted_components, "residual_ridge")
    path = bundles.save_bundle(tmp_path / "bundle", components, metadata)
    for key in ("feature_contract_id", "baseline_contract_id"):
        with pytest.raises(ValueError, match="contract"):
            _predict(path, metadata, rows, baselines, **{key: "f" * 64})
    _change_manifest(path, lambda m: m["components"]["total"].update(training_rows=9999))
    with pytest.raises(ValueError, match="Serialized extension component"):
        _predict(path, metadata, rows, baselines)


def test_active_model_destination_and_symlinks_are_rejected(fitted_components, tmp_path, monkeypatch):
    components, metadata, _, _ = _inputs(fitted_components, "residual_ridge")
    root = tmp_path / "repo"
    monkeypatch.setattr(artifacts, "ROOT", root)
    with pytest.raises(ValueError, match="isolated experiment"):
        bundles.save_bundle(root / "Index/models/model", components, metadata)
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        bundles.save_bundle(tmp_path / "link/candidate", components, metadata)


@pytest.mark.parametrize("mutation", ["preprocessor", "params", "version", "feature_contract", "estimator"])
def test_fixed_component_contract_cannot_be_relabelled(fitted_components, tmp_path, mutation):
    components, metadata, _, _ = _inputs(fitted_components, "residual_ridge")
    model = deepcopy(components["total"])
    if mutation == "preprocessor":
        model.preprocessor.mean[0] += 1
    elif mutation == "params":
        model.metadata["params"]["alpha"] += 1
    elif mutation == "version":
        model.metadata["version"] = "different-model"
    elif mutation == "feature_contract":
        model.metadata["feature_contract_id"] = "f" * 64
    else:
        model.estimator.set_params(alpha=999)
    path = tmp_path / "invalid"
    with pytest.raises(ValueError, match="component identity"):
        bundles.save_bundle(path, {"total": model}, metadata)
    assert not path.exists()
