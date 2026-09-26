"""Explicit, unconfirmed extension bundles; no active-model or provider access.

Joblib is only suitable for these trusted local research artifacts. Checksums
are an integrity check, not authentication for an externally supplied pickle.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
import sys

import numpy as np

from Scripts.rag_ingest.core.model_features import digest
from . import artifacts
from .checkpoints import atomic_write_json, complete_atomic
from .data import expected_schema

VERSION = "phase3-extension-bundle.v1"
QUALIFICATION = "UNCONFIRMED_BATCH_C_EXTENSION"
_HASH = re.compile(r"[0-9a-f]{64}")
_KINDS = {
    "residual_ridge": ("ridge", {"total"}),
    "offset_xgboost": ("xgboost", {"total"}),
    "team_poisson": ("poisson", {"home", "away"}),
    "team_catboost": ("catboost", {"home", "away"}),
}


def _hash(value):
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _number(value, *, positive=False):
    try:
        return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        return False


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _metadata(metadata):
    if not isinstance(metadata, dict) or metadata.get("kind") not in _KINDS:
        raise ValueError("Unsupported extension kind")
    if (metadata.get("market") not in {"goals", "corners", "sot"}
            or metadata.get("names") != expected_schema()["names"]
            or metadata.get("feature_contract_id") != digest(expected_schema())
            or not _hash(metadata.get("baseline_contract_id"))):
        raise ValueError("Extension feature/baseline identity contract mismatch")
    if "baseline_contract" in metadata and digest(metadata["baseline_contract"]) != metadata["baseline_contract_id"]:
        raise ValueError("Extension baseline contract hash mismatch")
    if not _number(metadata.get("weight_ml")) or metadata["weight_ml"] > 1:
        raise ValueError("Invalid frozen extension blend weight")
    for key, value in (("qualification", QUALIFICATION), ("publication_enabled", False), ("promotion_allowed", False)):
        if key in metadata and (metadata[key] != value or isinstance(value, bool) and metadata[key] is not value):
            raise ValueError("Extension bundle must remain unconfirmed and isolated")


def _components(components, metadata):
    from . import estimators
    from .extension_models import FittedExtensionComponent, FIXED_PARAMS, VERSION as COMPONENT_VERSION

    family, names = _KINDS[metadata["kind"]]
    if not isinstance(components, dict) or set(components) != names:
        raise ValueError("Incorrect extension component set")
    for model in components.values():
        if (not isinstance(model, FittedExtensionComponent)
                or model.kind != metadata["kind"] or model.family != family
                or model.market != metadata["market"]
                or list(model.preprocessor.names) != metadata["names"]
                or model.metadata.get("kind") != model.kind
                or model.metadata.get("family") != model.family
                or model.metadata.get("market") != model.market
                or model.metadata.get("names") != metadata["names"]
                or model.metadata.get("feature_contract_id") != metadata["feature_contract_id"]
                or model.metadata.get("version") != COMPONENT_VERSION
                or model.metadata.get("params") != FIXED_PARAMS[model.kind]
                or model.metadata.get("preprocessing_version") != estimators.PREPROCESSING_VERSION
                or model.metadata.get("preprocessing") != model.preprocessor.metadata()
                or model.metadata.get("resolved_params") != estimators._parameter_metadata(model.estimator.get_params(deep=False))
                or model.metadata.get("qualification") != QUALIFICATION
                or model.metadata.get("publication_enabled") is not False
                or model.metadata.get("promotion_allowed") is not False):
            raise ValueError("Extension component identity differs from bundle")


def _path(path, *, saving=False):
    path = Path(path).absolute()
    if path.is_symlink() or path.resolve() != path:
        raise ValueError("Extension bundle path cannot contain symlinks")
    if (saving and path.is_relative_to(artifacts.ROOT / "Index")
            and not path.is_relative_to(artifacts.ROOT / "Index/prediction_experiments")):
        raise ValueError("Extension destination must be an isolated experiment")
    return path


def save_bundle(path, components, metadata):
    """Commit a new trusted bundle without overwriting any existing artifact."""
    import joblib

    _metadata(metadata)
    _components(components, metadata)
    path = _path(path, saving=True)
    actual = {**metadata, "version": VERSION, "qualification": QUALIFICATION,
              "publication_enabled": False, "promotion_allowed": False,
              "dependencies": artifacts.dependencies(), "python": sys.version.split()[0],
              "source_hashes": artifacts.source_hashes(),
              "components": {key: model.metadata for key, model in components.items()},
              "combination": "total_mean_then_convex_statistical_blend",
              "output_scale": "expected_count", "rounding": "none"}
    path.mkdir(parents=True, exist_ok=False)
    joblib.dump(components, path / "components.joblib", compress=3)
    atomic_write_json(path / "manifest.json", actual)
    complete_atomic(path)
    return path


def _inputs(rows, statistical_rows, metadata, feature_contract_id):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("Extension inference requires nonempty fixture rows")
    names = metadata["names"]
    fixtures, snapshots, values = [], set(), []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid extension inference row")
        fixture = row.get("fixture", {})
        fid = fixture.get("fixture_id") if isinstance(fixture, dict) else None
        snapshot = row.get("snapshot_id")
        vector = row.get("values")
        if (type(fid) is not int or fid <= 0 or fid in fixtures
                or not _hash(snapshot) or snapshot in snapshots
                or row.get("feature_contract_id") != feature_contract_id
                or ("names" in row and row["names"] != names)
                or not isinstance(vector, list) or len(vector) != len(names)
                or any(v is not None and not _finite(v) for v in vector)):
            raise ValueError("Invalid extension fixture/vector/snapshot identity")
        for index, name in enumerate(names):
            if name.endswith("__missing") and vector[index] != float(vector[names.index(name[:-9])] is None):
                raise ValueError("Extension missingness indicator disagrees with value")
        fixtures.append(fid)
        snapshots.add(snapshot)
        values.append([np.nan if value is None else value for value in vector])
    lookup = {}
    for row in statistical_rows:
        if not isinstance(row, dict) or type(row.get("fixture_id")) is not int or row["fixture_id"] <= 0:
            raise ValueError("Invalid extension baseline fixture")
        if row["fixture_id"] in lookup:
            raise ValueError("Duplicate extension baseline fixture")
        lookup[row["fixture_id"]] = row
    if set(fixtures) != set(lookup):
        raise ValueError("Extension requires identical baseline/inference fixtures")
    statistical = []
    for row, fid in zip(rows, fixtures):
        baseline = lookup[fid]
        value = baseline.get("statistical")
        if (baseline.get("snapshot_id") != row["snapshot_id"]
                or baseline.get("feature_contract_id") != feature_contract_id
                or baseline.get("market") != metadata["market"]):
            raise ValueError("Extension baseline snapshot/market/feature contract differs")
        if not _number(value, positive=metadata["kind"] == "offset_xgboost"):
            raise ValueError("Invalid extension baseline count or nonpositive offset")
        statistical.append(value)
    return np.asarray(values, dtype=float), np.asarray(statistical, dtype=float)


def predict_bundle(path, rows, *, feature_contract_id, statistical_rows, baseline_contract_id):
    """Replay the frozen strategy using equivalent explicit feature/baseline rows."""
    import joblib

    path = _path(path)
    if (path / "COMPLETE.json").is_symlink():
        raise ValueError("Unsafe extension completion manifest")
    complete = artifacts.verify_complete(path, required={"components.joblib", "manifest.json"})
    if set(complete) != {"components.joblib", "manifest.json"}:
        raise ValueError("Unexpected extension bundle artifacts")
    metadata = artifacts.read_json(path / "manifest.json")
    _metadata(metadata)
    if (metadata.get("version") != VERSION or metadata.get("qualification") != QUALIFICATION
            or metadata.get("publication_enabled") is not False or metadata.get("promotion_allowed") is not False
            or metadata.get("feature_contract_id") != feature_contract_id
            or metadata.get("baseline_contract_id") != baseline_contract_id
            or metadata.get("dependencies") != artifacts.dependencies()
            or metadata.get("python") != sys.version.split()[0]
            or metadata.get("combination") != "total_mean_then_convex_statistical_blend"
            or metadata.get("output_scale") != "expected_count" or metadata.get("rounding") != "none"):
        raise ValueError("Extension feature/baseline/environment/isolation contract mismatch")
    values, statistical = _inputs(rows, statistical_rows, metadata, feature_contract_id)
    components = joblib.load(path / "components.joblib")
    _components(components, metadata)
    if {key: model.metadata for key, model in components.items()} != metadata.get("components"):
        raise ValueError("Serialized extension component metadata mismatch")
    counts = np.zeros(len(rows), dtype=float)
    for key in sorted(components):
        kwargs = {"statistical": statistical} if metadata["kind"] in {"residual_ridge", "offset_xgboost"} else {}
        prediction = np.asarray(components[key].predict(values, names=metadata["names"], **kwargs), dtype=float)
        if prediction.shape != counts.shape or not np.isfinite(prediction).all() or np.any(prediction < 0):
            raise ValueError("Invalid extension component count predictions")
        counts += prediction
    weight = metadata["weight_ml"]
    result = (1 - weight) * statistical + weight * counts
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("Invalid frozen extension strategy counts")
    return result
