"""Numerical adapter for candidate estimators, never the active model loader.

The candidate fit and inference paths share schema checks and null-to-NaN
conversion. No imputations are learned here and no artifact paths are loaded.
"""
from __future__ import annotations

import numpy as np
import hashlib
from pathlib import Path

from .model_features import build_features, digest

FEATURE_IMPLEMENTATION = hashlib.sha256(Path(__file__).with_name("model_features.py").read_bytes()).hexdigest()


def candidate_matrix(snapshots, *, expected_contract=None):
    """Return X and its exact contract; require fitted preprocessing in a model.

    Historical backfills and true observed snapshots have different availability
    labels, but the same numerical schema/policy. Missing values remain NaN for
    a later training-only fitted imputer (not global mean filling).
    """
    rows = []
    contract = expected_contract
    for snapshot in snapshots:
        features = build_features(snapshot)
        actual = {"feature_version": features["feature_version"], "names": features["names"],
                  "policy_id": digest(snapshot["policy"]), "registry_id": digest(snapshot["competitions"]),
                  "implementation_id": FEATURE_IMPLEMENTATION}
        if contract is None:
            contract = actual
        if contract != actual:
            raise ValueError("Candidate feature schema/policy does not match the fitted model")
        rows.append([float("nan") if value is None else value for value in features["values"]])
    if not rows:
        raise ValueError("At least one input snapshot is required")
    return np.asarray(rows, dtype=np.float64), contract


def candidate_predict(model, snapshot, *, feature_contract):
    """Evaluate an explicitly supplied candidate estimator without promotion."""
    matrix, _ = candidate_matrix([snapshot], expected_contract=feature_contract)
    result = np.asarray(model.predict(matrix), dtype=float).reshape(-1)
    if result.size != 1 or not np.isfinite(result[0]):
        raise ValueError("Candidate returned an invalid prediction")
    return float(result[0])
