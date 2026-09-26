"""Four fixed Batch C extension architectures, one estimator per component.

Training and inference consume supplied arrays only. Baseline fixture/version
identity, chronological cohorts and experiment budgets belong to the caller.
Signed residuals are internal regression targets; published-scale predictions
always remain expected counts. Team components do not imply independence.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
from typing import Any, Sequence
import warnings

import numpy as np
from threadpoolctl import threadpool_limits

from Scripts.rag_ingest.core.model_features import digest
from . import estimators as base
from .data import expected_schema

VERSION = "phase3-extension-components.v1"
QUALIFICATION = "UNCONFIRMED_BATCH_C_EXTENSION"
KINDS = ("residual_ridge", "offset_xgboost", "team_poisson", "team_catboost")
FAMILIES = {"residual_ridge": "ridge", "offset_xgboost": "xgboost",
            "team_poisson": "poisson", "team_catboost": "catboost"}
FIXED_PARAMS = {"residual_ridge": {"alpha": 50.0}, "offset_xgboost": {"max_depth": 3},
                "team_poisson": {"alpha": 1.0}, "team_catboost": {"depth": 3}}
ANCHORED = frozenset({"residual_ridge", "offset_xgboost"})


def _inputs(X, names):
    names = base._names(names)
    if names != tuple(expected_schema()["names"]) or len(names) != 126:
        raise ValueError("Extension requires the exact ordered 126-feature reference contract")
    values = base._matrix(X, names)
    base._validate_competitions(values, names)
    return values, names


def _numeric_vector(values, count, *, name, positive=False, integer=False):
    if values is None:
        raise ValueError(name + " must be explicitly supplied")
    try:
        supplied = list(values)
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)) for v in supplied):
            raise ValueError(name + " contains a nonnumeric value or boolean")
        result = np.asarray(supplied, dtype=float)
    except (TypeError, OverflowError) as exc:
        raise ValueError(name + " must be a numeric vector") from exc
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(name + " must be finite with one value per fixture")
    if np.any(result <= 0 if positive else result < 0):
        raise ValueError(name + (" must be strictly positive; no fitted or implicit offset floor" if positive else " must be nonnegative"))
    if integer and np.any(result != np.floor(result)):
        raise ValueError(name + " must contain observed integer counts")
    return result


def _statistical(kind, values, count):
    if kind not in ANCHORED:
        if values is not None:
            raise ValueError("Team-count components do not consume statistical baseline arrays")
        return None
    return _numeric_vector(values, count, name="statistical baseline", positive=kind == "offset_xgboost")


def _checked_prediction(values, count, *, name, nonnegative=False):
    result = np.asarray(values, dtype=float)
    if result.shape != (count,) or not np.isfinite(result).all() or (nonnegative and np.any(result < 0)):
        raise base.EstimatorFitError("Invalid " + name + " shape, numerical value or count scale")
    return result


@dataclass
class FittedExtensionComponent:
    kind: str
    family: str
    market: str
    preprocessor: base.WeightedPreprocessor
    estimator: Any
    metadata: dict[str, Any]

    def predict_with_diagnostics(self, X, *, names: Sequence[str], statistical=None):
        values, names = _inputs(X, names)
        anchor = _statistical(self.kind, statistical, len(values))
        if self.kind not in ANCHORED:
            delegate = base.FittedEstimator(self.family, self.market, self.preprocessor,
                                            self.estimator, self.metadata["delegate"])
            return {**delegate.predict_with_diagnostics(values, names=names),
                    "kind": self.kind, "target_scope": "one_team_count"}
        transformed = self.preprocessor.transform(values, names=names)
        try:
            with warnings.catch_warnings(), threadpool_limits(limits=base.CPU_THREADS), np.errstate(over="raise", invalid="raise"):
                warnings.simplefilter("error")
                if self.kind == "residual_ridge":
                    correction = _checked_prediction(self.estimator.predict(transformed) if len(values) else [],
                                                     len(values), name="signed residual correction")
                    raw = _checked_prediction(anchor + correction, len(values), name="statistical plus residual total")
                    clipped = raw < 0
                    return {"values": np.maximum(raw, 0), "raw_values": raw, "raw_correction": correction,
                            "clipped_count": int(clipped.sum()), "clipped_fraction": float(clipped.mean()) if len(raw) else 0.,
                            "kind": self.kind, "output_scale": "expected_count", "rounding": "none",
                            "nonnegative_policy": "clip_only_statistical_plus_signed_correction"}
                offset = np.log(anchor)
                prediction = self.estimator.predict(transformed, base_margin=offset) if len(values) else []
                margin = self.estimator.predict(transformed, base_margin=offset, output_margin=True) if len(values) else []
                raw = _checked_prediction(prediction, len(values), name="offset expected count", nonnegative=True)
                margin = _checked_prediction(margin, len(values), name="offset native log margin")
                return {"values": raw, "raw_values": raw.copy(), "raw_margin": margin,
                        "log_offset": offset, "log_correction": margin - offset,
                        "clipped_count": 0, "clipped_fraction": 0., "kind": self.kind,
                        "output_scale": "expected_count", "rounding": "none",
                        "nonnegative_policy": "native_count_objective_no_posthoc_floor"}
        except (Warning, ValueError, FloatingPointError) as exc:
            if isinstance(exc, base.EstimatorFitError):
                raise
            raise base.EstimatorFitError(f"{self.kind} prediction failed: {exc}") from exc

    def predict(self, X, *, names: Sequence[str], statistical=None):
        return self.predict_with_diagnostics(X, names=names, statistical=statistical)["values"]

    def predict_raw(self, X, *, names: Sequence[str], statistical=None):
        return self.predict_with_diagnostics(X, names=names, statistical=statistical)["raw_values"]


def fit_component(kind, X, y, weights, names, market, *, statistical=None):
    """Fit exactly one fixed estimator; paired team models require two calls.

    y always enters this API as observed nonnegative integer counts. Residual
    Ridge alone internally fits y minus the supplied statistical mean, without
    invoking the existing count-target adapter or clipping that signed target.
    """
    if kind not in KINDS or market not in base.INTERACTION_BASES:
        raise ValueError("Unsupported fixed extension architecture or market")
    if base.PREPROCESSING_VERSION != "phase3-weighted-preprocessing.v2":
        raise ValueError("Extension requires frozen weighted preprocessing v2")
    values, names = _inputs(X, names)
    target = _numeric_vector(y, len(values), name="observed target", integer=True)
    anchor = _statistical(kind, statistical, len(values))
    mass = base._weights(weights, len(values))
    family, params = FAMILIES[kind], dict(FIXED_PARAMS[kind])
    delegate = None
    if kind not in ANCHORED:
        delegated = base.fit_estimator(family, values, target, weights, names, market, params=params)
        preprocessor, estimator, delegate = delegated.preprocessor, delegated.estimator, delegated.metadata
        package = delegated.metadata["package"]
    else:
        linear = kind == "residual_ridge"
        pairs = base.interaction_pairs(names, market) if linear else ()
        preprocessor = base.WeightedPreprocessor.fit(values, mass, names, scale=linear, pairs=pairs)
        transformed = preprocessor.transform(values, names=names)
        estimator, package = base._make_model(family, params)
        try:
            with warnings.catch_warnings(), threadpool_limits(limits=base.CPU_THREADS), np.errstate(over="raise", invalid="raise"):
                warnings.simplefilter("error")
                if kind == "residual_ridge":
                    estimator.fit(transformed, target - anchor, sample_weight=mass)
                else:
                    estimator.fit(transformed, target, sample_weight=mass, base_margin=np.log(anchor))
        except (Warning, ValueError, FloatingPointError) as exc:
            raise base.EstimatorFitError(f"{kind} fitting failed: {exc}") from exc
    contract = {"version": VERSION, "kind": kind, "family": family, "market": market,
                "names": list(names), "params": params, "preprocessing_version": base.PREPROCESSING_VERSION,
                "target_scope": "fixture_total" if kind in ANCHORED else "one_team_count",
                "baseline_use": "signed_additive_residual" if kind == "residual_ridge"
                    else "log_mean_base_margin_at_fit_and_inference" if kind == "offset_xgboost" else "none"}
    metadata = {**contract, "contract_hash": digest(contract), "feature_contract_id": digest(expected_schema()),
                "package": package, "package_version": importlib.metadata.version(package),
                "resolved_params": base._parameter_metadata(estimator.get_params(deep=False)),
                "preprocessing": preprocessor.metadata(), "training_rows": len(values),
                "positive_weight_rows": int((mass > 0).sum()), "weight_policy": "normalized_to_training_mean_one",
                "weight_sum": float(mass.sum()), "effective_sample_size": float(mass.sum() ** 2 / np.dot(mass, mass)),
                "seed": base.SEED, "cpu_threads": base.CPU_THREADS, "output_scale": "expected_count", "rounding": "none",
                "training_targets_sha256": digest(target.tolist()), "training_weights_sha256": digest(mass.tolist()),
                "training_statistical_sha256": digest(anchor.tolist()) if anchor is not None else None,
                "qualification": QUALIFICATION, "publication_enabled": False, "promotion_allowed": False}
    if delegate is not None:
        metadata["delegate"] = delegate
    if kind == "residual_ridge":
        residuals = target - anchor
        metadata["signed_training_targets"] = {"min": float(residuals.min()), "max": float(residuals.max()),
                                               "negative_count": int(np.sum(residuals < 0)),
                                               "sha256": digest(residuals.tolist())}
        metadata["nonnegative_policy"] = "clip_only_statistical_plus_signed_correction"
    if kind == "offset_xgboost":
        metadata["resolved_native_params"] = json.loads(estimator.get_booster().save_config())
        metadata["offset_policy"] = "strict_positive_finite_statistical_mean; log(mean); no_floor"
        metadata["training_log_offset_sha256"] = digest(np.log(anchor).tolist())
    result = FittedExtensionComponent(kind, family, market, preprocessor, estimator, metadata)
    training = result.predict_with_diagnostics(values, names=names, statistical=anchor)
    metadata["training_clipped_count"] = training["clipped_count"]
    metadata["training_clipped_fraction"] = training["clipped_fraction"]
    if kind == "residual_ridge":
        metadata["training_raw_correction"] = {"min": float(training["raw_correction"].min()),
                                                "max": float(training["raw_correction"].max())}
    json.dumps(metadata, allow_nan=False)
    return result
