"""Isolated, weighted Phase 3 expected-count estimators.

This module consumes already selected training arrays. It cannot read a dataset,
provider, production profile or active model. The caller owns chronological
selection and qualification. Every learned transformation is saved in the
returned object and reused without fitting during prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import json
import warnings
from typing import Any, Sequence

import numpy as np
from threadpoolctl import threadpool_limits


VERSION = "phase3-estimators.v1"
PREPROCESSING_VERSION = "phase3-weighted-preprocessing.v2"
INTERACTION_VERSION = "phase3-league-rate-interactions.v1"
FAMILIES = ("ridge", "poisson", "lightgbm", "xgboost", "catboost")
SEED = 42
CPU_THREADS = 2
COMPETITIONS = (
    "BelgianProLeague", "Bundesliga", "Championship", "EPL", "Eredivisie",
    "LaLiga", "Ligue1", "PrimeiraLiga", "SerieA", "SuperLig", "UCL", "UECL", "UEL",
)
# Home production meets away concession; away production meets home concession.
# Exactly these four rates, crossed with every existing league indicator, are
# declared before development scoring. No fixture/team/player IDs are features.
INTERACTION_BASES = {
    "goals": ("home_goals_for_pm", "away_goals_against_pm",
              "away_goals_for_pm", "home_goals_against_pm"),
    "corners": ("home_corners_pm", "away_corners_against_pm",
                "away_corners_pm", "home_corners_against_pm"),
    "sot": ("home_sot_for_pm", "away_sot_against_pm",
            "away_sot_for_pm", "home_sot_against_pm"),
}
DEFAULT_PARAMS = {
    "ridge": {"alpha": 50.0},
    "poisson": {"alpha": 1.0},
    "lightgbm": {"num_leaves": 7, "min_child_samples": 50},
    "xgboost": {"max_depth": 3},
    "catboost": {"depth": 3},
}


class EstimatorDependencyError(RuntimeError):
    """An agreed family is unavailable; the runner must not silently drop it."""


class EstimatorFitError(RuntimeError):
    """A fit/prediction warning or numerical failure invalidated a candidate."""


def _names(names: Sequence[str]) -> tuple[str, ...]:
    result = tuple(names)
    if not result or any(not isinstance(n, str) or not n for n in result):
        raise ValueError("Feature names must be nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError("Feature names must be unique")
    return result


def _matrix(X, names: tuple[str, ...]) -> np.ndarray:
    try:
        values = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Features must be a numerical two-dimensional matrix") from exc
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("Feature matrix shape does not match the exact ordered names")
    if np.isinf(values).any():
        raise ValueError("Infinite feature values are invalid; missing values use NaN/None")
    return values


def _weights(weights, count: int) -> np.ndarray:
    result = np.asarray(weights, dtype=np.float64)
    if result.shape != (count,) or not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("Weights must be finite nonnegative values, one per training row")
    if not count or not np.any(result > 0):
        raise ValueError("Training requires positive total weight")
    # Scaling first prevents overflow for large finite, proportionate weights.
    relative = result / result.max()
    return relative * (count / relative.sum())


def weighted_median(values, weights) -> float:
    """Lower weighted median; exact half-weight ties select the lower value.

    Missing observations and zero-weight observations contribute no evidence.
    An all-missing contributing column uses the declared zero placeholder.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("Weighted median requires matching one-dimensional arrays")
    if np.isinf(values).any() or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Invalid values or weights for weighted median")
    known = ~np.isnan(values) & (weights > 0)
    if not known.any():
        return 0.0
    order = np.argsort(values[known], kind="stable")
    selected = values[known][order]
    mass = weights[known][order]
    mass = mass / mass.max()
    cumulative = np.cumsum(mass)
    index = np.searchsorted(cumulative, cumulative[-1] / 2.0, side="left")
    return float(selected[index])


def interaction_pairs(names: Sequence[str], market: str) -> tuple[tuple[str, str], ...]:
    if market not in INTERACTION_BASES:
        raise ValueError(f"Unsupported benchmark market: {market}")
    pairs = tuple((base, f"competition_{league}")
                  for base in INTERACTION_BASES[market] for league in COMPETITIONS)
    absent = sorted({name for pair in pairs for name in pair} - set(names))
    if absent:
        raise ValueError(f"Missing predeclared interaction features: {absent}")
    return pairs


@dataclass
class WeightedPreprocessor:
    """Saved train-only imputation, declared interactions and optional scaling.

    The source schema must carry an explicit ``__missing`` column for each
    original value. Indicators are checked against the raw matrix, never
    replaced by a fitted inference-time missingness heuristic.
    """

    names: tuple[str, ...]
    output_names: tuple[str, ...]
    medians: np.ndarray
    all_missing: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    scaled: bool
    pairs: tuple[tuple[str, str], ...]

    @staticmethod
    def _validate_missingness(values: np.ndarray, names: tuple[str, ...]) -> None:
        positions = {name: i for i, name in enumerate(names)}
        for name, index in positions.items():
            if name.endswith("__missing"):
                base = name[:-9]
                if base not in positions:
                    raise ValueError(f"Orphan missingness indicator: {name}")
                expected = np.isnan(values[:, positions[base]]).astype(float)
                if not np.array_equal(values[:, index], expected):
                    raise ValueError(f"Missingness indicator disagrees with its raw feature: {name}")
            elif f"{name}__missing" not in positions:
                raise ValueError(f"Missing explicit missingness indicator for {name}")

    @classmethod
    def fit(cls, X, weights, names: Sequence[str], *, scale: bool,
            pairs: Sequence[tuple[str, str]] = ()) -> "WeightedPreprocessor":
        names = _names(names)
        values = _matrix(X, names)
        mass = _weights(weights, len(values))
        cls._validate_missingness(values, names)
        pairs = tuple(tuple(pair) for pair in pairs)
        if len(set(pairs)) != len(pairs) or any(len(p) != 2 for p in pairs):
            raise ValueError("Interaction pairs must be unique pairs of feature names")
        if any(name not in names for pair in pairs for name in pair):
            raise ValueError("An interaction references an unknown feature")
        output_names = names + tuple(f"interaction__{a}__x__{b}" for a, b in pairs)
        if len(set(output_names)) != len(output_names):
            raise ValueError("Derived feature names conflict with the source contract")
        medians = np.array([weighted_median(values[:, i], mass) for i in range(len(names))])
        all_missing = tuple(name for i, name in enumerate(names)
                            if np.isnan(values[mass > 0, i]).all())
        result = cls(names, output_names, medians, all_missing,
                     np.zeros(len(output_names)), np.ones(len(output_names)), bool(scale), pairs)
        expanded = result._impute_expand(values)
        if scale:
            contributing = mass > 0
            training, active_mass = expanded[contributing], mass[contributing]
            constant = np.all(training == training[0], axis=0)
            result.mean = np.average(training, axis=0, weights=active_mass)
            # Weighted summation can round a column of exact ones away from
            # one, creating a spurious tiny variance. Its inference scale must
            # remain one even when that previously constant feature changes.
            # This exact training-only test adds no nonconstant variance floor.
            result.mean[constant] = training[0, constant]
            variance = np.average((training - result.mean) ** 2, axis=0, weights=active_mass)
            result.scale = np.sqrt(variance)
            result.scale[result.scale == 0] = 1.0
            result.scale[constant] = 1.0
        if not np.isfinite(result.mean).all() or not np.isfinite(result.scale).all():
            raise ValueError("Preprocessing produced nonfinite weighted moments")
        result.transform(X, names=names)
        return result

    def _impute_expand(self, values: np.ndarray) -> np.ndarray:
        filled = np.where(np.isnan(values), self.medians, values)
        if self.pairs:
            positions = {name: i for i, name in enumerate(self.names)}
            terms = np.column_stack([filled[:, positions[a]] * filled[:, positions[b]]
                                     for a, b in self.pairs])
            filled = np.column_stack((filled, terms))
        return filled

    def transform(self, X, *, names: Sequence[str]) -> np.ndarray:
        if _names(names) != self.names:
            raise ValueError("Inference feature names/order differ from the saved contract")
        values = _matrix(X, self.names)
        self._validate_missingness(values, self.names)
        transformed = self._impute_expand(values)
        if self.scaled:
            transformed = (transformed - self.mean) / self.scale
        if not np.isfinite(transformed).all():
            raise ValueError("Transformed features must be finite")
        return transformed

    def metadata(self) -> dict[str, Any]:
        return {
            "version": PREPROCESSING_VERSION,
            "input_names": list(self.names), "output_names": list(self.output_names),
            "median_policy": "lower_observed_value_at_exact_half_weight_tie",
            "medians": self.medians.tolist(), "all_missing_columns": list(self.all_missing),
            "all_missing_placeholder": 0.0, "missingness": "validated_source_indicators",
            "scaled": self.scaled, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
            "variance": "weighted_population; zero_variance_scale_is_one",
            "constant_column_policy": "exact_equal_values_on_positive_weight_training_rows; exact_mean_and_unit_scale; no_nonconstant_variance_floor",
            "interaction_version": INTERACTION_VERSION if self.pairs else None,
            "interaction_pairs": [list(pair) for pair in self.pairs],
        }


def _params(family: str, params: dict | None) -> dict:
    if family not in FAMILIES:
        raise ValueError(f"Unsupported benchmark family: {family}")
    result = dict(DEFAULT_PARAMS[family])
    if params is not None:
        if not isinstance(params, dict) or set(params) - set(result):
            raise ValueError("Only declared family-grid parameters may be overridden")
        result.update(params)
    if any(isinstance(value, bool) for value in result.values()):
        raise ValueError("Boolean model parameters are invalid")
    valid = {
        "ridge": result.get("alpha") in (1, 10, 50, 100),
        "poisson": result.get("alpha") in (0.1, 1, 10),
        "lightgbm": (result.get("num_leaves"), result.get("min_child_samples")) in ((7, 50), (15, 100)),
        "xgboost": result.get("max_depth") in (3, 4),
        "catboost": result.get("depth") in (3, 4),
    }[family]
    if not valid:
        raise ValueError(f"Parameters are outside the predeclared {family} grid")
    return result


def _dependency(module: str):
    try:
        return importlib.import_module(module)
    except (ImportError, OSError) as exc:
        raise EstimatorDependencyError(f"Required research dependency {module!r} is unavailable: {exc}") from exc


def _parameter_metadata(value):
    """Represent native parameter sentinels without emitting nonstandard JSON."""
    if isinstance(value, dict):
        return {str(key): _parameter_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_parameter_metadata(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        # XGBoost declares missing=NaN even though this contract already imputes.
        # This is parameter evidence, not a nonfinite feature or prediction.
        return {"nonfinite_float_parameter": str(value)}
    return value


def _make_model(family: str, params: dict):
    if family in ("ridge", "poisson"):
        linear = _dependency("sklearn.linear_model")
        if family == "ridge":
            return linear.Ridge(alpha=params["alpha"], solver="svd"), "scikit-learn"
        return linear.PoissonRegressor(alpha=params["alpha"], max_iter=1000,
                                       tol=1e-8, solver="lbfgs"), "scikit-learn"
    if family == "lightgbm":
        library = _dependency("lightgbm")
        return library.LGBMRegressor(
            objective="poisson", n_estimators=300, learning_rate=0.03,
            num_leaves=params["num_leaves"], min_child_samples=params["min_child_samples"],
            random_state=SEED, n_jobs=CPU_THREADS, deterministic=True,
            force_col_wise=True, verbosity=-1,
        ), "lightgbm"
    if family == "xgboost":
        library = _dependency("xgboost")
        return library.XGBRegressor(
            objective="count:poisson", tree_method="hist", device="cpu",
            max_depth=params["max_depth"], reg_lambda=10.0,
            n_estimators=300, learning_rate=0.03, random_state=SEED, n_jobs=CPU_THREADS,
            subsample=1.0, colsample_bytree=1.0, verbosity=0,
        ), "xgboost"
    library = _dependency("catboost")
    return library.CatBoostRegressor(
        loss_function="Poisson", depth=params["depth"], l2_leaf_reg=10.0,
        iterations=300, learning_rate=0.03, bootstrap_type="No",
        task_type="CPU", thread_count=CPU_THREADS, random_seed=SEED,
        allow_writing_files=False, verbose=False,
    ), "catboost"


def _validate_competitions(X: np.ndarray, names: tuple[str, ...]) -> None:
    expected = tuple(f"competition_{league}" for league in COMPETITIONS)
    actual = {name for name in names if name.startswith("competition_") and not name.endswith("__missing")}
    if actual != set(expected):
        raise ValueError("The exact thirteen-competition indicator contract is required")
    indicators = X[:, [names.index(name) for name in expected]]
    if not np.isin(indicators, (0.0, 1.0)).all() or np.any(indicators.sum(axis=1) != 1.0):
        raise ValueError("Every fixture must have exactly one known competition indicator")


@dataclass
class FittedEstimator:
    family: str
    market: str
    preprocessor: WeightedPreprocessor
    estimator: Any
    metadata: dict[str, Any]

    def predict_raw(self, X, *, names: Sequence[str]) -> np.ndarray:
        values = _matrix(X, _names(names))
        _validate_competitions(values, _names(names))
        transformed = self.preprocessor.transform(values, names=names)
        if not len(transformed):
            return np.empty(0, dtype=float)
        try:
            with warnings.catch_warnings(), threadpool_limits(limits=CPU_THREADS):
                warnings.simplefilter("error")
                if self.family == "catboost":
                    # CatBoost's default Poisson prediction is the raw log mean.
                    prediction = self.estimator.predict(transformed, prediction_type="Exponent",
                                                        thread_count=CPU_THREADS)
                elif self.family == "lightgbm":
                    # Native booster avoids sklearn feature-name heuristics; our
                    # saved ordered contract above performs the exact validation.
                    prediction = self.estimator.booster_.predict(transformed, raw_score=False,
                                                                num_threads=CPU_THREADS)
                else:
                    prediction = self.estimator.predict(transformed)
        except (Warning, ValueError, FloatingPointError) as exc:
            raise EstimatorFitError(f"{self.family} prediction failed: {exc}") from exc
        result = np.asarray(prediction, dtype=float)
        if result.shape != (len(transformed),) or not np.isfinite(result).all():
            raise EstimatorFitError(f"{self.family} returned nonfinite or incorrectly shaped predictions")
        if self.family != "ridge" and np.any(result < 0):
            raise EstimatorFitError(f"{self.family} returned negative expected counts")
        return result

    def predict_with_diagnostics(self, X, *, names: Sequence[str]) -> dict[str, Any]:
        raw = self.predict_raw(X, names=names)
        clipped = raw < 0
        return {"values": np.maximum(raw, 0.0), "raw_values": raw,
                "clipped_count": int(clipped.sum()),
                "clipped_fraction": float(clipped.mean()) if len(raw) else 0.0,
                "output_scale": "expected_count", "rounding": "none"}

    def predict(self, X, *, names: Sequence[str]) -> np.ndarray:
        return self.predict_with_diagnostics(X, names=names)["values"]


def fit_estimator(family: str, X, y, weights, names: Sequence[str], market: str,
                  params: dict | None = None) -> FittedEstimator:
    """Fit one declared recipe using only the arrays supplied by its caller.

    Returns original-scale expected counts. Only Ridge receives a nonnegative
    projection; its raw predictions remain available for clipping diagnostics.
    Missing packages, warnings and invalid outputs fail the recipe explicitly.
    """
    recipe = _params(family, params)
    names = _names(names)
    if market not in INTERACTION_BASES:
        raise ValueError(f"Unsupported benchmark market: {market}")
    if any("card" in name.lower() for name in names):
        raise ValueError("Card-derived features are excluded by the benchmark contract")
    if any(name.removesuffix("__missing").endswith("_id") for name in names):
        raise ValueError("Arbitrary identity IDs cannot be numerical benchmark features")
    values = _matrix(X, names)
    target = np.asarray(y, dtype=np.float64)
    if target.shape != (len(values),) or not np.isfinite(target).all() or np.any(target < 0):
        raise ValueError("Targets must be known finite nonnegative counts")
    if np.any(target != np.floor(target)):
        raise ValueError("Numerical targets must be observed integer counts, not imputed rates")
    mass = _weights(weights, len(values))
    _validate_competitions(values, names)
    linear = family in ("ridge", "poisson")
    pairs = interaction_pairs(names, market) if linear else ()
    preprocessing = WeightedPreprocessor.fit(values, mass, names, scale=linear, pairs=pairs)
    transformed = preprocessing.transform(values, names=names)
    estimator, package = _make_model(family, recipe)
    try:
        with warnings.catch_warnings(), threadpool_limits(limits=CPU_THREADS):
            warnings.simplefilter("error")
            estimator.fit(transformed, target, sample_weight=mass)
    except (Warning, ValueError, FloatingPointError) as exc:
        raise EstimatorFitError(f"{family} fitting failed: {exc}") from exc
    contract = {"version": VERSION, "family": family, "market": market,
                "names": list(names), "interaction_pairs": [list(pair) for pair in pairs],
                "preprocessing_version": PREPROCESSING_VERSION}
    metadata = {
        **contract, "contract_hash": hashlib.sha256(json.dumps(contract, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "recipe": recipe, "seed": SEED, "cpu_threads": CPU_THREADS,
        "package": package, "package_version": importlib.metadata.version(package),
        "resolved_params": _parameter_metadata(estimator.get_params(deep=False)),
        "training_rows": len(values), "positive_weight_rows": int((mass > 0).sum()),
        "weight_policy": "normalized_to_training_mean_one",
        "weight_sum": float(mass.sum()), "effective_sample_size": float(mass.sum() ** 2 / np.dot(mass, mass)),
        "preprocessing": preprocessing.metadata(), "output_scale": "expected_count",
        "nonnegative_policy": "clip_at_zero_keep_raw" if family == "ridge" else "count_objective",
        "rounding": "none", "promotion_allowed": False,
    }
    if family == "catboost":
        metadata["resolved_native_params"] = _parameter_metadata(estimator.get_all_params())
    elif family == "xgboost":
        metadata["resolved_native_params"] = json.loads(estimator.get_booster().save_config())
    elif family == "lightgbm":
        metadata["resolved_native_params"] = _parameter_metadata(dict(estimator.booster_.params))
    result = FittedEstimator(family, market, preprocessing, estimator, metadata)
    training_prediction = result.predict_with_diagnostics(values, names=names)
    metadata["training_clipped_count"] = training_prediction["clipped_count"]
    metadata["training_clipped_fraction"] = training_prediction["clipped_fraction"]
    # Contract manifests must not hide unserializable defaults or NaN values.
    json.dumps(metadata, sort_keys=True, allow_nan=False)
    return result
