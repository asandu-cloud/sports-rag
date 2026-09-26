"""Offline adapter/transform checks; no database, provider or active artifacts."""
import importlib.util
import json
import warnings

import joblib
import numpy as np
import pytest

from Scripts.data_platform.features.benchmarks import estimators as adapters


def _with_indicators(values, names):
    values = np.asarray(values, dtype=float)
    return np.column_stack((values, np.isnan(values).astype(float))), tuple(names) + tuple(
        f"{name}__missing" for name in names
    )


def _training(market="goals", count=160):
    names = list(adapters.INTERACTION_BASES[market]) + [
        f"competition_{name}" for name in adapters.COMPETITIONS
    ] + ["optional_xg", "optional_context"]
    values = np.zeros((count, len(names)))
    index = np.arange(count)
    for i in range(4):
        values[:, i] = 1.0 + (index % (i + 5)) / (i + 2)
    values[index % 2 == 0, names.index("competition_EPL")] = 1
    values[index % 2 == 1, names.index("competition_LaLiga")] = 1
    values[:, -2] = np.nan
    values[:, -1] = index / count
    values[::7, -1] = np.nan
    values, names = _with_indicators(values, names)
    return values, (index % 5 + 1).astype(float), np.linspace(0.2, 2, count), names


def test_weighted_median_ties_zero_mass_and_missing_are_explicit():
    assert adapters.weighted_median([0, 10], [1, 1]) == 0
    assert adapters.weighted_median([10, 0], [1, 1]) == 0
    assert adapters.weighted_median([0, 10, np.nan], [1, 3, 100]) == 10
    assert adapters.weighted_median([np.nan, 999], [2, 0]) == 0
    assert adapters.weighted_median([1, 10], [0, 2]) == 10
    with pytest.raises(ValueError, match="Invalid"):
        adapters.weighted_median([1, 2], [1, -1])


def test_weighted_imputation_scaling_matches_hand_computation_and_keeps_missingness():
    X, names = _with_indicators([[0, np.nan], [10, np.nan], [np.nan, np.nan]], ["rate", "optional"])
    transform = adapters.WeightedPreprocessor.fit(X, [1, 3, 2], names, scale=True)
    assert transform.medians.tolist() == [10, 0, 0, 1]
    assert transform.mean.tolist() == pytest.approx([25 / 3, 0, 1 / 3, 1])
    assert transform.scale.tolist() == pytest.approx([np.sqrt(125 / 9), 1, np.sqrt(2 / 9), 1])
    assert transform.all_missing == ("optional",)
    probe, _ = _with_indicators([[0, 0], [0, np.nan]], ["rate", "optional"])
    observed = transform.transform(probe, names=names)
    assert observed[0, 1] == observed[1, 1] == 0
    assert observed[0, 3] == -1 and observed[1, 3] == 0


def test_weighted_constant_missingness_cannot_amplify_newly_observed_validation_values():
    count = 12_435
    raw = np.column_stack((np.full(count, np.nan), np.arange(count, dtype=float) / count,
                           np.full(count, .1)))
    values, names = _with_indicators(raw, ["optional_xg", "varying", "constant_decimal"])
    weights = np.exp(-np.arange(count, dtype=float) / 2000)
    transform = adapters.WeightedPreprocessor.fit(values, weights, names, scale=True)
    indicator = names.index("optional_xg__missing")
    decimal = names.index("constant_decimal")
    assert transform.mean[indicator] == 1 and transform.scale[indicator] == 1
    assert transform.mean[decimal] == .1 and transform.scale[decimal] == 1
    assert transform.transform(values, names=names)[:, indicator].tolist() == [0.] * count
    probe, _ = _with_indicators([[2., .5, .1], [np.nan, .5, .1]], ["optional_xg", "varying", "constant_decimal"])
    before = json.dumps(transform.metadata(), sort_keys=True)
    actual = transform.transform(probe, names=names)
    assert actual[:, indicator].tolist() == [-1., 0.]
    assert np.max(np.abs(actual)) < 10
    assert json.dumps(transform.metadata(), sort_keys=True) == before
    assert transform.metadata()["version"] == "phase3-weighted-preprocessing.v2"
    assert "no_nonconstant_variance_floor" in transform.metadata()["constant_column_policy"]


def test_constant_detection_uses_only_positive_weight_training_rows():
    values, names = _with_indicators([[.1], [.1], [1e6]], ["rate"])
    transform = adapters.WeightedPreprocessor.fit(values, [1, 3, 0], names, scale=True)
    assert transform.mean.tolist() == [.1, 0.]
    assert transform.scale.tolist() == [1., 1.]
    heldout, _ = _with_indicators([[.2]], ["rate"])
    assert transform.transform(heldout, names=names)[0, 0] == pytest.approx(.1)


def test_nonconstant_tiny_variance_keeps_existing_weighted_moments_without_floor():
    count = 100
    raw = (1 + np.arange(count, dtype=float) * 1e-12)[:, None]
    values, names = _with_indicators(raw, ["tiny_varying"])
    weights = np.linspace(.2, 2., count)
    transform = adapters.WeightedPreprocessor.fit(values, weights, names, scale=True)
    mass = adapters._weights(weights, count)
    expected_mean = np.average(values, axis=0, weights=mass)
    expected_scale = np.sqrt(np.average((values - expected_mean) ** 2, axis=0, weights=mass))
    assert transform.mean[0] == expected_mean[0]
    assert transform.scale[0] == expected_scale[0]
    assert 0 < transform.scale[0] < 1e-8
    assert transform.scale[0] != 1


def test_heldout_values_and_zero_weight_rows_cannot_affect_saved_transforms():
    X, names = _with_indicators([[0], [10], [np.nan], [1e9]], ["rate"])
    weights = [1, 3, 2, 0]
    transform = adapters.WeightedPreprocessor.fit(X, weights, names, scale=True)
    clean = adapters.WeightedPreprocessor.fit(X[:3], weights[:3], names, scale=True)
    assert transform.medians == pytest.approx(clean.medians)
    assert transform.mean == pytest.approx(clean.mean)
    assert transform.scale == pytest.approx(clean.scale)
    before = json.dumps(transform.metadata(), sort_keys=True)
    for heldout in ([[np.nan]], [[-1e6]], [[1e12]]):
        values, _ = _with_indicators(heldout, ["rate"])
        assert np.isfinite(transform.transform(values, names=names)).all()
    assert json.dumps(transform.metadata(), sort_keys=True) == before


def test_contract_rejects_reordering_and_false_or_missing_indicators():
    X, names = _with_indicators([[0], [np.nan]], ["rate"])
    transform = adapters.WeightedPreprocessor.fit(X, [1, 1], names, scale=False)
    with pytest.raises(ValueError, match="names/order"):
        transform.transform(X[:, ::-1], names=names[::-1])
    broken = X.copy()
    broken[1, 1] = 0
    with pytest.raises(ValueError, match="disagrees"):
        transform.transform(broken, names=names)
    with pytest.raises(ValueError, match="Missing explicit"):
        adapters.WeightedPreprocessor.fit(X[:, :1], [1, 1], ["rate"], scale=False)
    with pytest.raises(ValueError, match="Infinite"):
        transform.transform([[np.inf, 0]], names=names)


@pytest.mark.parametrize("market", ["goals", "corners", "sot"])
def test_exact_declared_interactions_form_after_imputation(market):
    X, _, weights, names = _training(market)
    pairs = adapters.interaction_pairs(names, market)
    assert len(pairs) == 52 and len(set(pairs)) == 52
    first_rate = names.index(adapters.INTERACTION_BASES[market][0])
    X[0, first_rate] = np.nan
    X[0, names.index(f"{names[first_rate]}__missing")] = 1
    transform = adapters.WeightedPreprocessor.fit(X, weights, names, scale=False, pairs=pairs)
    output = transform.transform(X, names=names)
    epl_term = pairs.index((names[first_rate], "competition_EPL"))
    liga_term = pairs.index((names[first_rate], "competition_LaLiga"))
    assert output[0, len(names) + epl_term] == transform.medians[first_rate]
    assert output[0, len(names) + liga_term] == 0
    assert transform.output_names[len(names) + epl_term] == f"interaction__{names[first_rate]}__x__competition_EPL"


@pytest.mark.parametrize("family", adapters.FAMILIES)
def test_family_count_scale_weights_and_saved_inference(family, tmp_path):
    package = "sklearn" if family in ("ridge", "poisson") else family
    if importlib.util.find_spec(package) is None:
        pytest.skip(f"Local unit environment lacks {package}; production runner must fail explicitly")
    X, y, weights, names = _training()
    fitted = adapters.fit_estimator(family, X, y, weights, names, "goals")
    result = fitted.predict_with_diagnostics(X[:12], names=names)
    assert np.isfinite(result["values"]).all()
    assert (result["values"] >= 0).all()
    assert result["output_scale"] == "expected_count"
    assert fitted.metadata["weight_sum"] == pytest.approx(len(y))
    normalized = weights / weights.mean()
    assert fitted.metadata["effective_sample_size"] == pytest.approx(normalized.sum() ** 2 / (normalized ** 2).sum())
    assert fitted.metadata["cpu_threads"] == 2 and fitted.metadata["seed"] == 42
    assert fitted.preprocessor.scaled == (family in ("ridge", "poisson"))
    assert len(fitted.preprocessor.pairs) == (52 if family in ("ridge", "poisson") else 0)
    json.dumps(fitted.metadata, allow_nan=False)
    transformed = fitted.preprocessor.transform(X[:12], names=names)
    if family == "catboost":
        raw = fitted.estimator.predict(transformed, prediction_type="RawFormulaVal", thread_count=2)
        assert result["values"] == pytest.approx(np.exp(raw))
    elif family == "xgboost":
        raw = fitted.estimator.predict(transformed, output_margin=True)
        assert result["values"] == pytest.approx(np.exp(raw), rel=1e-6)
    elif family == "lightgbm":
        raw = fitted.estimator.booster_.predict(transformed, raw_score=True, num_threads=2)
        assert result["values"] == pytest.approx(np.exp(raw))
    elif family == "poisson":
        assert result["values"] == pytest.approx(np.exp(transformed @ fitted.estimator.coef_ + fitted.estimator.intercept_))
    path = tmp_path / f"{family}.joblib"
    joblib.dump(fitted, path)
    restored = joblib.load(path)
    assert restored.predict(X[:12], names=names) == pytest.approx(result["values"], abs=1e-12)
    assert restored.metadata == fitted.metadata


def test_sample_weights_reach_estimator_and_are_normalized():
    X, _, _, names = _training(count=20)
    # All predictors identical: the Ridge intercept is exactly the weighted
    # target mean. Unequal weights must affect the actual estimator fit.
    X[:] = X[0]
    y = np.array([0] * 10 + [10] * 10, dtype=float)
    weights = np.array([1] * 10 + [3] * 10, dtype=float)
    fitted = adapters.fit_estimator("ridge", X, y, weights, names, "goals")
    assert fitted.predict(X[:1], names=names) == pytest.approx([7.5])
    scaled = adapters.fit_estimator("ridge", X, y, weights * 100, names, "goals")
    assert scaled.predict(X[:1], names=names) == pytest.approx([7.5])


def test_ridge_keeps_negative_raw_predictions_and_counts_clipping():
    X, y, weights, names = _training()
    fitted = adapters.fit_estimator("ridge", X, y, weights, names, "goals")
    fitted.estimator.coef_[:] = 0
    fitted.estimator.intercept_ = -0.25
    result = fitted.predict_with_diagnostics(X[:3], names=names)
    assert result["values"].tolist() == [0, 0, 0]
    assert result["raw_values"].tolist() == [-0.25] * 3
    assert result["clipped_count"] == 3 and result["clipped_fraction"] == 1


def test_fit_warnings_are_failures_and_missing_dependencies_are_explicit(monkeypatch):
    X, y, weights, names = _training()

    class WarningEstimator:
        def fit(self, *args, **kwargs):
            warnings.warn("synthetic convergence failure", UserWarning)

    monkeypatch.setattr(adapters, "_make_model", lambda *args: (WarningEstimator(), "scikit-learn"))
    with pytest.raises(adapters.EstimatorFitError, match="convergence failure"):
        adapters.fit_estimator("ridge", X, y, weights, names, "goals")
    real_import = adapters.importlib.import_module

    def missing_import(name):
        if name == "xgboost":
            raise ImportError("not installed in isolated research environment")
        return real_import(name)

    monkeypatch.setattr(adapters.importlib, "import_module", missing_import)
    with pytest.raises(adapters.EstimatorDependencyError, match="xgboost"):
        adapters._dependency("xgboost")


@pytest.mark.parametrize("change, match", [
    ({"family": "unknown"}, "Unsupported benchmark family"),
    ({"market": "cards"}, "Unsupported benchmark market"),
    ({"params": {"alpha": 0.00001}}, "outside the predeclared"),
    ({"params": {"n_jobs": 128}}, "Only declared"),
])
def test_unapproved_grid_and_scope_are_rejected(change, match):
    X, y, weights, names = _training()
    kwargs = dict(family="ridge", X=X, y=y, weights=weights, names=names, market="goals")
    kwargs.update(change)
    with pytest.raises(ValueError, match=match):
        adapters.fit_estimator(**kwargs)


def test_invalid_targets_competitions_and_outputs_fail():
    X, y, weights, names = _training()
    y[0] = np.nan
    with pytest.raises(ValueError, match="Targets must be known"):
        adapters.fit_estimator("ridge", X, y, weights, names, "goals")
    y[0] = 1.5
    with pytest.raises(ValueError, match="observed integer"):
        adapters.fit_estimator("ridge", X, y, weights, names, "goals")
    y[0] = 1
    fitted = adapters.fit_estimator("ridge", X, y, weights, names, "goals")
    broken = X[:1].copy()
    broken[0, names.index("competition_LaLiga")] = 1
    with pytest.raises(ValueError, match="exactly one"):
        fitted.predict(broken, names=names)
    fitted.estimator.intercept_ = np.nan
    with pytest.raises(adapters.EstimatorFitError, match="nonfinite"):
        fitted.predict(X[:1], names=names)
