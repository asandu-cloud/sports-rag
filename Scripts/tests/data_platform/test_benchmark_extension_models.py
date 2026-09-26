"""Synthetic-only fixed extension architecture contracts and native parity."""
import json

import joblib
import numpy as np
import pytest

from Scripts.data_platform.features.benchmarks import extension_models as extension
from Scripts.data_platform.features.benchmarks import estimators as base
from Scripts.data_platform.features.benchmarks.data import expected_schema


def _training(count=120):
    names = tuple(expected_schema()["names"])
    indices = np.arange(count)
    columns = {}
    for position, name in enumerate(names):
        if name.endswith("__missing"):
            continue
        if name.startswith("competition_"):
            columns[name] = (indices % len(base.COMPETITIONS) == base.COMPETITIONS.index(name[len("competition_"):])).astype(float)
        elif name in {"home_xg_pm", "away_xg_pm"}:
            columns[name] = np.full(count, np.nan)
        else:
            columns[name] = .5 + ((indices + position % 5) % 7) / (position % 3 + 2)
    for name in names:
        if name.endswith("__missing"):
            columns[name] = np.isnan(columns[name[:-9]]).astype(float)
    values = np.column_stack([columns[name] for name in names])
    return values, (indices % 5).astype(float), np.linspace(.25, 2., count), names, 2. + (indices % 3) / 2


@pytest.fixture(scope="module")
def fitted_components():
    values, targets, weights, names, statistical = _training()
    models = {kind: extension.fit_component(kind, values, targets, weights, names, "goals",
              statistical=statistical if kind in extension.ANCHORED else None) for kind in extension.KINDS}
    return models, values, targets, weights, names, statistical


@pytest.mark.parametrize("kind", extension.KINDS)
def test_fixed_component_parameters_preprocessing_count_scale_and_serialization(kind, fitted_components, tmp_path):
    models, values, targets, weights, names, statistical = fitted_components
    model = models[kind]
    argument = statistical[:12] if kind in extension.ANCHORED else None
    result = model.predict_with_diagnostics(values[:12], names=names, statistical=argument)
    assert np.isfinite(result["values"]).all() and np.all(result["values"] >= 0)
    assert result["values"].shape == (12,) and result["output_scale"] == "expected_count"
    assert model.metadata["kind"] == kind and model.metadata["params"] == extension.FIXED_PARAMS[kind]
    assert model.metadata["cpu_threads"] == 2 and model.metadata["seed"] == 42
    assert model.metadata["publication_enabled"] is False and model.metadata["promotion_allowed"] is False
    assert model.metadata["preprocessing_version"] == "phase3-weighted-preprocessing.v2"
    assert model.metadata["training_rows"] == len(targets)
    assert model.preprocessor.scaled == (kind in {"residual_ridge", "team_poisson"})
    assert len(model.preprocessor.pairs) == (52 if model.preprocessor.scaled else 0)
    assert model.metadata["weight_sum"] == pytest.approx(len(targets))
    json.dumps(model.metadata, allow_nan=False)
    path = tmp_path / (kind + ".joblib")
    joblib.dump(model, path)
    restored = joblib.load(path)
    assert np.array_equal(restored.predict(values[:12], names=names, statistical=argument), result["values"])
    assert restored.metadata == model.metadata


def test_signed_residual_target_bypasses_count_adapter_and_correction_is_not_clipped(monkeypatch):
    values, _, weights, names, _ = _training(count=80)
    values[:] = values[0]
    def count_adapter_forbidden(*args, **kwargs):
        pytest.fail("A signed residual must not enter the existing count-target adapter")
    monkeypatch.setattr(base, "fit_estimator", count_adapter_forbidden)
    statistical, target = np.full(len(values), 5.), np.ones(len(values))
    model = extension.fit_component("residual_ridge", values, target, weights, names, "goals", statistical=statistical)
    result = model.predict_with_diagnostics(values, names=names, statistical=statistical)
    assert result["raw_correction"] == pytest.approx(np.full(len(values), -4.))
    assert result["values"] == pytest.approx(target)
    assert model.metadata["signed_training_targets"]["negative_count"] == len(values)
    assert result["clipped_count"] == 0
    model.estimator.coef_[:] = 0
    model.estimator.intercept_ = -10
    clipped = model.predict_with_diagnostics(values[:3], names=names, statistical=statistical[:3])
    assert clipped["raw_correction"].tolist() == [-10, -10, -10]
    assert clipped["raw_values"].tolist() == [-5, -5, -5]
    assert clipped["values"].tolist() == [0, 0, 0] and clipped["clipped_count"] == 3


def test_residual_weighted_intercept_and_zero_baseline_are_valid():
    values, _, _, names, _ = _training(count=20)
    values[:] = values[0]
    target = np.asarray([0.] * 10 + [10.] * 10)
    weights = np.asarray([1.] * 10 + [3.] * 10)
    statistical = np.full(20, 5.)
    model = extension.fit_component("residual_ridge", values, target, weights, names, "goals", statistical=statistical)
    assert model.predict(values[:1], names=names, statistical=[5.]) == pytest.approx([7.5])
    assert model.predict(values[:1], names=names, statistical=[0.]) == pytest.approx([2.5])


def test_offset_training_and_inference_match_native_base_margin_contract(fitted_components):
    models, values, targets, weights, names, statistical = fitted_components
    model = models["offset_xgboost"]
    transformed = model.preprocessor.transform(values, names=names)
    mass = base._weights(weights, len(targets))
    native, _ = base._make_model("xgboost", {"max_depth": 3})
    # Separate synthetic native fit verifies that training really received the
    # log baseline margin, not an adjusted target or an extra feature column.
    native.fit(transformed, targets, sample_weight=mass, base_margin=np.log(statistical))
    result = model.predict_with_diagnostics(values, names=names, statistical=statistical)
    assert np.array_equal(result["values"], native.predict(transformed, base_margin=np.log(statistical)))
    expected_margin = native.predict(transformed, base_margin=np.log(statistical), output_margin=True)
    assert np.array_equal(result["raw_margin"], expected_margin)
    assert result["values"] == pytest.approx(np.exp(expected_margin), rel=1e-6)
    doubled = model.predict(values, names=names, statistical=2 * statistical)
    assert doubled == pytest.approx(2 * result["values"], rel=2e-6)
    params = model.metadata["resolved_params"]
    assert params["objective"] == "count:poisson" and params["tree_method"] == "hist"
    assert params["n_estimators"] == 300 and params["learning_rate"] == .03
    assert params["max_depth"] == 3 and params["reg_lambda"] == 10
    assert model.metadata["offset_policy"].endswith("no_floor")


@pytest.mark.parametrize("kind", extension.KINDS)
def test_heldout_changes_do_not_refit_preprocessing_and_constant_indicators_remain_bounded(kind, fitted_components):
    models, values, _, _, names, statistical = fitted_components
    model = models[kind]
    before = json.dumps(model.metadata, sort_keys=True)
    probe = values[:2].copy()
    for prefix in ("home", "away"):
        probe[:, names.index(prefix + "_xg_pm")] = 2.
        probe[:, names.index(prefix + "_xg_pm__missing")] = 0
    result = model.predict(probe, names=names, statistical=statistical[:2] if kind in extension.ANCHORED else None)
    assert np.isfinite(result).all()
    for prefix in ("home", "away"):
        indicator = names.index(prefix + "_xg_pm__missing")
        assert model.preprocessor.scale[indicator] == 1
        assert np.max(np.abs(model.preprocessor.transform(probe, names=names)[:, indicator])) <= 1
    assert json.dumps(model.metadata, sort_keys=True) == before


@pytest.mark.parametrize("kind", extension.KINDS)
@pytest.mark.parametrize("bad", [-1, .5, True, None, float("nan"), float("inf")])
def test_all_components_require_observed_team_or_fixture_count_targets_before_fitting(kind, bad, monkeypatch):
    values, target, weights, names, statistical = _training(count=10)
    target = target.tolist()
    target[0] = bad
    monkeypatch.setattr(base, "_make_model", lambda *a, **k: pytest.fail("invalid labels must fail before model creation"))
    with pytest.raises(ValueError):
        extension.fit_component(kind, values, target, weights, names, "goals",
                                statistical=statistical if kind in extension.ANCHORED else None)


@pytest.mark.parametrize("bad", [None, [0.], [-1.], [float("nan")], [float("inf")], [True], ["1"], [1., 2.]])
def test_offset_baseline_requires_positive_finite_exact_shape_at_fit_and_inference(bad, fitted_components, monkeypatch):
    models, values, _, weights, names, _ = fitted_components
    monkeypatch.setattr(base, "_make_model", lambda *a, **k: pytest.fail("invalid offsets must fail before model creation"))
    with pytest.raises(ValueError):
        extension.fit_component("offset_xgboost", values[:1], [1.], weights[:1], names, "goals", statistical=bad)
    with pytest.raises(ValueError):
        models["offset_xgboost"].predict(values[:1], names=names, statistical=bad)


@pytest.mark.parametrize("kind", ("team_poisson", "team_catboost"))
def test_team_components_delegate_one_native_count_estimator_and_reject_baseline_inputs(kind, monkeypatch):
    values, targets, weights, names, statistical = _training(count=60)
    actual, calls = base.fit_estimator, []
    def record(*args, **kwargs):
        calls.append((args[0], np.asarray(args[2]).copy(), kwargs["params"]))
        return actual(*args, **kwargs)
    monkeypatch.setattr(base, "fit_estimator", record)
    model = extension.fit_component(kind, values, targets, weights, names, "corners")
    assert len(calls) == 1 and calls[0][0] == extension.FAMILIES[kind]
    assert np.array_equal(calls[0][1], targets)
    assert calls[0][2] == extension.FIXED_PARAMS[kind]
    assert model.metadata["target_scope"] == "one_team_count"
    with pytest.raises(ValueError, match="do not consume"):
        model.predict(values, names=names, statistical=statistical)


def test_extension_rejects_schema_reordering_wrong_competitions_and_missing_indicators(fitted_components):
    models, values, targets, weights, names, statistical = fitted_components
    with pytest.raises(ValueError, match="126-feature"):
        models["residual_ridge"].predict(values[:, ::-1], names=names[::-1], statistical=statistical)
    wrong = values.copy()
    wrong[:, names.index("competition_EPL")] = 1
    with pytest.raises(ValueError, match="exactly one"):
        extension.fit_component("residual_ridge", wrong, targets, weights, names, "goals", statistical=statistical)
    wrong = values.copy()
    wrong[:, names.index("home_xg_pm__missing")] = 0
    with pytest.raises(ValueError, match="Missingness"):
        models["team_poisson"].predict(wrong, names=names)


def test_empty_inference_is_well_defined_without_native_fit_or_predict(fitted_components):
    models, values, _, _, names, _ = fitted_components
    for kind, model in models.items():
        result = model.predict_with_diagnostics(values[:0], names=names, statistical=[] if kind in extension.ANCHORED else None)
        assert result["values"].shape == (0,) and result["clipped_count"] == 0
