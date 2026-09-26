from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pytest

from Scripts.data_platform.features.benchmarks.selection import (
    FAMILIES, HALF_LIVES, LOOKBACKS, SEED_PARAMETERS, choose_recipe,
    chronological_choices, fit_blend, fit_oof_blend, recipes, tie_key,
)
from Scripts.rag_ingest.core.model_features import digest

FOLDS = [
    {"fold_id": "development-01", "fit_cutoff": "2022-01-01T00:00:00Z", "validation_end": "2022-07-01T00:00:00Z"},
    {"fold_id": "development-02", "fit_cutoff": "2022-07-01T00:00:00Z", "validation_end": "2023-01-01T00:00:00Z"},
    {"fold_id": "development-03", "fit_cutoff": "2023-01-01T00:00:00Z", "validation_end": "2023-07-01T00:00:00Z"},
    {"fold_id": "development-04", "fit_cutoff": "2023-07-01T00:00:00Z", "validation_end": "2024-01-01T00:00:00Z"},
]


def utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def records():
    output = []
    for fold in FOLDS:
        cutoff = utc(fold["fit_cutoff"])
        for recipe in recipes():
            output.append({"recipe_id": recipe["recipe_id"], "fold_id": fold["fold_id"], "status": "complete",
                "rmse": 2.0, "n": 600, "validation_membership_sha256": digest([fold["fold_id"], "same-fixtures"]),
                "targets_sha256": digest([fold["fold_id"], "same-targets"]), "fit_cutoff": cutoff.isoformat(),
                "training_max_label_available_at": (cutoff - timedelta(hours=1)).isoformat(),
                "max_label_available_at": (cutoff + timedelta(days=90)).isoformat(),
                "support": {"training_unique_fixtures": 6000, "effective_sample_size": 2500, "validation_unique_fixtures": 600}})
    return output


def choose(items, **kwargs):
    return choose_recipe(items, ["development-01", "development-02"], fit_cutoff=FOLDS[2]["fit_cutoff"], **kwargs)


def path(items=None, family=None):
    items = records() if items is None else items
    return chronological_choices({f["fold_id"]: items for f in FOLDS}, FOLDS, family=family)


def oof_rows(choices):
    output = []
    for index, fold in enumerate(FOLDS[:2]):
        cutoff = utc(fold["fit_cutoff"])
        for offset in range(2):
            kickoff = cutoff + timedelta(days=30 + offset)
            output.append({"fixture_id": 10 * index + offset + 1, "fold_id": fold["fold_id"],
                "recipe_id": choices[fold["fold_id"]]["recipe_id"], "kickoff": kickoff.isoformat(),
                "label_available_at": (kickoff + timedelta(hours=3)).isoformat(),
                "training_max_label_available_at": (cutoff - timedelta(hours=1)).isoformat(),
                "target": 1, "statistical": 0., "ml": 2.})
    return output


def test_exact_frozen_grid_counts_parameters_and_stable_ids():
    grid = recipes()
    assert len(grid) == len({r["recipe_id"] for r in grid}) == 156
    assert Counter(r["family"] for r in grid) == {"ridge": 48, "poisson": 36, "lightgbm": 24, "xgboost": 24, "catboost": 24}
    assert {r["params"]["alpha"] for r in grid if r["family"] == "ridge"} == {1, 10, 50, 100}
    assert {r["params"]["alpha"] for r in grid if r["family"] == "poisson"} == {.1, 1, 10}
    assert {(r["params"]["num_leaves"], r["params"]["min_child_samples"]) for r in grid if r["family"] == "lightgbm"} == {(7, 50), (15, 100)}
    assert {r["params"]["max_depth"] for r in grid if r["family"] == "xgboost"} == {3, 4}
    assert {r["params"]["depth"] for r in grid if r["family"] == "catboost"} == {3, 4}
    for family in FAMILIES:
        assert {(r["lookback_days"], r["half_life_days"]) for r in grid if r["family"] == family} == {(a, b) for a in LOOKBACKS for b in HALF_LIVES}
    grid[0]["params"]["alpha"] = 999
    assert recipes()[0]["params"]["alpha"] == 1


def test_exact_ties_are_simple_family_then_shorter_window_then_stable_id():
    result = choose(records())
    expected = min(recipes(), key=tie_key)
    assert result["recipe"] == expected
    assert result["recipe"]["family"] == "ridge"
    assert result["recipe"]["lookback_days"] == 730
    assert len(result["ranking"]) == 156 and not result["excluded"]
    assert result["n"] == 1200 and result["pooled_rmse"] == 2
    assert choose(list(reversed(records()))) == result


def test_primary_ranking_is_fixture_pooled_rmse_not_mean_of_fold_rmse():
    values = records()
    first, second = recipes()[:2]
    for value in values:
        value["rmse"] = 20
        if value["fold_id"] == "development-02":
            value["n"] = value["support"]["validation_unique_fixtures"] = 1800
        if value["recipe_id"] == first["recipe_id"]:
            value["rmse"] = 1 if value["fold_id"] == "development-01" else 3
        elif value["recipe_id"] == second["recipe_id"]:
            value["rmse"] = 2.5
    selected = choose(values)
    assert selected["recipe_id"] == second["recipe_id"]  # first has lower mean fold RMSE but worse pooled error
    assert selected["n"] == 2400


def test_current_future_label_changes_cannot_change_earlier_choice_or_hashes():
    original = records()
    expected = choose(original)
    changed = deepcopy(original)
    for record in changed:
        if record["fold_id"] in {"development-03", "development-04"}:
            record.update(rmse=float("nan"), n=-1, targets_sha256="corrupt", recipe_id="undeclared")
    assert choose(changed) == expected
    assert choose(original[:312]) == expected


@pytest.mark.parametrize("field,value", [("validation_membership_sha256", "a" * 64), ("targets_sha256", "b" * 64),
                                        ("max_label_available_at", "2022-04-10T00:00:00Z")])
def test_unequal_fixture_or_target_cohorts_are_integrity_failures(field, value):
    items = records()
    items[0][field] = value
    with pytest.raises(ValueError, match="identical"):
        choose(items)


@pytest.mark.parametrize("change", ["duplicate", "unknown_recipe", "invalid_score", "invalid_count", "count_support", "score_after_cutoff", "score_equal_cutoff",
                                     "training_equal_cutoff", "naive_time", "source_equal_receiving", "source_disagreement"])
def test_invalid_provenance_and_scores_fail_closed(change):
    items = records()
    record = items[0]
    if change == "duplicate":
        items.append(deepcopy(record))
    elif change == "unknown_recipe":
        record["recipe_id"] = "extra-posthoc-recipe"
    elif change == "invalid_score":
        record["rmse"] = float("nan")
    elif change == "invalid_count":
        record["n"] = True
    elif change == "count_support":
        record["support"]["validation_unique_fixtures"] += 1
    elif change in {"score_after_cutoff", "score_equal_cutoff"}:
        record["max_label_available_at"] = "2023-01-02T00:00:00Z" if change == "score_after_cutoff" else FOLDS[2]["fit_cutoff"]
    elif change == "training_equal_cutoff":
        record["training_max_label_available_at"] = record["fit_cutoff"]
    elif change == "naive_time":
        record["fit_cutoff"] = "2022-01-01T00:00:00"
    elif change == "source_equal_receiving":
        record["fit_cutoff"] = FOLDS[2]["fit_cutoff"]
    elif change == "source_disagreement":
        record["fit_cutoff"] = "2022-01-01T12:00:00Z"
    with pytest.raises(ValueError):
        choose(items)


def test_missing_failed_and_unsupported_recipes_stay_visible_without_changing_cohort():
    items = records()
    missing, failed, unsupported = [r["recipe_id"] for r in recipes()[:3]]
    items = [r for r in items if r["recipe_id"] != missing]
    for record in items:
        if record["recipe_id"] == failed:
            record["status"] = "failed"
        elif record["recipe_id"] == unsupported:
            record["support"]["effective_sample_size"] = 999
    result = choose(items)
    assert result["status"] == "selected" and len(result["ranking"]) == 153
    assert {r["recipe_id"] for r in result["excluded"]} == {missing, failed, unsupported}
    assert result["n"] == 1200


def test_trees_keep_stricter_gates_and_no_supported_recipe_is_explicit():
    items = records()
    for record in items:
        record["support"]["training_unique_fixtures"] = 4999
    result = choose(items)
    assert len(result["ranking"]) == 84 and len(result["excluded"]) == 72
    assert choose(items, family="lightgbm")["status"] == "insufficient_development_evidence"
    one = choose_recipe(items, ["development-01"], fit_cutoff=FOLDS[1]["fit_cutoff"])
    assert one["recipe"] is None
    assert all("insufficient_prior_validation_folds" in entry["reasons"] for entry in one["excluded"])


@pytest.mark.parametrize("family", (None, *FAMILIES))
def test_each_adaptive_path_has_predeclared_seed_explicit_warmup_then_two_prior_folds(family):
    choices = path(family=family)
    first = choices["development-01"]
    expected_family = family or "ridge"
    assert first["recipe"]["family"] == expected_family
    assert first["recipe"]["params"] == SEED_PARAMETERS[expected_family]
    assert first["recipe"]["lookback_days"] is None and first["recipe"]["half_life_days"] == 365
    assert first["selection_fold_ids"] == [] and first["stage"] == "fixed_seed"
    assert choices["development-02"]["stage"] == "one_fold_warmup"
    assert choices["development-02"]["minimum_prior_folds"] == 1
    assert choices["development-03"]["selection_fold_ids"] == ["development-01", "development-02"]
    assert choices["development-04"]["selection_fold_ids"] == ["development-01", "development-02", "development-03"]
    assert [c["outer_evaluation_eligible"] for c in choices.values()] == [False, False, True, True]


def test_adaptive_oof_does_not_reuse_retrospectively_selected_fixed_recipe():
    items = records()
    a = recipes()[0]["recipe_id"]
    b = next(r["recipe_id"] for r in recipes() if r["family"] == "poisson")
    for record in items:
        record["rmse"] = 10
        if record["recipe_id"] == a:
            record["rmse"] = 1 if record["fold_id"] == "development-01" else 4
        if record["recipe_id"] == b:
            record["rmse"] = 3 if record["fold_id"] == "development-01" else 1
    choices = path(items)
    assert choices["development-02"]["recipe_id"] == a
    assert choices["development-03"]["recipe_id"] == b
    correct = oof_rows(choices)
    assert fit_oof_blend(correct, choices, fit_cutoff=FOLDS[2]["fit_cutoff"])["weight_ml"] == .5
    retrospectively_selected = deepcopy(correct)
    for row in retrospectively_selected:
        row["recipe_id"] = b
    with pytest.raises(ValueError, match="Retrospective"):
        fit_oof_blend(retrospectively_selected, choices, fit_cutoff=FOLDS[2]["fit_cutoff"])


@pytest.mark.parametrize("targets,statistical,ml,weight,reason", [
    ([1, 2], [0, 0], [2, 4], .5, "interior"),
    ([1, 2], [1, 2], [3, 4], 0, "boundary_statistical"),
    ([3, 4], [1, 2], [3, 4], 1, "boundary_ml"),
    ([2, 3], [1, 2], [1, 2], 0, "identical_predictions"),
    ([0, 0], [1, 2], [3, 4], 0, "boundary_statistical"),
    ([4, 6], [0, 0], [1, 2], 1, "boundary_ml"),
])
def test_analytic_convex_blend_and_zero_ml_is_valid(targets, statistical, ml, weight, reason):
    result = fit_blend(targets, statistical, ml)
    assert result["weight_ml"] == weight
    assert result["weight_statistical"] == 1 - weight
    assert result["reason"] == reason
    expected = sum((y - ((1 - weight) * s + weight * p)) ** 2 for y, s, p in zip(targets, statistical, ml))
    assert result["sse"] == expected
    assert result["rmse"] == math.sqrt(expected / len(targets))


@pytest.mark.parametrize("values", [[], [None], [True], ["1"], [-1], [float("nan")], [float("inf")], [[1]], None])
def test_blend_never_silently_filters_invalid_pairs(values):
    for column in range(3):
        inputs = [[1], [1], [1]]
        inputs[column] = values
        with pytest.raises(ValueError):
            fit_blend(*inputs)
    with pytest.raises(ValueError):
        fit_blend([1, 2], [1], [1])
    with pytest.raises(ValueError):
        fit_blend([1.5], [1], [1])


def test_blend_matches_bounded_grid_and_input_order_does_not_change_weight():
    y = [0, 4, 2, 8]
    s = [1, 3, 3, 5]
    p = [2, 6, 1, 9]
    result = fit_blend(y, s, p)
    objectives = [sum((a - (b + w * (c - b))) ** 2 for a, b, c in zip(y, s, p)) for w in np.linspace(0, 1, 1001)]
    assert result["sse"] <= min(objectives)
    assert fit_blend(y[::-1], s[::-1], p[::-1])["weight_ml"] == result["weight_ml"]


def test_oof_blend_binds_membership_selection_hashes_and_is_order_stable():
    choices = path()
    rows = oof_rows(choices)
    result = fit_oof_blend(rows, choices, fit_cutoff=FOLDS[2]["fit_cutoff"])
    assert result["weight_ml"] == .5 and result["n"] == 4
    assert result["fold_ids"] == ["development-01", "development-02"]
    assert fit_oof_blend(rows[::-1], choices, fit_cutoff=FOLDS[2]["fit_cutoff"]) == result
    changed = deepcopy(choices)
    changed["development-04"]["pooled_rmse"] = -999  # later results do not enter earlier blending
    assert fit_oof_blend(rows, changed, fit_cutoff=FOLDS[2]["fit_cutoff"]) == result


@pytest.mark.parametrize("change", ["duplicate", "late_label", "equal_label", "source_training", "own_fold_selection", "tampered_choice",
                                     "future_recipe", "too_few_folds", "wrong_source_fold", "wrong_seed", "ranking_substitution"])
def test_oof_provenance_rejects_leakage_and_retrospective_paths(change):
    choices = path()
    rows = oof_rows(choices)
    if change == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif change in {"late_label", "equal_label"}:
        rows[0]["label_available_at"] = "2023-01-02T00:00:00Z" if change == "late_label" else FOLDS[2]["fit_cutoff"]
    elif change == "source_training":
        rows[0]["training_max_label_available_at"] = FOLDS[0]["fit_cutoff"]
    elif change == "own_fold_selection":
        choices["development-02"]["selection_fold_ids"].append("development-02")
    elif change == "tampered_choice":
        choices["development-01"]["recipe"]["params"]["alpha"] = 999
    elif change == "future_recipe":
        rows[0]["recipe_id"] = choices["development-03"]["recipe_id"]
    elif change == "too_few_folds":
        rows = rows[:2]
    elif change == "wrong_source_fold":
        rows[0]["kickoff"], rows[0]["label_available_at"] = "2022-08-01T00:00:00Z", "2022-08-01T03:00:00Z"
    elif change == "wrong_seed":
        choices["development-01"]["recipe"] = recipes()[0]
        choices["development-01"]["recipe_id"] = recipes()[0]["recipe_id"]
        rows[0]["recipe_id"] = rows[1]["recipe_id"] = recipes()[0]["recipe_id"]
    elif change == "ranking_substitution":
        choices["development-02"]["ranking"][0]["recipe_id"] = recipes()[-1]["recipe_id"]
    if change in {"own_fold_selection", "wrong_seed", "ranking_substitution"}:
        for value in choices.values():
            value["choice_id"] = digest({k: v for k, v in value.items() if k != "choice_id"})
    with pytest.raises(ValueError):
        fit_oof_blend(rows, choices, fit_cutoff=FOLDS[2]["fit_cutoff"])


def test_final_development_choice_uses_all_four_prior_folds_but_no_confirmation_records():
    items = records()
    result = choose_recipe(items, [f["fold_id"] for f in FOLDS], fit_cutoff="2024-01-01T00:00:00Z")
    assert result["n"] == 2400 and len(result["selection_fold_ids"]) == 4
    items.append({"fold_id": "phase3_confirmation", "rmse": -1000, "targets": "never inspect"})
    assert choose_recipe(items, [f["fold_id"] for f in FOLDS], fit_cutoff="2024-01-01T00:00:00Z") == result
