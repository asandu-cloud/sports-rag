"""Independent chronological/cohort checks for the bounded Batch C extension."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import random

import pytest

from Scripts.data_platform.features.benchmarks.extension_selection import (
    KINDS, blend_from_oof, choose_architecture)
from Scripts.rag_ingest.core.model_features import digest


CUTOFFS = ["2022-01-01T00:00:00+00:00", "2022-07-01T00:00:00+00:00",
           "2023-01-01T00:00:00+00:00", "2023-07-01T00:00:00+00:00",
           "2024-01-01T00:00:00+00:00"]


def _rows(block=1, *, errors=(1, 0, 2, 3)):
    start = datetime.fromisoformat(CUTOFFS[block - 1])
    result = []
    for kind, error in zip(KINDS, errors):
        for index, target in enumerate((2, 4)):
            kickoff = start + timedelta(days=10 + index, hours=12)
            fixture = block * 100 + index
            result.append({"fixture_id": fixture, "snapshot_id": f"{fixture:064x}",
                           "kind": kind, "fold_id": f"development-{block:02d}",
                           "kickoff": kickoff.isoformat(), "fit_cutoff": CUTOFFS[block - 1],
                           "training_max_label_available_at": (start - timedelta(hours=1)).isoformat(),
                           "label_available_at": (kickoff + timedelta(hours=3)).isoformat(),
                           "target": target, "prediction": target + error, "statistical": 3.0})
    return result


def _adaptive():
    first = choose_architecture([], prior_fold_ids=[], cutoff=CUTOFFS[0])
    second = choose_architecture(_rows(1), prior_fold_ids=["development-01"], cutoff=CUTOFFS[1])
    choices = {"development-01": first, "development-02": second}
    oof = []
    for number in (1, 2):
        choice = choices[f"development-{number:02d}"]
        oof.extend({**row, "choice_id": choice["choice_id"]}
                   for row in _rows(number) if row["kind"] == choice["kind"])
    return oof, choices


def test_first_block_fixed_seed_cannot_inspect_predictions():
    result = choose_architecture([], prior_fold_ids=[], cutoff=CUTOFFS[0])
    assert result["kind"] == "residual_ridge" and result["policy"] == "fixed_seed"
    assert result["choice_id"] == digest({k: v for k, v in result.items() if k != "choice_id"})
    with pytest.raises(ValueError, match="cannot inspect"):
        choose_architecture(_rows(1), prior_fold_ids=[], cutoff=CUTOFFS[0])


def test_architecture_membership_and_hash_are_invariant_to_input_order():
    original = _rows(1) + _rows(2)
    expected = choose_architecture(original, prior_fold_ids=["development-01", "development-02"], cutoff=CUTOFFS[2])
    random.Random(913).shuffle(original)
    assert choose_architecture(original, prior_fold_ids=["development-01", "development-02"], cutoff=CUTOFFS[2]) == expected
    assert expected["kind"] == "offset_xgboost"
    assert expected["membership_sha256"] == digest([100, 101, 200, 201])


def test_architecture_exact_tie_uses_frozen_table_order():
    result = choose_architecture(_rows(1, errors=(0, 0, 0, 0)), prior_fold_ids=["development-01"], cutoff=CUTOFFS[1])
    assert result["kind"] == KINDS[0]


@pytest.mark.parametrize("available", [CUTOFFS[1], "2022-07-01T03:00:00+00:00"])
def test_late_or_exact_cutoff_labels_cannot_change_architecture_choice(available):
    rows = _rows(1)
    for row in rows:
        if row["fixture_id"] == 101:
            row["label_available_at"] = available
    original = choose_architecture(rows, prior_fold_ids=["development-01"], cutoff=CUTOFFS[1])
    changed = deepcopy(rows)
    for row in changed:
        if row["fixture_id"] == 101:
            row["target"] = 9999
            row["prediction"] = 0.0 if row["kind"] == "team_catboost" else 50000.0
    assert choose_architecture(changed, prior_fold_ids=["development-01"], cutoff=CUTOFFS[1]) == original
    assert original["late_label_fixture_ids"] == [101]
    assert all(score["n"] == 1 for score in original["scores"])


def test_all_labels_unavailable_is_insufficient_evidence():
    rows = _rows(1)
    for row in rows:
        row["label_available_at"] = CUTOFFS[1]
    with pytest.raises(ValueError, match="Insufficient earlier fold evidence"):
        choose_architecture(rows, prior_fold_ids=["development-01"], cutoff=CUTOFFS[1])


def test_architecture_cohorts_must_match_without_dropping_fixtures():
    rows = _rows(1)
    rows.pop()
    with pytest.raises(ValueError, match="cohorts differ"):
        choose_architecture(rows, prior_fold_ids=["development-01"], cutoff=CUTOFFS[1])


@pytest.mark.parametrize("field,value", [("target", 8), ("statistical", 9.0),
                                          ("snapshot_id", "a" * 64),
                                          ("kickoff", "2022-01-11T13:00:00+00:00"),
                                          ("label_available_at", "2022-01-11T16:00:00+00:00")])
def test_architecture_identity_targets_and_baselines_must_match(field, value):
    rows = _rows(1)
    rows[2][field] = value  # same fixture, second architecture
    with pytest.raises(ValueError, match="targets or snapshots differ"):
        choose_architecture(rows, prior_fold_ids=["development-01"], cutoff=CUTOFFS[1])


@pytest.mark.parametrize("change_snapshot", [False, True])
def test_duplicate_fixture_or_revision_never_crosses_selection_folds(change_snapshot):
    rows = _rows(1) + _rows(2)
    duplicate = deepcopy(rows[0])
    if change_snapshot:
        duplicate["snapshot_id"] = "f" * 64
        duplicate["fold_id"] = "development-02"
    rows.append(duplicate)
    with pytest.raises(ValueError, match="Duplicate fixture/version"):
        choose_architecture(rows, prior_fold_ids=["development-01", "development-02"], cutoff=CUTOFFS[2])


@pytest.mark.parametrize("prior", [["development-02"], ["development-01", "development-03"],
                                   ["development-02", "development-01"]])
def test_selection_requires_consecutive_earlier_fold_ids(prior):
    with pytest.raises(ValueError, match="consecutive earlier folds"):
        choose_architecture(_rows(1), prior_fold_ids=prior, cutoff=CUTOFFS[2])


def test_current_or_future_fold_cannot_enter_architecture_selection():
    with pytest.raises(ValueError, match="Future or undeclared"):
        choose_architecture(_rows(1) + _rows(2), prior_fold_ids=["development-01"], cutoff=CUTOFFS[1])


@pytest.mark.parametrize("field,value", [
    ("training_max_label_available_at", CUTOFFS[0]),
    ("fit_cutoff", "2022-01-12T00:00:00+00:00"),
    ("label_available_at", "2022-01-11T12:00:00+00:00"),
])
def test_prediction_requires_strict_training_and_observation_cutoffs(field, value):
    rows = _rows(1)
    rows[0][field] = value
    with pytest.raises(ValueError, match="training or label cutoff"):
        choose_architecture(rows, prior_fold_ids=["development-01"], cutoff=CUTOFFS[1])


def test_blend_requires_two_earlier_blocks_and_keeps_zero_weight_warmup():
    oof, choices = _adaptive()
    first = [row for row in oof if row["fold_id"] == "development-01"]
    result = blend_from_oof(first, choices, cutoff=CUTOFFS[1])
    assert result["status"] == "warmup" and result["weight_ml"] == 0
    fitted = blend_from_oof(oof, choices, cutoff=CUTOFFS[2])
    assert fitted["status"] == "fitted" and 0 <= fitted["weight_ml"] <= 1
    assert fitted["n"] == 4


def test_blend_membership_hash_and_fit_are_invariant_to_input_order():
    oof, choices = _adaptive()
    expected = blend_from_oof(oof, choices, cutoff=CUTOFFS[2])
    random.Random(22).shuffle(oof)
    assert blend_from_oof(oof, choices, cutoff=CUTOFFS[2]) == expected


def test_late_blend_target_cannot_change_weight_or_evidence():
    oof, choices = _adaptive()
    late = {**oof[-1], "fixture_id": 999, "snapshot_id": "f" * 64,
            "label_available_at": CUTOFFS[2], "target": 500}
    expected = blend_from_oof(oof, choices, cutoff=CUTOFFS[2])
    assert blend_from_oof([*oof, late], choices, cutoff=CUTOFFS[2]) == expected
    late["target"] = 90000
    assert blend_from_oof([*oof, late], choices, cutoff=CUTOFFS[2]) == expected


def test_adaptive_blend_rejects_retrospective_architecture_substitution():
    oof, choices = _adaptive()
    assert oof[0]["kind"] == "residual_ridge"
    assert choices["development-02"]["kind"] == "offset_xgboost"
    oof[0]["kind"] = choices["development-02"]["kind"]
    with pytest.raises(ValueError, match="earlier architecture choice"):
        blend_from_oof(oof, choices, cutoff=CUTOFFS[2])


def test_blend_rejects_tampered_choice_identity():
    oof, choices = _adaptive()
    choices["development-01"]["policy"] = "retrospective"
    with pytest.raises(ValueError, match="earlier architecture choice"):
        blend_from_oof(oof, choices, cutoff=CUTOFFS[2])


def test_blend_rejects_resealed_choice_using_future_selection_fold():
    oof, choices = _adaptive()
    bad = choices["development-02"]
    bad["prior_fold_ids"] = ["development-01", "development-02"]
    bad["choice_id"] = digest({k: v for k, v in bad.items() if k != "choice_id"})
    for row in oof:
        if row["fold_id"] == "development-02":
            row["choice_id"] = bad["choice_id"]
    with pytest.raises(ValueError, match="future/nonconsecutive"):
        blend_from_oof(oof, choices, cutoff=CUTOFFS[2])


def test_blend_rejects_choice_using_result_at_its_cutoff():
    oof, choices = _adaptive()
    bad = choices["development-02"]
    bad["max_label_available_at"] = CUTOFFS[1]
    bad["choice_id"] = digest({k: v for k, v in bad.items() if k != "choice_id"})
    for row in oof:
        if row["fold_id"] == "development-02":
            row["choice_id"] = bad["choice_id"]
    with pytest.raises(ValueError, match="future result"):
        blend_from_oof(oof, choices, cutoff=CUTOFFS[2])


def test_blend_rejects_duplicate_fixture_revision():
    oof, choices = _adaptive()
    duplicate = {**oof[0], "snapshot_id": "e" * 64}
    with pytest.raises(ValueError, match="Duplicate adaptive OOF fixture"):
        blend_from_oof([*oof, duplicate], choices, cutoff=CUTOFFS[2])


def test_blend_rejects_current_forecast_even_if_label_would_be_filtered():
    oof, choices = _adaptive()
    with pytest.raises(ValueError, match="current/future forecasts"):
        blend_from_oof(oof, choices, cutoff=CUTOFFS[1])
