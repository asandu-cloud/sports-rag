"""Independent integration checks for forward reporting and label cutoffs."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from Scripts.data_platform.features.benchmarks import forward
from Scripts.data_platform.features.benchmarks.evaluation import paired_comparison, summarize


def test_forward_prediction_rows_feed_market_specific_evaluation_without_reweighting():
    names = ["home_current_matches", "away_current_matches", "optional",
             "home_current_matches__missing", "away_current_matches__missing", "optional__missing"]
    start = datetime(2023, 1, 2, 12, tzinfo=timezone.utc)
    source = []
    for index in range(200):
        source.append({"fixture": {"fixture_id": index + 1, "competition": "EPL", "season": 2023,
                                   "kickoff": (start + timedelta(weeks=index // 10)).isoformat()},
                       "values": [10, 3, None, 0, 0, 1], "target": 2 + index % 3,
                       "support": {market: {side: {"count": count} for side in ("home", "away")}
                                   for market, count in (("goals", 6), ("sot", 12))},
                       "snapshot_id": f"{index:064x}", "source_class": "verified_local_reconstruction",
                       "round_group": "domestic_regular", "forecast_stage": "reconstructed_immediately_before_kickoff"})
    records = []
    for market in ("goals", "sot"):
        for method, error in (("statistical", 2), ("selected_blend", 1)):
            records.extend(forward.prediction_rows(
                source, [row["target"] + error for row in source], market=market,
                fold={"fold_id": "development-03"}, method=method, names=names,
            ))
    assert len(records) == 800
    assert {row["season_stage"] for row in records} == {"0-7"}
    assert {row["missingness_band"] for row in records} == {">25%"}
    assert {row["support_band"] for row in records if row["market"] == "goals"} == {"5-9"}
    assert {row["support_band"] for row in records if row["market"] == "sot"} == {"10+"}
    paired = paired_comparison(records, "selected_blend", "statistical", resamples=100)
    for market in ("goals", "sot"):
        result = paired["by_market"][market]["overall"]
        assert result["support"]["unique_fixtures"] == 200
        assert result["relative_rmse_improvement"] == 0.5
    # The summary adapter accepts every required field emitted by the runner.
    small = [row for row in records if row["fixture_id"] <= 2]
    report = summarize(small)
    assert report["by_market"]["goals"]["methods"]["selected_blend"]["overall"]["n"] == 2


def test_selection_rescoring_cannot_see_labels_at_or_after_receiving_cutoff(monkeypatch):
    recipe = {"recipe_id": "synthetic-recipe"}
    monkeypatch.setattr(forward, "recipes", lambda: [recipe])
    fold = {"fold_id": "development-01", "fit_cutoff": "2022-01-01T00:00:00Z"}
    rows = [{"fixture": {"fixture_id": 1}, "target": 2, "label_available_at": "2022-06-30T20:00:00Z"},
            {"fixture": {"fixture_id": 2}, "target": 500, "label_available_at": "2022-07-01T00:00:00Z"},
            {"fixture": {"fixture_id": 3}, "target": 999, "label_available_at": "2022-07-01T03:00:00Z"}]
    grid = {("goals", "development-01", "synthetic-recipe"): {
        "status": "complete", "predictions": np.array([3., 1., 1.]), "support": {},
        "training_max_label_available_at": "2021-12-31T12:00:00Z",
    }}
    kwargs = dict(market="goals", prior_folds=[fold], receiving_cutoff="2022-07-01T00:00:00Z")
    original = forward.selection_records(grid, {("goals", "development-01"): rows}, **kwargs)
    changed = deepcopy(rows)
    changed[1]["target"], changed[2]["target"] = 8000, 9000
    assert forward.selection_records(grid, {("goals", "development-01"): changed}, **kwargs) == original
    assert original[0]["n"] == 1 and original[0]["rmse"] == 1
    assert original[0]["max_label_available_at"] == "2022-06-30T20:00:00Z"


def test_final_refit_respects_elapsed_window_and_strict_label_availability():
    cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc)
    earliest = cutoff - timedelta(days=730)

    def row(fid, kickoff, available, eligible=True):
        return {"fixture": {"fixture_id": fid, "kickoff": kickoff.isoformat()},
                "label_available_at": available.isoformat(), "labels": {"goals": 3},
                "market_eligibility": {"goals": {"eligible": eligible}}}

    rows = [row(1, earliest - timedelta(seconds=1), earliest + timedelta(hours=3)),
            row(2, earliest, earliest + timedelta(hours=3)),
            row(3, cutoff - timedelta(hours=3), cutoff),
            row(4, cutoff - timedelta(hours=4), cutoff - timedelta(hours=1)),
            row(5, cutoff - timedelta(days=1), cutoff - timedelta(hours=20), False)]
    data = SimpleNamespace(rows=rows, splits={"boundaries": {"phase3_confirmation_start": cutoff.isoformat()}})
    selected = forward.final_training_rows(data, "goals", {"lookback_days": 730})
    assert [row["fixture"]["fixture_id"] for row in selected] == [2, 4]
    assert [row["target"] for row in selected] == [3, 3]
    assert all("target" not in row for row in rows)


def test_earlier_blend_ignores_future_targets_and_rejects_future_selection():
    rows = [
        {"fixture_id": 1, "fold_id": "development-01", "kickoff": "2022-01-10T12:00:00Z",
         "label_available_at": "2022-01-10T15:00:00Z", "fit_cutoff": "2022-01-01T00:00:00Z",
         "selection_cutoff": "2022-01-01T00:00:00Z", "selection_validation_ends": [],
         "training_max_label_available_at": "2021-12-31T12:00:00Z", "target": 2, "statistical": 3, "ml": 2},
        {"fixture_id": 2, "fold_id": "development-02", "kickoff": "2022-07-10T12:00:00Z",
         "label_available_at": "2022-07-10T15:00:00Z", "fit_cutoff": "2022-07-01T00:00:00Z",
         "selection_cutoff": "2022-07-01T00:00:00Z", "selection_validation_ends": ["2022-07-01T00:00:00Z"],
         "training_max_label_available_at": "2022-06-30T12:00:00Z", "target": 4, "statistical": 3, "ml": 4},
    ]
    cutoff = "2023-01-01T00:00:00Z"
    expected = forward.earlier_blend(rows, cutoff=cutoff)
    future = {**rows[-1], "fixture_id": 3, "label_available_at": cutoff, "target": 999}
    assert forward.earlier_blend([*rows, future], cutoff=cutoff) == expected
    assert expected["weight_ml"] == 1
    changed = deepcopy(rows)
    changed[1]["selection_validation_ends"] = ["2022-08-01T00:00:00Z"]
    with pytest.raises(ValueError, match="not selected/trained strictly forward"):
        forward.earlier_blend(changed, cutoff=cutoff)
