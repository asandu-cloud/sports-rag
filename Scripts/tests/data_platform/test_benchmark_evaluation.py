"""Pure chronological evaluation checks, with hand-computed paired evidence."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math

import numpy as np
import pytest

from Scripts.data_platform.features.benchmarks import evaluation as ev


def _row(fid, kickoff, method, *, target=2, prediction=3, market="goals", competition="EPL"):
    return {"fixture_id": fid, "kickoff": kickoff, "competition": competition, "season": 2024,
            "market": market, "fold_id": "outer-01", "method": method,
            "target": target, "prediction": prediction,
            "source_class": "verified_local_reconstruction", "round_group": "domestic_regular",
            "season_stage": "8-19", "support_band": "10+", "missingness_band": "0-10%",
            "forecast_stage": "reconstructed_immediately_before_kickoff"}


def _paired(*, weeks=20, per_week=10, baseline_error=2, candidate_error=1, markets=("goals",)):
    start = datetime(2024, 1, 1, 15, tzinfo=timezone.utc)
    rows = []
    for week in range(weeks):
        kickoff = (start + timedelta(weeks=week)).isoformat()
        for fixture in range(per_week):
            fid = 1 + week * per_week + fixture
            for market in markets:
                for method, error in (("statistical", baseline_error), ("selected_ml", candidate_error)):
                    row = _row(fid, kickoff, method, target=2 + fixture % 4,
                               prediction=2 + fixture % 4 + error, market=market)
                    row["fold_id"] = "outer-01" if week < weeks // 2 else "outer-02"
                    rows.append(row)
    return rows


def test_summary_equal_fixture_metrics_macro_league_weighting_and_poisson_once(monkeypatch):
    calls = []
    actual = ev.evaluate_predictions

    def counted(targets, predictions):
        calls.append(len(targets))
        return actual(targets, predictions)

    monkeypatch.setattr(ev, "evaluate_predictions", counted)
    rows = []
    for fid in range(1, 5):
        for method in ("statistical", "selected_ml"):
            target = float(fid)
            row = _row(fid, f"2024-01-0{fid}T15:00:00+00:00", method,
                       target=target, prediction=target + (4 if fid == 4 else 0),
                       competition="LaLiga" if fid == 4 else "EPL")
            row["fold_id"] = "outer-01" if fid <= 2 else "outer-02"
            rows.append(row)
    report = ev.summarize(rows)
    assert calls == [4, 4]
    item = report["by_market"]["goals"]["methods"]["selected_ml"]
    assert item["overall"]["rmse"] == 2
    assert item["overall"]["mae"] == item["overall"]["bias"] == 1
    assert item["overall"]["r2"] == pytest.approx(1 - 16 / 5)
    assert item["equal_league"]["rmse"] == pytest.approx(math.sqrt(8))
    assert item["equal_league"]["mae"] == item["equal_league"]["bias"] == 2
    assert item["equal_league"]["league_fixture_counts"] == {"EPL": 3, "LaLiga": 1}
    assert "poisson_log_loss" in item["overall"]
    assert "poisson_log_loss" not in item["slices"]["competition"]["EPL"]
    assert "per_fixture" not in item["overall"]
    assert set(item["by_fold"]) == {"outer-01", "outer-02"}
    assert set(item["slices"]) == set(ev.SLICE_FIELDS)
    assert report["by_market"]["goals"]["identical_method_fixture_membership"] is True
    assert report["betting_metrics"] == "not_computed"
    json.dumps(report, allow_nan=False)


def test_constant_target_r2_is_null_and_coverage_differences_are_visible():
    rows = [_row(fid, f"2024-01-0{fid}T12:00:00Z", "selected_ml") for fid in range(1, 4)]
    rows.extend([_row(1, "2024-01-01T12:00:00Z", "statistical")])
    report = ev.summarize(rows)
    selected = report["by_market"]["goals"]["methods"]["selected_ml"]
    assert selected["overall"]["r2"] is None
    assert selected["overall"]["r2_unavailable_reason"] == "constant_targets"
    assert selected["equal_league"]["r2"] is None
    assert report["by_market"]["goals"]["identical_method_fixture_membership"] is False
    with pytest.raises(ValueError, match="membership mismatch"):
        ev.paired_comparison(rows, "selected_ml", "statistical", resamples=20)


def test_calendar_blocks_use_utc_mondays_and_fixed_fortnight_anchor():
    assert ev.calendar_block("2024-01-07T23:30:00-02:00") == "2024-01-08"
    assert ev.calendar_block("2024-01-08T01:30:00Z") == "2024-01-08"
    assert ev.calendar_block("1970-01-11T23:59:59Z") == "1970-01-05"
    assert ev.calendar_block("1970-01-12T00:00:00Z", block_weeks=2) == "1970-01-05"
    assert ev.calendar_block("1970-01-19T00:00:00Z", block_weeks=2) == "1970-01-19"
    # Weeks with no fixtures do not move fortnight boundaries.
    assert ev.calendar_block("1970-02-02T00:00:00Z", block_weeks=2) == "1970-02-02"
    with pytest.raises(ValueError, match="explicit timezone"):
        ev.calendar_block("2024-01-01T12:00:00")


def test_main_bonferroni_interval_and_supported_slice_deterioration():
    report = ev.paired_comparison(_paired(), "selected_ml", "statistical", resamples=200)
    main = report["by_market"]["goals"]["overall"]
    assert main["relative_rmse_improvement"] == 0.5
    assert main["improvement_interval"] == {"confidence": 1 - 0.05 / 6, "lower": 0.5, "upper": 0.5}
    assert main["improvement_interval_excludes_zero"] is True
    assert main["valid_bootstrap_resamples"] == 200
    league = report["by_market"]["goals"]["slices"]["competition"]["EPL"]
    assert league["deterioration_upper_95"] == -0.5
    assert league["slice_deterioration_below_five_percent"] is True
    assert league["improvement_interval"] is None
    assert main["qualification"] == league["qualification"] == "not_assessed"
    assert report["configuration"]["relative_metric_units"] == "fraction_not_percentage"
    json.dumps(report, allow_nan=False)


def test_slice_regression_fails_even_when_its_precision_is_supported():
    report = ev.paired_comparison(_paired(candidate_error=2.2), "selected_ml", "statistical", resamples=100)
    league = report["by_market"]["goals"]["slices"]["competition"]["EPL"]
    assert league["interval_status"] == "available"
    assert league["deterioration_upper_95"] == pytest.approx(0.1)
    assert league["slice_deterioration_below_five_percent"] is False


def test_week_aggregate_bootstrap_matches_expanded_fixture_calculation():
    start = datetime(2024, 1, 1, 15, tzinfo=timezone.utc)
    rows, week_rows = [], []
    for week in range(20):
        observed = []
        # Week sizes differ; averaging weekly RMSE instead of fixture SSE fails.
        for fixture in range(1 + week % 4):
            base_error = 1.0 + fixture + week % 3
            candidate_error = base_error * (0.4 + 0.02 * week)
            fid = 100 * week + fixture + 1
            kickoff = (start + timedelta(weeks=week)).isoformat()
            for method, error in (("statistical", base_error), ("selected_ml", candidate_error)):
                rows.append(_row(fid, kickoff, method, target=5, prediction=5 + error))
            observed.append((base_error, candidate_error))
        week_rows.append(observed)
    report = ev.paired_comparison(rows, "selected_ml", "statistical", resamples=200, seed=42)
    draws = np.random.default_rng(42).multinomial(20, np.full(20, 1 / 20), size=200)
    expanded = []
    for draw in draws:
        fixtures = [pair for week, repeat in enumerate(draw) for _ in range(repeat) for pair in week_rows[week]]
        baseline_rmse = math.sqrt(sum(pair[0] ** 2 for pair in fixtures) / len(fixtures))
        candidate_rmse = math.sqrt(sum(pair[1] ** 2 for pair in fixtures) / len(fixtures))
        expanded.append(1 - candidate_rmse / baseline_rmse)
    alpha = (1 - ev.MAIN_CONFIDENCE) / 2
    expected = np.quantile(expanded, [alpha, 1 - alpha], method="linear")
    actual = report["by_market"]["goals"]["overall"]["improvement_interval"]
    assert [actual["lower"], actual["upper"]] == pytest.approx(expected, abs=1e-14)


def test_reordering_is_identical_and_common_draws_span_markets_and_fixtures():
    rows = _paired(markets=("goals", "corners"))
    for row in rows:
        if row["market"] == "corners":
            row["support_band"] = "5-9"
            row["missingness_band"] = "10-20%"
    normal = ev.paired_comparison(rows, "selected_ml", "statistical", resamples=200)
    shuffled = deepcopy(rows)
    np.random.default_rng(71).shuffle(shuffled)
    reordered = ev.paired_comparison(shuffled, "selected_ml", "statistical", resamples=200)
    assert reordered == normal
    assert normal["configuration"]["nonempty_global_blocks"] == 20
    assert normal["by_market"]["goals"]["overall"] == normal["by_market"]["corners"]["overall"]
    assert normal["by_market"]["corners"]["slices"]["support_band"].keys() == {"5-9"}
    sensitivity = ev.paired_comparison(rows, "selected_ml", "statistical", resamples=200, block_weeks=2)
    assert sensitivity["configuration"]["block_weeks"] == 2
    assert sensitivity["configuration"]["draws_sha256"] != normal["configuration"]["draws_sha256"]
    assert sensitivity["by_market"]["goals"]["overall"]["support"]["calendar_weeks"] == 20


def test_summary_is_order_invariant_and_keeps_market_errors_separate():
    rows = _paired(weeks=2, per_week=2, markets=("goals", "sot"))
    for row in rows:
        if row["market"] == "sot":
            row["target"] += 5
            row["prediction"] += 10
    report = ev.summarize(rows)
    assert ev.summarize(list(reversed(rows))) == report
    assert "overall" not in report
    assert report["by_market"]["goals"]["methods"]["selected_ml"]["overall"]["rmse"] == 1
    assert report["by_market"]["sot"]["methods"]["selected_ml"]["overall"]["rmse"] == 6


def test_insufficient_slices_and_calendar_precision_remain_descriptive():
    sparse = ev.paired_comparison(_paired(per_week=1), "selected_ml", "statistical", resamples=100)
    league = sparse["by_market"]["goals"]["slices"]["competition"]["EPL"]
    assert league["unavailable_reasons"] == ["fewer_than_200_slice_fixtures"]
    assert league["deterioration_upper_95"] is None
    assert sparse["by_market"]["goals"]["overall"]["improvement_interval"] is not None
    short = ev.paired_comparison(_paired(weeks=19, per_week=11), "selected_ml", "statistical", resamples=100)
    overall = short["by_market"]["goals"]["overall"]
    assert overall["unavailable_reasons"] == ["fewer_than_20_calendar_weeks"]
    assert overall["improvement_interval"] is None
    assert overall["relative_rmse_improvement"] == 0.5
    league = short["by_market"]["goals"]["slices"]["competition"]["EPL"]
    assert league["unavailable_reasons"] == ["fewer_than_20_slice_match_dates", "fewer_than_20_calendar_weeks"]


def test_zero_baseline_and_invalid_resamples_are_not_silently_removed():
    zero = ev.paired_comparison(_paired(baseline_error=0), "selected_ml", "statistical", resamples=100)
    main = zero["by_market"]["goals"]["overall"]
    assert main["relative_rmse_improvement"] is None
    assert main["unavailable_reasons"] == ["zero_baseline_rmse"]
    rows = _paired(per_week=1)
    first_kickoff = rows[0]["kickoff"]
    for row in rows:
        if row["method"] == "statistical" and row["kickoff"] != first_kickoff:
            row["prediction"] = row["target"]
    report = ev.paired_comparison(rows, "selected_ml", "statistical", resamples=200)
    main = report["by_market"]["goals"]["overall"]
    assert main["unavailable_reasons"] == ["zero_baseline_rmse_in_bootstrap_resamples"]
    assert main["invalid_bootstrap_resamples"]["zero_baseline_rmse"] > 0
    assert main["improvement_interval"] is None
    assert main["valid_bootstrap_resamples"] < 200


@pytest.mark.parametrize("mutation, match", [
    (lambda rows: rows.append(dict(rows[0])), "Duplicate fixture"),
    (lambda rows: rows[1].update(target=999), "target or slice metadata"),
    (lambda rows: rows[1].update(support_band="different"), "target or slice metadata"),
    (lambda rows: rows[1].update(fold_id="future-fold"), "fold assignment"),
    (lambda rows: rows[1].update(prediction=float("nan")), "finite nonnegative"),
    (lambda rows: rows[1].update(prediction=True), "finite nonnegative"),
    (lambda rows: rows[1].update(target=1.5), "integer counts"),
    (lambda rows: rows[1].update(kickoff="2024-01-01T12:00:00"), "explicit timezone"),
    (lambda rows: rows[1].pop("source_class"), "source_class"),
])
def test_invalid_predictions_identity_and_targets_fail_closed(mutation, match):
    rows = _paired(weeks=2, per_week=1)
    mutation(rows)
    with pytest.raises(ValueError, match=match):
        ev.paired_comparison(rows, "selected_ml", "statistical", resamples=20)


@pytest.mark.parametrize("options", [{"resamples": 1}, {"resamples": 20001}, {"resamples": True},
                                      {"block_weeks": 3}, {"seed": -1}])
def test_invalid_bootstrap_configuration_fails(options):
    with pytest.raises(ValueError):
        ev.paired_comparison(_paired(weeks=1, per_week=1), "selected_ml", "statistical", **options)
