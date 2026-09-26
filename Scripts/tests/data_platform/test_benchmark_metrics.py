import math

import numpy as np
import pytest
from scipy.stats import poisson

from Scripts.data_platform.features.benchmarks.metrics import (
    MAX_SUPPORT, MEAN_FLOOR, RPS_TAIL_TOLERANCE, TAIL_MASS_TOLERANCE,
    evaluate_predictions, numerical_metrics, poisson_scores,
)


def test_numerical_metrics_analytic_and_bias_direction():
    result = numerical_metrics([0, 2, 4], [1, 1, 5])
    assert result["n"] == 3
    assert result["mae"] == result["rmse"] == 1
    assert result["bias"] == pytest.approx(1/3)
    assert result["r2"] == pytest.approx(1 - 3/8)
    assert numerical_metrics([3, 3], [2, 4])["r2"] is None
    assert numerical_metrics([3, 3], [2, 4])["r2_unavailable_reason"] == "constant_targets"


def test_poisson_logloss_analytic_and_zero_mean_floor():
    result = poisson_scores([0, 1, 2], [0, 1, 2])
    assert result["poisson_log_loss"] == pytest.approx((MEAN_FLOOR + 1 + 2 - math.log(2))/3)
    assert result["numerical_policy"]["floored_means"] == 1
    assert result["per_fixture"]["poisson_log_loss"][0] == pytest.approx(MEAN_FLOOR, abs=1e-20)
    assert poisson_scores([1], [0])["poisson_log_loss"] == pytest.approx(-math.log(MEAN_FLOOR) + MEAN_FLOOR)
    assert numerical_metrics([0], [0])["rmse"] == 0  # distribution floor does not alter point score


@pytest.mark.parametrize("target,mean", [(0, .01), (0, 2), (4, 2), (15, 30), (0, 100), (100, 1)])
def test_rps_matches_long_direct_cdf_sum_with_certified_tail(target, mean):
    result = poisson_scores([target], [mean])
    counts = np.arange(1000)
    direct = float(np.sum((poisson.cdf(counts, mean) - (counts >= target)) ** 2))
    assert result["rps"] == pytest.approx(direct, abs=1e-11)
    policy = result["numerical_policy"]
    assert policy["maximum_omitted_tail_mass"] <= TAIL_MASS_TOLERANCE
    assert policy["maximum_rps_remainder_bound"] <= RPS_TAIL_TOLERANCE
    assert policy["maximum_used_cutoff"] >= target


def test_intervals_are_inclusive_central_quantiles_with_width_not_count():
    result = poisson_scores([0, 1, 5], [1, 1, 1])
    fifty = result["per_fixture"]["intervals"]["50"]
    assert fifty["lower"] == [0, 0, 0]
    assert fifty["upper"] == [2, 2, 2]
    assert fifty["covered"] == [True, True, False]
    assert result["intervals"]["50"]["coverage"] == pytest.approx(2/3)
    assert result["intervals"]["50"]["mean_width"] == 2
    assert set(result["intervals"]) == {"50", "80", "95"}


@pytest.mark.parametrize("targets,means", [([], []), ([1], []), ([1], [1, 2]), ([1.1], [1]),
    ([-1], [1]), ([1], [-1]), ([None], [1]), ([1], [None]), ([True], [1]), ([1], [False]),
    ([1], [float("nan")]), ([float("inf")], [1]), ([[1]], [[1]]), (["1"], [1]), ([1], ["1"])])
def test_invalid_inputs_rejected_without_silent_row_filtering(targets, means):
    for function in (numerical_metrics, poisson_scores, evaluate_predictions):
        with pytest.raises(ValueError):
            function(targets, means)


def test_explicit_numerical_range_failure_and_generator_equivalence():
    for targets, means in [([MAX_SUPPORT + 1], [1]), ([1], [MAX_SUPPORT + 1]), ([1], [MAX_SUPPORT])]:
        with pytest.raises(ValueError, match="support"):
            poisson_scores(targets, means)
    expected = evaluate_predictions([0, 2], [1, 3])
    assert evaluate_predictions(iter([0, 2]), iter([1, 3])) == expected
    assert expected["n"] == 2
