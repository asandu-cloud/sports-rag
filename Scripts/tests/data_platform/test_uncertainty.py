"""A3.4: count distributions whose width reflects how many matches the mean rests on."""
import numpy as np
import pytest
from scipy.stats import nbinom

from data_platform import uncertainty as u


def test_variance_has_a_fixed_part_and_a_part_that_shrinks_with_evidence():
    assert u.total_variance(10, alpha=0.0, matches=1e12) == pytest.approx(10)  # Poisson in the limit
    assert u.total_variance(10, alpha=0.025, matches=1e12) == pytest.approx(10 + 0.025 * 100)
    early, late = u.total_variance(10, 0.025, matches=3), u.total_variance(10, 0.025, matches=30)
    assert early > late > 10 + 2.5


def test_log_probability_matches_the_negative_binomial():
    mean, variance = 9.5, 14.0
    alpha = (variance - mean) / mean ** 2
    expected = nbinom.logpmf(11, 1 / alpha, 1 / (1 + alpha * mean))
    assert u.log_pmf(11, mean, variance) == pytest.approx(expected)
    assert u.log_pmf(3, 4.0, 4.0) == pytest.approx(np.log(np.exp(-4) * 4 ** 3 / 6))  # Poisson when var == mean


def test_fit_recovers_dispersion_and_evidence_scale():
    rng = np.random.default_rng(11)
    rows = []
    for _ in range(12000):
        mean = rng.uniform(6, 12)
        matches = rng.integers(1, 30)
        variance = u.total_variance(mean, 0.03, matches=matches, kappa=0.8)
        alpha = (variance - mean) / mean ** 2
        y = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mean))
        rows.append({"mean": mean, "matches": matches, "y": int(y)})
    fit = u.fit(rows)
    assert fit["alpha"] == pytest.approx(0.03, abs=0.012)
    assert fit["kappa"] == pytest.approx(0.8, rel=0.35)


def test_effective_matches_follow_the_engine_rule():
    team = {"current": 3, "prior": 38}
    assert u.effective_matches(team) == 3 + 8  # current plus at most the prior strength (8)
    assert u.effective_matches({"current": 20, "prior": 38}) == 20
