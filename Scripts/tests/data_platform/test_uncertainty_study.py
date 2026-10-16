"""A3.4 study: evidence-aware count variance against the floors and the fixed dispersion."""
import numpy as np

from data_platform import uncertainty as u
from Scripts.ops import uncertainty_study as study


def rows(n, year, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        mean = rng.uniform(7, 12)
        matches = int(rng.integers(1, 30))
        true_variance = u.total_variance(mean, 0.03, matches, kappa=0.5)
        alpha = (true_variance - mean) / mean ** 2
        y = int(rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mean)))
        out.append({"mean": mean, "matches": matches, "y": y, "control_variance": max(1.5 * mean, 8.0),
                    "week": f"{year}-W{1 + i * 50 // n:02d}", "season_stage": "0-7" if matches < 8 else "16+"})
    return out


def test_fitted_evidence_variance_beats_floors_and_fixed_dispersion_when_early_means_are_noisy():
    result = study.evaluate(rows(6000, 2022, 1), rows(4000, 2023, 2), fixed_alpha=0.03, resamples=200)
    assert result["fit"]["kappa"] < 2
    assert result["evidence_vs_fixed"]["interval95"][1] < 0
    assert "0-7" in result["by_season_stage"]
