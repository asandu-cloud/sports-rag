"""A3.6: Dixon–Coles rho fitted per league on the engine's own goal means."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rag_ingest"))

from data_platform import dixon_coles_rho as dcr  # noqa: E402
from prob_models import dixon_coles_scoreline_prob  # noqa: E402


def simulate(n, rho, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        h, a = rng.uniform(0.8, 2.0), rng.uniform(0.6, 1.6)
        cells = [(i, j) for i in range(10) for j in range(10)]
        p = np.array([dixon_coles_scoreline_prob(i, j, h, a, rho) for i, j in cells])
        i, j = cells[rng.choice(len(cells), p=p / p.sum())]
        rows.append({"home_mean": h, "away_mean": a, "home_goals": int(i), "away_goals": int(j)})
    return rows


def test_log_loss_uses_the_engines_own_scoreline_probability():
    row = {"home_mean": 1.4, "away_mean": 1.1, "home_goals": 1, "away_goals": 1}
    assert dcr.nll([row], -0.1) == pytest.approx(-np.log(dixon_coles_scoreline_prob(1, 1, 1.4, 1.1, -0.1)))


def test_fit_recovers_rho():
    fit = dcr.fit(simulate(8000, -0.15, 1))
    assert fit["rho"] == pytest.approx(-0.15, abs=0.04)


def test_fitted_rho_never_makes_a_low_score_negative():
    rows = [{"home_mean": 3.0, "away_mean": 2.5, "home_goals": 0, "away_goals": 0}] * 50
    fit = dcr.fit(rows)
    assert 1 - 3.0 * 2.5 * fit["rho"] >= 0 and 1 - fit["rho"] >= 0


def test_shrinkage_pulls_small_leagues_toward_the_pooled_value():
    assert dcr.shrink(-0.30, 100, pooled=-0.10, strength=300) == pytest.approx(-0.10 + (-0.20) * 100 / 400)
    assert dcr.shrink(-0.30, 1e9, pooled=-0.10, strength=300) == pytest.approx(-0.30, abs=1e-6)
