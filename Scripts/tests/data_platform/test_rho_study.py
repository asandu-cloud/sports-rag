"""A3.6 study: rho variants on unseen fixtures."""
from Scripts.ops import rho_study as study
from .test_dixon_coles_rho import simulate


def tagged(rows, league, year):
    return [dict(r, league=league, week=f"{year}-W{1 + i % 40:02d}") for i, r in enumerate(rows)]


def test_variants_are_compared_against_the_fixed_value_and_by_league():
    train = tagged(simulate(3000, -0.2, 1), "A", 2022) + tagged(simulate(3000, 0.0, 2), "B", 2022)
    test = tagged(simulate(2000, -0.2, 3), "A", 2023) + tagged(simulate(2000, 0.0, 4), "B", 2023)
    result = study.evaluate(train, test, resamples=100)
    assert set(result["variants"]) == {"fixed", "pooled", "league", "league_shrunk"}
    assert result["fitted"]["league"]["A"] < result["fitted"]["league"]["B"]
    assert result["comparisons"]["league_vs_fixed"]["mean_difference"] < 0
    assert set(result["by_league"]) == {"A", "B"}
