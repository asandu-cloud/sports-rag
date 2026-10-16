"""A3.3 study: time-ordered evaluation of the market–Spix blend."""
import numpy as np

from Scripts.ops import blend_study as study


def rows(n=4000, spix_signal=0.6, seed=5):
    """Outcomes depend on the market and, partly, on information only Spix has."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        base = rng.normal(0, 1)
        private = rng.normal(0, 1)
        true = 1 / (1 + np.exp(-(base + spix_signal * private)))
        market = 1 / (1 + np.exp(-base))
        spix = 1 / (1 + np.exp(-(0.5 * base + private)))
        y = int(rng.random() < true)
        quarter = f"2023-Q{1 + i * 4 // n}"
        out.append({"quarter": quarter, "week": f"2023-W{1 + i * 52 // n:02d}", "market": [market, 1 - market],
                    "spix": [spix, 1 - spix], "closing": [market, 1 - market], "y": 1 - y})
    return out


def test_blend_beats_the_market_when_spix_has_private_information():
    result = study.cross_validate(rows(), resamples=200)
    assert result["blend_vs_market"]["interval95"][1] < 0
    assert result["blend_vs_recalibrated"]["interval95"][1] < 0  # the gain comes from Spix, not recalibration
    assert all(fold["c"] > 0 for fold in result["folds"])


def test_without_private_information_spix_earns_no_weight():
    result = study.cross_validate(rows(spix_signal=0.0), resamples=200)
    assert result["blend_vs_recalibrated"]["interval95"][1] >= 0  # no reliable gain from Spix


def test_each_fold_trains_only_on_earlier_quarters():
    result = study.cross_validate(rows(n=800), resamples=50)
    for fold in result["folds"]:
        assert max(fold["trained_on"]) < fold["tested_on"]
    assert [f["tested_on"] for f in result["folds"]] == ["2023-Q2", "2023-Q3", "2023-Q4"]
