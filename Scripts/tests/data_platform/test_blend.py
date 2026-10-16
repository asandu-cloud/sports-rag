"""A3.3: fitted logarithmic pool of the market and Spix."""
import math
import random

import numpy as np
import pytest

from data_platform import blend


def simulate(n, a, b, c, classes=3, seed=1):
    """Outcomes drawn from p ∝ exp(a_k) · market_k^b · spix_k^c."""
    rng = np.random.default_rng(seed)
    market = rng.dirichlet([4] * classes, size=n)
    spix = rng.dirichlet([4] * classes, size=n)
    logits = np.array(a)[None, :] + b * np.log(market) + c * np.log(spix)
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(classes, p=row) for row in p])
    return market, spix, y


def test_fit_recovers_the_pooling_weights():
    market, spix, y = simulate(6000, [0.1, -0.2, 0.0], 0.9, 0.3)
    model = blend.fit(market, spix, y)
    assert model["b"] == pytest.approx(0.9, abs=0.08) and model["c"] == pytest.approx(0.3, abs=0.08)
    assert model["intercepts"][0] - model["intercepts"][2] == pytest.approx(0.1, abs=0.1)


def test_market_only_model_fixes_spix_weight_at_zero():
    market, spix, y = simulate(2000, [0, 0, 0], 1.0, 0.5, seed=2)
    model = blend.fit(market, spix, y, use_spix=False)
    assert model["c"] == 0.0


def test_predictions_are_probabilities_and_identity_reproduces_the_market():
    market, spix, _ = simulate(50, [0, 0, 0], 1, 0, seed=3)
    identity = {"intercepts": [0.0, 0.0, 0.0], "b": 1.0, "c": 0.0}
    p = blend.predict(identity, market, spix)
    assert np.allclose(p, market) and np.allclose(p.sum(axis=1), 1)


def test_two_way_pool_is_logistic_regression_on_logits():
    market, spix, y = simulate(4000, [0.2, 0.0], 1.1, 0.25, classes=2, seed=4)
    model = blend.fit(market, spix, y)
    from sklearn.linear_model import LogisticRegression
    X = np.column_stack([np.log(market[:, 0] / market[:, 1]), np.log(spix[:, 0] / spix[:, 1])])
    reference = LogisticRegression(penalty=None, max_iter=1000).fit(X, (y == 0).astype(int))
    assert model["b"] == pytest.approx(reference.coef_[0][0], abs=1e-3)
    assert model["c"] == pytest.approx(reference.coef_[0][1], abs=1e-3)
