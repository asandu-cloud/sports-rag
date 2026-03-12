"""Probabilistic primitives for the betting RAG.

Pure functions, no side effects, no external dependencies beyond stdlib.
Provides Poisson / negative-binomial modeling for count events (goals,
corners, cards, SoT), implied probability extraction from bookmaker odds,
expected value, value edge, and Kelly criterion.
"""

from math import exp, lgamma, log
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Poisson distribution
# ---------------------------------------------------------------------------

def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for Poisson(lam). Uses log-space to avoid overflow."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return exp(k * log(lam) - lam - lgamma(k + 1))


def poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k) for Poisson(lam)."""
    return sum(poisson_pmf(i, lam) for i in range(k + 1))


def poisson_over_prob(lam: float, line: float) -> float:
    """P(X > line).

    For half-integer lines (e.g. 2.5, 9.5) there is no push —
    P(Over 9.5) = P(X >= 10) = 1 - P(X <= 9).
    """
    threshold = int(line)
    return 1.0 - poisson_cdf(threshold, lam)


def poisson_under_prob(lam: float, line: float) -> float:
    """P(X < line).

    For half-integer lines (e.g. 2.5), P(Under 2.5) = P(X <= 2).
    """
    threshold = int(line)
    return poisson_cdf(threshold, lam)


# ---------------------------------------------------------------------------
# Negative binomial (over-dispersed counts: variance > mean)
# ---------------------------------------------------------------------------

def negbin_pmf(k: int, r: float, p: float) -> float:
    """Negative binomial PMF.  r = shape, p = success prob.  Mean = r(1-p)/p."""
    if r <= 0 or p <= 0 or p > 1:
        return 0.0
    log_pmf = (lgamma(k + r) - lgamma(k + 1) - lgamma(r)
               + r * log(p) + k * log(1 - p))
    return exp(log_pmf)


def negbin_cdf(k: int, r: float, p: float) -> float:
    """P(X <= k) for NegBin(r, p)."""
    return sum(negbin_pmf(i, r, p) for i in range(k + 1))


def negbin_from_mean_var(mean: float, var: float) -> Tuple[float, float]:
    """Convert (mean, variance) to negative-binomial (r, p) parameters.

    Only valid when var > mean (over-dispersed).  Returns (r, p).
    """
    if var <= mean or mean <= 0:
        return (mean, 0.5)  # degenerate fallback
    p = mean / var
    r = mean * p / (1 - p)
    return r, p


# ---------------------------------------------------------------------------
# Unified over/under with automatic distribution selection
# ---------------------------------------------------------------------------

def over_prob(lam: float, line: float,
              variance: Optional[float] = None) -> float:
    """P(X > line).

    Uses negative binomial when *variance* is provided and exceeds the mean
    (over-dispersion), otherwise falls back to Poisson.
    """
    if lam <= 0:
        return 0.0
    if variance is not None and variance > lam:
        r, p = negbin_from_mean_var(lam, variance)
        threshold = int(line)
        return 1.0 - negbin_cdf(threshold, r, p)
    return poisson_over_prob(lam, line)


def under_prob(lam: float, line: float,
               variance: Optional[float] = None) -> float:
    """P(X < line).  Complement of over_prob."""
    return 1.0 - over_prob(lam, line, variance)


def interval_prob(lam: float, low: int, high: int,
                  variance: Optional[float] = None) -> float:
    """P(low <= X <= high).

    Uses negative binomial when *variance* is provided and exceeds the mean
    (over-dispersion), otherwise falls back to Poisson.
    For open-ended upper bands (e.g. 6+), pass a large *high* value (30).
    """
    if lam <= 0:
        return 1.0 if low <= 0 else 0.0
    if variance is not None and variance > lam:
        r, p = negbin_from_mean_var(lam, variance)
        upper = negbin_cdf(high, r, p)
        lower = negbin_cdf(low - 1, r, p) if low > 0 else 0.0
        return upper - lower
    upper = poisson_cdf(high, lam)
    lower = poisson_cdf(low - 1, lam) if low > 0 else 0.0
    return upper - lower


# ---------------------------------------------------------------------------
# Implied probability from bookmaker odds
# ---------------------------------------------------------------------------

def implied_prob(decimal_odds: float) -> float:
    """Extract raw implied probability from decimal odds."""
    if decimal_odds <= 1.0:
        return 1.0
    return 1.0 / decimal_odds


def remove_vig_two_way(odds_a: float, odds_b: float) -> Tuple[float, float]:
    """Remove vig from a two-way market (over/under, home/away spread).

    Returns (fair_prob_a, fair_prob_b) summing to 1.0.
    """
    raw_a = implied_prob(odds_a)
    raw_b = implied_prob(odds_b)
    total = raw_a + raw_b
    if total <= 0:
        return (0.5, 0.5)
    return (raw_a / total, raw_b / total)


# ---------------------------------------------------------------------------
# Value edge & expected value
# ---------------------------------------------------------------------------

def value_edge(model_prob: float, fair_implied: float) -> float:
    """Value edge = model probability - fair implied probability.

    Positive means the model thinks the outcome is more likely than the
    bookmaker's price implies → potential value bet.
    """
    return model_prob - fair_implied


def expected_value(model_prob: float, decimal_odds: float) -> float:
    """EV per unit staked.  EV = (prob × odds) - 1.

    Positive EV means the bet is profitable in the long run.
    """
    return (model_prob * decimal_odds) - 1.0


# ---------------------------------------------------------------------------
# Kelly criterion
# ---------------------------------------------------------------------------

def kelly_fraction(model_prob: float, decimal_odds: float) -> float:
    """Kelly criterion optimal bet fraction.

    f* = (p·b - q) / b   where b = odds-1, q = 1-p.
    Returns 0 when there is no edge.
    """
    b = decimal_odds - 1.0
    q = 1.0 - model_prob
    if b <= 0:
        return 0.0
    f = (model_prob * b - q) / b
    return max(0.0, f)
