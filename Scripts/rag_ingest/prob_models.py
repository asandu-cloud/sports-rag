"""Probabilistic primitives for the betting RAG.

Pure functions, no side effects, no external dependencies beyond stdlib.
Provides Poisson / negative-binomial modeling for count events (goals,
corners, cards, SoT), implied probability extraction from bookmaker odds,
expected value, value edge, and Kelly criterion.
"""

from math import ceil, exp, lgamma, log, sqrt
from typing import Dict, List, Optional, Tuple

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
    """P(X < line), excluding pushes on whole-number lines."""
    if lam <= 0:
        return 1.0 if line > 0 else 0.0
    # ``ceil(line) - 1`` gives <= 2 for both an Under 2.5 and an
    # Under 3.0.  The latter intentionally excludes the push at exactly 3.
    threshold = ceil(line) - 1
    if variance is not None and variance > lam:
        r, p = negbin_from_mean_var(lam, variance)
        return negbin_cdf(threshold, r, p)
    return poisson_cdf(threshold, lam)


# ---------------------------------------------------------------------------
# Asian totals settlement
# ---------------------------------------------------------------------------

def _count_pmf(k: int, mean: float, variance: Optional[float]) -> float:
    if variance is not None and variance > mean:
        r, p = negbin_from_mean_var(mean, variance)
        return negbin_pmf(k, r, p)
    return poisson_pmf(k, mean)


def _asian_total_sub_lines(line: float) -> List[float]:
    """Split an Asian quarter total into its two half-stake component lines."""
    line = round(float(line) * 4.0) / 4.0
    whole = int(line)
    fraction = round(line - whole, 2)
    if abs(fraction - 0.25) < 0.01:
        return [float(whole), float(whole) + 0.5]
    if abs(fraction - 0.75) < 0.01:
        return [float(whole) + 0.5, float(whole) + 1.0]
    return [line]


def _asian_total_single_outcome(total: int, line: float, side: str) -> str:
    side = str(side or "").lower().strip()
    if side not in {"over", "under"}:
        raise ValueError(f"Unsupported totals side: {side!r}")
    if side == "over":
        if total > line:
            return "win"
        if total < line:
            return "loss"
        return "push"
    if total < line:
        return "win"
    if total > line:
        return "loss"
    return "push"


def _asian_total_bucket(total: int, line: float, side: str) -> str:
    outcomes = [_asian_total_single_outcome(total, sub_line, side)
                for sub_line in _asian_total_sub_lines(line)]
    if len(outcomes) == 1:
        return {"win": "full_win", "loss": "full_loss", "push": "push"}[outcomes[0]]

    wins = outcomes.count("win")
    losses = outcomes.count("loss")
    pushes = outcomes.count("push")
    if wins == 2:
        return "full_win"
    if losses == 2:
        return "full_loss"
    if pushes == 2:
        return "push"
    if wins == 1 and pushes == 1:
        return "half_win"
    if losses == 1 and pushes == 1:
        return "half_loss"
    # A valid quarter line cannot produce a one-win/one-loss split.  Keep the
    # fallback conservative if malformed provider data ever reaches here.
    return "push"


def asian_total_settlement_outcome(total: int, line: float, side: str) -> str:
    """Settle an observed total against an Asian line.

    Returns one of ``full_win``, ``half_win``, ``push``, ``half_loss`` or
    ``full_loss``. The prediction tracker uses this same settlement vocabulary
    as the probability/EV model, preventing a 2.75-line result of exactly
    three from being recorded as a full win.
    """
    try:
        observed_total = int(total)
    except (TypeError, ValueError) as exc:
        raise ValueError("Asian total settlement requires an integer observed total.") from exc
    return _asian_total_bucket(observed_total, float(line), side)


def _asian_total_max_count(mean: float, variance: Optional[float], line: float) -> int:
    dispersion = max(float(variance or mean), mean, 1.0)
    return min(500, max(40, int(ceil(mean + 12.0 * sqrt(dispersion) + 8.0)), int(ceil(line)) + 8))


def asian_total_settlement_profile(
    mean: float,
    line: float,
    side: str,
    variance: Optional[float] = None,
) -> Dict[str, float]:
    """Return full/half win, push, and loss probabilities for an Asian total.

    The profile is based on the same count distribution as the existing totals
    model, but it settles whole and quarter lines correctly.  The residual tail
    above the finite calculation window is allocated using that tail's outcome
    bucket, so all returned probabilities sum to one.
    """
    profile = {
        "full_win": 0.0,
        "half_win": 0.0,
        "push": 0.0,
        "half_loss": 0.0,
        "full_loss": 0.0,
    }
    if mean <= 0:
        profile[_asian_total_bucket(0, line, side)] = 1.0
        return profile

    max_count = _asian_total_max_count(mean, variance, line)
    assigned = 0.0
    for total in range(max_count + 1):
        probability = _count_pmf(total, mean, variance)
        assigned += probability
        profile[_asian_total_bucket(total, line, side)] += probability

    tail = max(0.0, 1.0 - assigned)
    if tail:
        profile[_asian_total_bucket(max_count + 1, line, side)] += tail

    total_probability = sum(profile.values())
    if total_probability > 0:
        for key in profile:
            profile[key] /= total_probability
    return profile


def asian_total_expected_value(profile: Dict[str, float], decimal_odds: float) -> float:
    """Expected net return per unit stake under Asian total settlement rules."""
    if decimal_odds <= 1.0:
        return -1.0
    win_profit = decimal_odds - 1.0
    return (
        profile.get("full_win", 0.0) * win_profit
        + profile.get("half_win", 0.0) * win_profit * 0.5
        - profile.get("half_loss", 0.0) * 0.5
        - profile.get("full_loss", 0.0)
    )


def asian_total_equivalent_probability(profile: Dict[str, float]) -> float:
    """Return the no-push equivalent probability used for price comparison.

    It is the win-stake fraction divided by the resolved win-or-loss stake
    fraction.  For a half line this is ordinary win probability; on Asian
    lines it correctly excludes returned stake and half-settlement effects.
    """
    win_fraction = profile.get("full_win", 0.0) + 0.5 * profile.get("half_win", 0.0)
    loss_fraction = profile.get("full_loss", 0.0) + 0.5 * profile.get("half_loss", 0.0)
    resolved = win_fraction + loss_fraction
    if resolved <= 0:
        return 0.5
    return max(0.0, min(1.0, win_fraction / resolved))


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


# ---------------------------------------------------------------------------
# Dixon-Coles bivariate Poisson model
# ---------------------------------------------------------------------------
# Standard independent Poisson assumes home and away goals are independent.
# In practice, low-scoring outcomes (0-0, 1-0, 0-1, 1-1) occur more often
# than the independent model predicts.  Dixon & Coles (1997) introduced a
# correlation parameter rho that adjusts probabilities for these scorelines.
#
# A negative rho (typical for football, around -0.10) inflates the
# probability of low-scoring draws relative to independent Poisson.
# ---------------------------------------------------------------------------

def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles tau adjustment factor for scoreline (x, y).

    Adjusts the independent Poisson probability for low-scoring outcomes:
      - (0,0): 1 - lam*mu*rho
      - (1,0): 1 + lam*rho
      - (0,1): 1 + mu*rho
      - (1,1): 1 - rho
      - otherwise: 1  (no adjustment)

    Parameters
    ----------
    x : int   -- home goals
    y : int   -- away goals
    lam : float -- home expected goals (lambda)
    mu : float  -- away expected goals
    rho : float -- correlation parameter (typically -0.15 to -0.05)
    """
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 1 and y == 0:
        return 1.0 + lam * rho
    if x == 0 and y == 1:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def dixon_coles_scoreline_prob(home_goals: int, away_goals: int,
                               lambda_h: float, lambda_a: float,
                               rho: float = -0.10) -> float:
    """Probability of a single scoreline under the Dixon-Coles model.

    Computes independent Poisson probabilities for each team, then applies
    the tau adjustment for scorelines where both teams score 0 or 1.

    Parameters
    ----------
    home_goals : int  -- home team goals
    away_goals : int  -- away team goals
    lambda_h : float  -- home team expected goals
    lambda_a : float  -- away team expected goals
    rho : float       -- correlation parameter (default -0.10)

    Returns
    -------
    float -- adjusted scoreline probability
    """
    p_home = poisson_pmf(home_goals, lambda_h)
    p_away = poisson_pmf(away_goals, lambda_a)
    tau = _tau(home_goals, away_goals, lambda_h, lambda_a, rho)
    return p_home * p_away * tau


def dixon_coles_scoreline_matrix(lambda_h: float, lambda_a: float,
                                 rho: float = -0.10,
                                 max_goals: int = 7) -> list:
    """Full scoreline probability matrix under Dixon-Coles.

    Returns a nested list of shape (max_goals+1) x (max_goals+1) where
    matrix[i][j] = P(Home=i, Away=j).  The matrix is normalized so that
    all probabilities sum to 1.0.

    Parameters
    ----------
    lambda_h : float  -- home team expected goals
    lambda_a : float  -- away team expected goals
    rho : float       -- correlation parameter (default -0.10)
    max_goals : int   -- maximum goals per team to consider (default 7)

    Returns
    -------
    list[list[float]] -- (max_goals+1) x (max_goals+1) probability matrix
    """
    size = max_goals + 1
    matrix = [[0.0] * size for _ in range(size)]
    total = 0.0
    for i in range(size):
        for j in range(size):
            p = dixon_coles_scoreline_prob(i, j, lambda_h, lambda_a, rho)
            matrix[i][j] = p
            total += p
    # Normalize to account for truncation at max_goals
    if total > 0:
        for i in range(size):
            for j in range(size):
                matrix[i][j] /= total
    return matrix


def dixon_coles_match_probs(lambda_h: float, lambda_a: float,
                            rho: float = -0.10) -> Tuple[float, float, float]:
    """Match outcome probabilities under Dixon-Coles.

    Returns (p_home_win, p_draw, p_away_win) by summing the scoreline
    matrix.  More accurate than independent Poisson for 1X2 markets
    because it accounts for the correlation in low-scoring outcomes.

    Parameters
    ----------
    lambda_h : float  -- home team expected goals
    lambda_a : float  -- away team expected goals
    rho : float       -- correlation parameter (default -0.10)
    """
    matrix = dixon_coles_scoreline_matrix(lambda_h, lambda_a, rho)
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if i > j:
                p_home += matrix[i][j]
            elif i == j:
                p_draw += matrix[i][j]
            else:
                p_away += matrix[i][j]
    return (p_home, p_draw, p_away)


def dixon_coles_btts_prob(lambda_h: float, lambda_a: float,
                          rho: float = -0.10) -> float:
    """P(Both Teams To Score) under Dixon-Coles.

    Sums all scoreline probabilities where home >= 1 AND away >= 1.
    More accurate than the independent approximation P(H>=1)*P(A>=1)
    because Dixon-Coles adjusts the (0,0), (1,0), (0,1) cells.

    Parameters
    ----------
    lambda_h : float  -- home team expected goals
    lambda_a : float  -- away team expected goals
    rho : float       -- correlation parameter (default -0.10)
    """
    matrix = dixon_coles_scoreline_matrix(lambda_h, lambda_a, rho)
    btts = 0.0
    for i in range(1, len(matrix)):
        for j in range(1, len(matrix[i])):
            btts += matrix[i][j]
    return btts


def dixon_coles_total_goals_prob(lambda_h: float, lambda_a: float,
                                 line: float,
                                 rho: float = -0.10) -> float:
    """P(total goals > line) under Dixon-Coles.

    Sums all scoreline probabilities where i + j > line.  For half-integer
    lines (e.g. 2.5), this is equivalent to P(total >= 3).

    Parameters
    ----------
    lambda_h : float  -- home team expected goals
    lambda_a : float  -- away team expected goals
    line : float      -- the over/under line (e.g. 2.5)
    rho : float       -- correlation parameter (default -0.10)
    """
    matrix = dixon_coles_scoreline_matrix(lambda_h, lambda_a, rho)
    threshold = int(line)
    prob = 0.0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if i + j > threshold:
                prob += matrix[i][j]
    return prob
