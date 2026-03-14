"""Core live betting projection engine.

Updates pre-match Poisson projections with live match data using
Bayesian conjugate updating (Gamma-Poisson), pace-decay corrections,
situational adjustments, shot-based xG estimation, score-state
transition modeling, and Poisson goodness-of-fit checking.

Mathematical foundations
------------------------
The pre-match model produces a rate parameter lambda for each stat
(goals, corners, cards, SoT).  As the match progresses and we observe
actual events, we update lambda via the Gamma-Poisson conjugate:

    Prior:     Gamma(alpha_0, beta_0)
               where alpha_0 = lambda * S, beta_0 = S
               and S = prior_strength (conceptual "equivalent matches")

    Posterior: Gamma(alpha_0 + observed, beta_0 + elapsed_fraction)

    Remaining expected count = posterior_mean * remaining_fraction
    where posterior_mean = (alpha_0 + observed) / (beta_0 + elapsed_fraction)

This smoothly transitions from the pre-match projection at minute 0
to the observed pace as the match progresses.

Usage
-----
    from live_state import LiveMatchState
    from live_engine import compute_snapshot

    state = LiveMatchState(...)  # populated from API polling
    snapshot = compute_snapshot(state)
"""

import sys
import os
from datetime import datetime, timezone
from math import exp, log, lgamma, sqrt, pi, erf
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup for sibling package imports
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RAG_INGEST_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "rag_ingest")
if _RAG_INGEST_DIR not in sys.path:
    sys.path.insert(0, _RAG_INGEST_DIR)

from prob_models import (
    over_prob,
    under_prob,
    poisson_pmf,
    poisson_cdf,
    interval_prob,
)

from live_config import (
    LIVE_PRIOR_STRENGTH,
    SITUATIONAL,
    STAT_TIME_DISTRIBUTIONS,
    XG_SOT_CONVERSION,
    XG_OFF_TARGET_RATE,
    MOMENTUM_WEIGHTS,
    MOMENTUM_WINDOW_MIN,
)
from live_state import LiveMatchState, LiveSnapshot, MatchEvent


# ===================================================================
#  A.  BAYESIAN RATE UPDATE (Gamma-Poisson Conjugate)
# ===================================================================

def bayesian_update_lambda(
    pre_match_lambda: float,
    elapsed_min: float,
    observed_count: int,
    total_min: float = 90.0,
    prior_strength: float = 6.0,
) -> float:
    """Compute the posterior expected remaining count for a Poisson stat.

    Uses Gamma-Poisson conjugate updating:

        alpha_0 = pre_match_lambda * prior_strength
        beta_0  = prior_strength
        elapsed_fraction = elapsed_min / total_min

        Posterior parameters:
            alpha_n = alpha_0 + observed_count
            beta_n  = beta_0  + elapsed_fraction

        Posterior rate (per full match):
            lambda_post = alpha_n / beta_n

        Remaining expected count:
            remaining = lambda_post * remaining_fraction

    At minute 0 this returns the full pre-match projection.  As the
    match progresses, the posterior rate smoothly converges toward the
    observed pace.

    Parameters
    ----------
    pre_match_lambda : float
        Pre-match projected total for the stat (e.g. 2.7 goals).
    elapsed_min : float
        Minutes played so far (0-90+).
    observed_count : int
        Number of events observed so far.
    total_min : float
        Total match duration in minutes (default 90).
    prior_strength : float
        Conceptual number of "equivalent matches" the prior is worth.
        Higher = trust the prior longer.

    Returns
    -------
    float
        Expected remaining count for the rest of the match.
    """
    if pre_match_lambda <= 0:
        return 0.0
    if elapsed_min >= total_min:
        return 0.0

    elapsed_fraction = min(elapsed_min / total_min, 1.0)
    remaining_fraction = 1.0 - elapsed_fraction

    alpha_0 = pre_match_lambda * prior_strength
    beta_0 = prior_strength

    alpha_n = alpha_0 + observed_count
    beta_n = beta_0 + elapsed_fraction

    posterior_rate = alpha_n / beta_n
    remaining = posterior_rate * remaining_fraction

    return max(0.0, remaining)


# ===================================================================
#  B.  SITUATIONAL ADJUSTMENTS
# ===================================================================

def situational_adjustments(
    state: LiveMatchState,
    stat: str,
    base_remaining_lambda: float,
) -> float:
    """Apply game-state modifiers to the Bayesian posterior.

    Captures effects that a simple Poisson model does not handle:
    - Red cards reduce the sent-off team's attacking output and
      increase card accumulation for the remaining players.
    - Tight finishes (within 1 goal) late in the match produce
      more tactical fouls and therefore more cards.
    - Trailing teams push forward late, generating more corners.
    - Second halves historically see ~10% more goals than first halves.

    Parameters
    ----------
    state : LiveMatchState
        Current match state.
    stat : str
        One of "goals", "corners", "cards", "sot".
    base_remaining_lambda : float
        Posterior remaining lambda before situational adjustments.

    Returns
    -------
    float
        Adjusted remaining lambda.
    """
    adjusted = base_remaining_lambda
    cfg = SITUATIONAL

    total_reds = state.red_cards_home + state.red_cards_away

    # --- Red card effects ---
    if total_reds > 0:
        if stat in ("goals", "sot"):
            # Each red card reduces remaining attacking output
            adjusted *= cfg["red_card_attack_reduction"] ** total_reds
        elif stat == "cards":
            # More fouls from reduced-man team
            adjusted *= cfg["red_card_card_increase"] ** total_reds

    # --- Tight finish (within 1 goal, minute >= 70) ---
    if stat == "cards" and state.elapsed_min >= 70:
        if abs(state.goal_diff) <= 1:
            adjusted *= cfg["tight_finish_card_increase"]

    # --- Trailing team pushing forward (75'+) ---
    if stat == "corners" and state.elapsed_min >= 75:
        if state.goal_diff != 0:
            # At least one team is trailing and chasing
            adjusted *= cfg["trailing_corner_increase"]

    # --- Second half goal boost ---
    if stat == "goals" and state.is_second_half:
        adjusted *= cfg["second_half_goal_boost"]

    return max(0.0, adjusted)


# ===================================================================
#  C.  LIVE PROBABILITY FUNCTIONS
# ===================================================================

def live_over_prob(
    pre_match_lambda: float,
    elapsed_min: float,
    observed_count: int,
    line: float,
    stat: str,
    state: LiveMatchState,
) -> float:
    """P(total > line) given the live match state.

    Computes the posterior remaining lambda, applies situational
    adjustments, then uses the Poisson CDF on the *remaining* count
    needed to clear the line.

    Example: line=2.5, observed=1 goal, remaining_lambda=1.4
    Need 2+ more goals -> P(remaining >= 2) = 1 - P(remaining <= 1)

    Parameters
    ----------
    pre_match_lambda : float
        Pre-match projected total.
    elapsed_min : float
        Minutes played.
    observed_count : int
        Events observed so far.
    line : float
        The betting line (e.g. 2.5 for Over 2.5 Goals).
    stat : str
        Stat type ("goals", "corners", "cards", "sot").
    state : LiveMatchState
        Full match state for situational adjustments.

    Returns
    -------
    float
        Probability of finishing over the line.
    """
    # If already over the line, probability = 1.0
    if observed_count > line:
        return 1.0

    prior_strength = LIVE_PRIOR_STRENGTH.get(stat, 5.0)
    remaining_lambda = bayesian_update_lambda(
        pre_match_lambda, elapsed_min, observed_count,
        prior_strength=prior_strength,
    )
    remaining_lambda = situational_adjustments(state, stat, remaining_lambda)

    # Apply pace-decay correction
    remaining_lambda = _apply_pace_decay(
        remaining_lambda, elapsed_min, stat, pre_match_lambda,
    )

    # How many more do we need to clear the line?
    need = line - observed_count
    if need < 0:
        return 1.0

    # P(remaining > need) using Poisson on the remaining count
    return over_prob(remaining_lambda, need)


def live_under_prob(
    pre_match_lambda: float,
    elapsed_min: float,
    observed_count: int,
    line: float,
    stat: str,
    state: LiveMatchState,
) -> float:
    """P(total < line) = 1 - P(total > line).  Convenience wrapper."""
    return 1.0 - live_over_prob(
        pre_match_lambda, elapsed_min, observed_count, line, stat, state,
    )


def live_btts_prob(
    prior_home: float,
    prior_away: float,
    elapsed_min: float,
    goals_home: int,
    goals_away: int,
    state: LiveMatchState,
) -> float:
    """Live P(Both Teams To Score).

    If both teams have already scored, returns 1.0.
    Otherwise, computes per-team remaining scoring probability using
    the Bayesian posterior and returns P(home >= 1) * P(away >= 1)
    for the remaining portion of the match.

    Parameters
    ----------
    prior_home : float
        Pre-match projected goals for the home team.
    prior_away : float
        Pre-match projected goals for the away team.
    elapsed_min : float
        Minutes played.
    goals_home, goals_away : int
        Goals scored so far by each team.
    state : LiveMatchState
        Full match state for situational adjustments.

    Returns
    -------
    float
        P(BTTS Yes) given live data.
    """
    if goals_home > 0 and goals_away > 0:
        return 1.0

    prior_strength = LIVE_PRIOR_STRENGTH["goals"]

    # Home team remaining lambda
    rem_home = bayesian_update_lambda(
        prior_home, elapsed_min, goals_home,
        prior_strength=prior_strength,
    )
    rem_home = situational_adjustments(state, "goals", rem_home)
    rem_home = _apply_pace_decay(rem_home, elapsed_min, "goals", prior_home)

    # Away team remaining lambda
    rem_away = bayesian_update_lambda(
        prior_away, elapsed_min, goals_away,
        prior_strength=prior_strength,
    )
    rem_away = situational_adjustments(state, "goals", rem_away)
    rem_away = _apply_pace_decay(rem_away, elapsed_min, "goals", prior_away)

    # P(team scores at least 1 more)
    p_home_scores = 1.0 - poisson_pmf(0, rem_home) if goals_home == 0 else 1.0
    p_away_scores = 1.0 - poisson_pmf(0, rem_away) if goals_away == 0 else 1.0

    return p_home_scores * p_away_scores


def live_moneyline_probs(
    prior_home: float,
    prior_away: float,
    elapsed_min: float,
    goals_home: int,
    goals_away: int,
    state: LiveMatchState,
    max_remaining: int = 6,
) -> Dict[str, float]:
    """Live 3-way moneyline probabilities (Home Win / Draw / Away Win).

    Uses the Bayesian posterior for each team's remaining goals,
    then enumerates all (remaining_home, remaining_away) scoreline
    combinations to compute P(Home Win), P(Draw), P(Away Win).

    Parameters
    ----------
    prior_home, prior_away : float
        Pre-match projected goals per team.
    elapsed_min : float
        Minutes played.
    goals_home, goals_away : int
        Current score.
    state : LiveMatchState
        Full match state for situational adjustments.
    max_remaining : int
        Maximum remaining goals to enumerate per team (default 6).

    Returns
    -------
    dict
        {"home_win": float, "draw": float, "away_win": float}
    """
    prior_strength = LIVE_PRIOR_STRENGTH["goals"]

    rem_home = bayesian_update_lambda(
        prior_home, elapsed_min, goals_home,
        prior_strength=prior_strength,
    )
    rem_home = situational_adjustments(state, "goals", rem_home)
    rem_home = _apply_pace_decay(rem_home, elapsed_min, "goals", prior_home)

    rem_away = bayesian_update_lambda(
        prior_away, elapsed_min, goals_away,
        prior_strength=prior_strength,
    )
    rem_away = situational_adjustments(state, "goals", rem_away)
    rem_away = _apply_pace_decay(rem_away, elapsed_min, "goals", prior_away)

    p_home_win = 0.0
    p_draw = 0.0
    p_away_win = 0.0

    for rh in range(max_remaining + 1):
        ph = poisson_pmf(rh, rem_home)
        for ra in range(max_remaining + 1):
            pa = poisson_pmf(ra, rem_away)
            joint = ph * pa
            final_home = goals_home + rh
            final_away = goals_away + ra
            if final_home > final_away:
                p_home_win += joint
            elif final_home == final_away:
                p_draw += joint
            else:
                p_away_win += joint

    # Normalize to sum to 1.0 (rounding from truncation)
    total = p_home_win + p_draw + p_away_win
    if total > 0:
        p_home_win /= total
        p_draw /= total
        p_away_win /= total

    return {
        "home_win": round(p_home_win, 4),
        "draw": round(p_draw, 4),
        "away_win": round(p_away_win, 4),
    }


def live_correct_score_probs(
    prior_home: float,
    prior_away: float,
    elapsed_min: float,
    goals_home: int,
    goals_away: int,
    state: LiveMatchState,
    max_remaining: int = 5,
) -> Dict[str, float]:
    """Live correct-score probabilities.

    Given the current score and remaining goal lambdas, enumerates
    all possible final scorelines and returns their probabilities.

    Parameters
    ----------
    prior_home, prior_away : float
        Pre-match projected goals per team.
    elapsed_min : float
        Minutes played.
    goals_home, goals_away : int
        Current score.
    state : LiveMatchState
        Full match state for situational adjustments.
    max_remaining : int
        Maximum remaining goals per team to enumerate (default 5).

    Returns
    -------
    dict
        {"3-1": 0.12, "2-1": 0.18, ...} sorted by probability descending.
    """
    prior_strength = LIVE_PRIOR_STRENGTH["goals"]

    rem_home = bayesian_update_lambda(
        prior_home, elapsed_min, goals_home,
        prior_strength=prior_strength,
    )
    rem_home = situational_adjustments(state, "goals", rem_home)
    rem_home = _apply_pace_decay(rem_home, elapsed_min, "goals", prior_home)

    rem_away = bayesian_update_lambda(
        prior_away, elapsed_min, goals_away,
        prior_strength=prior_strength,
    )
    rem_away = situational_adjustments(state, "goals", rem_away)
    rem_away = _apply_pace_decay(rem_away, elapsed_min, "goals", prior_away)

    scores: Dict[str, float] = {}
    for rh in range(max_remaining + 1):
        ph = poisson_pmf(rh, rem_home)
        for ra in range(max_remaining + 1):
            pa = poisson_pmf(ra, rem_away)
            final_h = goals_home + rh
            final_a = goals_away + ra
            key = f"{final_h}-{final_a}"
            scores[key] = ph * pa

    # Sort by probability descending
    sorted_scores = dict(
        sorted(scores.items(), key=lambda x: x[1], reverse=True)
    )
    return sorted_scores


# ===================================================================
#  D.  MOMENTUM SCORE
# ===================================================================

def momentum_score(
    state: LiveMatchState,
    window_min: int = MOMENTUM_WINDOW_MIN,
) -> Dict[str, float]:
    """Compute a momentum share between home and away teams.

    Considers recent events (last *window_min* minutes) plus current
    possession to produce a 0-1 share for each side.

    Weights are defined in MOMENTUM_WEIGHTS (live_config.py):
    - shots: raw shot count ratio
    - possession: current possession percentage
    - corners: recent corner ratio
    - sot: shots on target ratio
    - recent_goals: heavily weighted (a recent goal is a strong signal)
    - cards_against: opponent cards slightly boost momentum

    Parameters
    ----------
    state : LiveMatchState
        Current match state.
    window_min : int
        Minutes to look back for event-based signals (default 15).

    Returns
    -------
    dict
        {"home": float, "away": float} summing to 1.0.
    """
    if state.elapsed_min <= 0:
        return {"home": 0.50, "away": 0.50}

    weights = MOMENTUM_WEIGHTS
    home_score = 0.0
    away_score = 0.0

    # --- Shot ratio (cumulative) ---
    total_shots = state.shots_home + state.shots_away
    if total_shots > 0:
        shot_share_home = state.shots_home / total_shots
        home_score += weights["shots"] * shot_share_home
        away_score += weights["shots"] * (1 - shot_share_home)
    else:
        home_score += weights["shots"] * 0.5
        away_score += weights["shots"] * 0.5

    # --- SoT ratio (cumulative) ---
    total_sot = state.sot_home + state.sot_away
    if total_sot > 0:
        sot_share_home = state.sot_home / total_sot
        home_score += weights["sot"] * sot_share_home
        away_score += weights["sot"] * (1 - sot_share_home)
    else:
        home_score += weights["sot"] * 0.5
        away_score += weights["sot"] * 0.5

    # --- Possession ---
    home_score += weights["possession"] * (state.possession_home / 100.0)
    away_score += weights["possession"] * (state.possession_away / 100.0)

    # --- Recent corners ---
    recent = state.events_in_window(window_min)
    recent_corners_home = sum(
        1 for e in recent if e.event_type == "corner" and e.team_side == "home"
    )
    recent_corners_away = sum(
        1 for e in recent if e.event_type == "corner" and e.team_side == "away"
    )
    total_recent_corners = recent_corners_home + recent_corners_away
    if total_recent_corners > 0:
        home_score += weights["corners"] * (recent_corners_home / total_recent_corners)
        away_score += weights["corners"] * (recent_corners_away / total_recent_corners)
    else:
        home_score += weights["corners"] * 0.5
        away_score += weights["corners"] * 0.5

    # --- Recent goals (strong signal) ---
    recent_goals_home = sum(
        1 for e in recent if e.event_type == "goal" and e.team_side == "home"
    )
    recent_goals_away = sum(
        1 for e in recent if e.event_type == "goal" and e.team_side == "away"
    )
    home_score += weights["recent_goals"] * recent_goals_home
    away_score += weights["recent_goals"] * recent_goals_away

    # --- Cards against opponent (negative signal for recipient) ---
    recent_cards_home = sum(
        1 for e in recent
        if e.event_type in ("yellow_card", "red_card") and e.team_side == "home"
    )
    recent_cards_away = sum(
        1 for e in recent
        if e.event_type in ("yellow_card", "red_card") and e.team_side == "away"
    )
    # Cards against home team reduce home momentum (opponent gains)
    away_score += abs(weights["cards_against"]) * recent_cards_home
    home_score += abs(weights["cards_against"]) * recent_cards_away

    # Normalize to [0, 1] shares
    total = home_score + away_score
    if total <= 0:
        return {"home": 0.50, "away": 0.50}

    return {
        "home": round(home_score / total, 4),
        "away": round(away_score / total, 4),
    }


# ===================================================================
#  E.  EXTRA EDGE: Statistical Methods Beyond Basic Bayesian
# ===================================================================

# -------------------------------------------------------------------
# E1. Shot-Based xG Live Adjustment
# -------------------------------------------------------------------

def live_xg_estimate(state: LiveMatchState) -> Tuple[float, float]:
    """Estimate live expected goals from shot data.

    Football analytics shows shots on target convert at ~30% and
    off-target shots at ~3% (deflections, keeper errors, etc.).

    This creates an xG figure independent of the actual score:
    - Arsenal 8 SoT, 0 goals -> xG ~2.4 (underperforming, regression likely)
    - Chelsea 1 SoT, 2 goals -> xG ~0.33 (overperforming, regression likely)

    The xG estimate can be blended with the score-based projection
    to create a more robust moneyline estimate.

    Parameters
    ----------
    state : LiveMatchState
        Current match state with shot data.

    Returns
    -------
    tuple of (float, float)
        (xg_home, xg_away) — expected goals from shot quality.
    """
    off_target_home = max(0, state.shots_home - state.sot_home)
    off_target_away = max(0, state.shots_away - state.sot_away)

    xg_home = (state.sot_home * XG_SOT_CONVERSION
               + off_target_home * XG_OFF_TARGET_RATE)
    xg_away = (state.sot_away * XG_SOT_CONVERSION
               + off_target_away * XG_OFF_TARGET_RATE)

    return (round(xg_home, 3), round(xg_away, 3))


def xg_adjusted_moneyline(
    state: LiveMatchState,
    score_moneyline: Dict[str, float],
    xg_weight: float = 0.30,
) -> Dict[str, float]:
    """Blend score-based and xG-based moneyline probabilities.

    When the xG differs significantly from the actual score, this
    provides a "regression-to-the-mean" correction.  A team that is
    dominating on shots but behind on the scoreboard will get a
    probability boost.

    Parameters
    ----------
    state : LiveMatchState
        Current match state.
    score_moneyline : dict
        Score-based moneyline from live_moneyline_probs().
    xg_weight : float
        Weight given to the xG-based moneyline (default 0.30).

    Returns
    -------
    dict
        Blended {"home_win": P, "draw": P, "away_win": P}.
    """
    xg_home, xg_away = live_xg_estimate(state)

    if state.elapsed_min <= 0:
        return score_moneyline

    # Build xG-based moneyline using remaining xG as a score proxy
    # Treat xG as "true score" and compute moneyline from it
    prior_strength = LIVE_PRIOR_STRENGTH["goals"]
    prior_home = state.prior_goals_home or 1.35
    prior_away = state.prior_goals_away or 1.35

    # Use xG as the "observed score" for a parallel moneyline calc
    # Scale xG to integer-ish for the enumeration
    xg_ml = live_moneyline_probs(
        prior_home, prior_away,
        state.elapsed_min,
        # Use floor of xG as "virtual goals" for the xG-based view
        int(xg_home), int(xg_away),
        state,
    )

    # Blend
    blended = {}
    for key in ("home_win", "draw", "away_win"):
        blended[key] = round(
            (1 - xg_weight) * score_moneyline.get(key, 0.33)
            + xg_weight * xg_ml.get(key, 0.33),
            4,
        )

    # Normalize
    total = sum(blended.values())
    if total > 0:
        for key in blended:
            blended[key] = round(blended[key] / total, 4)

    return blended


# -------------------------------------------------------------------
# E2. Pace Decay Model
# -------------------------------------------------------------------

def pace_decay_adjustment(elapsed_min: float, stat: str) -> float:
    """Compute the fraction of remaining match "action" using empirical
    time distributions.

    Football events are NOT uniformly distributed across 90 minutes.
    Goals cluster in the final 15 minutes, cards accelerate late due
    to tactical fouling, etc.

    Instead of a naive ``remaining_fraction = remaining_min / 90``,
    this function sums the distribution weights for time windows that
    have not yet been played.

    Parameters
    ----------
    elapsed_min : float
        Minutes played so far.
    stat : str
        One of "goals", "corners", "cards", "sot".

    Returns
    -------
    float
        Fraction of total expected events that remain (0.0 to 1.0).
        If the stat distribution is unknown, falls back to uniform.
    """
    dist = STAT_TIME_DISTRIBUTIONS.get(stat)
    if dist is None:
        # Fallback to uniform
        return max(0.0, (90.0 - elapsed_min) / 90.0)

    remaining_weight = 0.0
    elapsed_weight = 0.0

    for (lo, hi), weight in dist.items():
        if elapsed_min >= hi:
            # This window is fully elapsed
            elapsed_weight += weight
        elif elapsed_min <= lo:
            # This window is fully remaining
            remaining_weight += weight
        else:
            # Partially elapsed window — linear interpolation
            window_len = hi - lo
            remaining_in_window = hi - elapsed_min
            fraction_remaining = remaining_in_window / window_len
            remaining_weight += weight * fraction_remaining
            elapsed_weight += weight * (1 - fraction_remaining)

    total = elapsed_weight + remaining_weight
    if total <= 0:
        return 0.0

    return remaining_weight / total


def _apply_pace_decay(
    remaining_lambda: float,
    elapsed_min: float,
    stat: str,
    pre_match_lambda: float,
) -> float:
    """Adjust remaining_lambda using the pace-decay model.

    The Bayesian update assumes uniform event distribution.  This
    corrects for the actual time distribution by scaling the remaining
    lambda proportionally.

    If the pace-decay says 60% of the action remains but a uniform
    model would say 50%, we scale up by 60%/50% = 1.2x.

    Parameters
    ----------
    remaining_lambda : float
        Bayesian posterior remaining lambda (uniform assumption).
    elapsed_min : float
        Minutes played.
    stat : str
        Stat type.
    pre_match_lambda : float
        Original pre-match lambda (for ratio computation).

    Returns
    -------
    float
        Pace-decay-adjusted remaining lambda.
    """
    if elapsed_min <= 0 or elapsed_min >= 90:
        return remaining_lambda

    uniform_remaining = (90.0 - elapsed_min) / 90.0
    decay_remaining = pace_decay_adjustment(elapsed_min, stat)

    if uniform_remaining <= 0:
        return remaining_lambda

    # Ratio: if decay says more action remains than uniform assumes,
    # scale up; if less, scale down
    adjustment_ratio = decay_remaining / uniform_remaining
    return remaining_lambda * adjustment_ratio


# -------------------------------------------------------------------
# E3. Score-State Transition Model
# -------------------------------------------------------------------

def score_state_probabilities(state: LiveMatchState) -> Dict[str, float]:
    """Compute next-goal transition probabilities and expected time.

    Given the current score and remaining goal rates, computes:
    - P(next goal is scored by home team)
    - P(next goal is scored by away team)
    - P(no more goals in the match)
    - Expected minutes until the next goal

    These are useful for "next goal" markets, cashout decisions, and
    assessing whether a pre-match bet is still alive.

    The model treats each team's remaining goals as independent Poisson
    processes with rates proportional to remaining time.

    Parameters
    ----------
    state : LiveMatchState
        Current match state.

    Returns
    -------
    dict
        {
            "next_home": float,     # P(next goal is home)
            "next_away": float,     # P(next goal is away)
            "no_more": float,       # P(no more goals)
            "expected_min_to_next": float  # expected minutes to next goal
        }
    """
    prior_home = state.prior_goals_home or 1.35
    prior_away = state.prior_goals_away or 1.35
    prior_strength = LIVE_PRIOR_STRENGTH["goals"]

    rem_home = bayesian_update_lambda(
        prior_home, state.elapsed_min, state.goals_home,
        prior_strength=prior_strength,
    )
    rem_home = situational_adjustments(state, "goals", rem_home)
    rem_home = _apply_pace_decay(rem_home, state.elapsed_min, "goals", prior_home)

    rem_away = bayesian_update_lambda(
        prior_away, state.elapsed_min, state.goals_away,
        prior_strength=prior_strength,
    )
    rem_away = situational_adjustments(state, "goals", rem_away)
    rem_away = _apply_pace_decay(rem_away, state.elapsed_min, "goals", prior_away)

    total_rem = rem_home + rem_away

    # P(no more goals) = P(home=0) * P(away=0)
    p_no_more = poisson_pmf(0, rem_home) * poisson_pmf(0, rem_away)

    # P(at least one goal) * P(it's home | at least one goal)
    # For competing Poisson processes, P(next is home) = rate_home / total_rate
    if total_rem > 0:
        p_next_home = (1 - p_no_more) * (rem_home / total_rem)
        p_next_away = (1 - p_no_more) * (rem_away / total_rem)
    else:
        p_next_home = 0.0
        p_next_away = 0.0

    # Expected time to next goal (exponential inter-arrival)
    remaining_min = max(1.0, 90.0 - state.elapsed_min)
    if total_rem > 0:
        # Rate per minute = total_rem / remaining_min
        rate_per_min = total_rem / remaining_min
        expected_wait = 1.0 / rate_per_min  # mean of exponential
    else:
        expected_wait = remaining_min  # no goals expected

    return {
        "next_home": round(p_next_home, 4),
        "next_away": round(p_next_away, 4),
        "no_more": round(p_no_more, 4),
        "expected_min_to_next": round(min(expected_wait, remaining_min), 1),
    }


# -------------------------------------------------------------------
# E4. Poisson Process Goodness-of-Fit Check
# -------------------------------------------------------------------

def poisson_fit_quality(
    pre_match_lambda: float,
    elapsed_min: float,
    observed: int,
) -> float:
    """Assess how well the observed data fits the pre-match Poisson model.

    Uses the Poisson CDF to compute a two-sided p-value-like score.
    If the observed count is highly improbable under the prior, the
    fit quality is low, signaling that the pre-match model may be
    miscalibrated for this match.

    A low fit quality (< 0.3) suggests the match is behaving unusually
    and the engine should rely more on observed pace (lower effective
    prior_strength).

    Parameters
    ----------
    pre_match_lambda : float
        Pre-match projected total for the full match.
    elapsed_min : float
        Minutes played.
    observed : int
        Number of events observed so far.

    Returns
    -------
    float
        Score in [0, 1] where 1.0 = perfect fit, 0.0 = extreme deviation.
    """
    if elapsed_min <= 0 or pre_match_lambda <= 0:
        return 1.0  # no data yet, assume good fit

    elapsed_fraction = min(elapsed_min / 90.0, 1.0)
    expected_so_far = pre_match_lambda * elapsed_fraction

    if expected_so_far <= 0:
        return 1.0

    # Two-sided p-value: how extreme is the observation?
    # P(X <= observed) and P(X >= observed) under Poisson(expected_so_far)
    p_lower = poisson_cdf(observed, expected_so_far)
    p_upper = 1.0 - poisson_cdf(max(0, observed - 1), expected_so_far)

    # Two-sided p-value: 2 * min(p_lower, p_upper), capped at 1.0
    p_value = min(1.0, 2.0 * min(p_lower, p_upper))

    # Transform to a 0-1 quality score
    # p_value near 1.0 = observation is typical -> high quality
    # p_value near 0.0 = observation is extreme -> low quality
    return round(p_value, 4)


# ===================================================================
#  F.  COMPUTE FULL SNAPSHOT
# ===================================================================

# Standard lines for probability computation
_GOAL_LINES   = [0.5, 1.5, 2.5, 3.5, 4.5]
_CORNER_LINES = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
_CARD_LINES   = [2.5, 3.5, 4.5, 5.5]
_SOT_LINES    = [6.5, 7.5, 8.5, 9.5]


def compute_snapshot(state: LiveMatchState) -> LiveSnapshot:
    """Orchestrate all projection methods into a complete LiveSnapshot.

    This is the main entry point for the live engine.  Given a
    LiveMatchState (populated from API polling), it:

    1. Computes Bayesian posterior remaining lambda for each stat
    2. Applies situational adjustments and pace-decay corrections
    3. Calculates over/under probabilities for standard lines
    4. Computes live BTTS and moneyline probabilities
    5. Estimates momentum score
    6. Computes xG-based adjustments
    7. Computes score-state transition probabilities
    8. Assesses Poisson fit quality for each stat

    Parameters
    ----------
    state : LiveMatchState
        Current match state, fully populated.

    Returns
    -------
    LiveSnapshot
        Complete projection output ready for clients.
    """
    elapsed = state.elapsed_min
    prior_strength_goals   = LIVE_PRIOR_STRENGTH["goals"]
    prior_strength_corners = LIVE_PRIOR_STRENGTH["corners"]
    prior_strength_cards   = LIVE_PRIOR_STRENGTH["cards"]
    prior_strength_sot     = LIVE_PRIOR_STRENGTH["sot"]

    # --- Priors (with safe defaults) ---
    prior_goals   = state.prior_goals or 2.7
    prior_corners = state.prior_corners or 10.5
    prior_cards   = state.prior_cards or 4.1
    prior_sot     = state.prior_sot or 8.5
    prior_goals_h = state.prior_goals_home or (prior_goals * 0.53)  # slight home bias
    prior_goals_a = state.prior_goals_away or (prior_goals * 0.47)

    # --- Bayesian remaining lambdas ---
    rem_goals = bayesian_update_lambda(
        prior_goals, elapsed, state.total_goals,
        prior_strength=prior_strength_goals,
    )
    rem_goals = situational_adjustments(state, "goals", rem_goals)
    rem_goals = _apply_pace_decay(rem_goals, elapsed, "goals", prior_goals)

    rem_corners = bayesian_update_lambda(
        prior_corners, elapsed, state.total_corners,
        prior_strength=prior_strength_corners,
    )
    rem_corners = situational_adjustments(state, "corners", rem_corners)
    rem_corners = _apply_pace_decay(rem_corners, elapsed, "corners", prior_corners)

    rem_cards = bayesian_update_lambda(
        prior_cards, elapsed, state.total_cards,
        prior_strength=prior_strength_cards,
    )
    rem_cards = situational_adjustments(state, "cards", rem_cards)
    rem_cards = _apply_pace_decay(rem_cards, elapsed, "cards", prior_cards)

    rem_sot = bayesian_update_lambda(
        prior_sot, elapsed, state.total_sot,
        prior_strength=prior_strength_sot,
    )
    rem_sot = situational_adjustments(state, "sot", rem_sot)
    rem_sot = _apply_pace_decay(rem_sot, elapsed, "sot", prior_sot)

    # --- Projected totals (observed + remaining) ---
    proj_goals   = state.total_goals + rem_goals
    proj_corners = state.total_corners + rem_corners
    proj_cards   = state.total_cards + rem_cards
    proj_sot     = state.total_sot + rem_sot

    # --- Over probabilities for standard lines ---
    def _build_over_probs(prior_lam, observed, lines, stat):
        result = {}
        for line in lines:
            p = live_over_prob(prior_lam, elapsed, observed, line, stat, state)
            result[str(line)] = round(p, 4)
        return result

    prob_over_goals   = _build_over_probs(prior_goals, state.total_goals, _GOAL_LINES, "goals")
    prob_over_corners = _build_over_probs(prior_corners, state.total_corners, _CORNER_LINES, "corners")
    prob_over_cards   = _build_over_probs(prior_cards, state.total_cards, _CARD_LINES, "cards")
    prob_over_sot     = _build_over_probs(prior_sot, state.total_sot, _SOT_LINES, "sot")

    # --- BTTS ---
    btts = live_btts_prob(
        prior_goals_h, prior_goals_a, elapsed,
        state.goals_home, state.goals_away, state,
    )

    # --- Moneyline ---
    ml = live_moneyline_probs(
        prior_goals_h, prior_goals_a, elapsed,
        state.goals_home, state.goals_away, state,
    )

    # --- Momentum ---
    mom = momentum_score(state)

    # --- xG ---
    xg_home, xg_away = live_xg_estimate(state)
    xg_ml = xg_adjusted_moneyline(state, ml) if elapsed > 0 else None

    # --- Score-state transitions ---
    next_goal = score_state_probabilities(state)

    # --- Poisson fit quality ---
    fit_goals   = poisson_fit_quality(prior_goals, elapsed, state.total_goals)
    fit_corners = poisson_fit_quality(prior_corners, elapsed, state.total_corners)
    fit_cards   = poisson_fit_quality(prior_cards, elapsed, state.total_cards)
    fit_sot     = poisson_fit_quality(prior_sot, elapsed, state.total_sot)

    # --- Recent events (for display) ---
    recent = state.events_in_window(MOMENTUM_WINDOW_MIN)
    recent_dicts = [
        {
            "minute": e.minute,
            "type": e.event_type,
            "team": e.team_side,
            "player": e.player,
            "detail": e.detail,
        }
        for e in recent
    ]

    return LiveSnapshot(
        fixture_id=state.fixture_id,
        home_team=state.home_team,
        away_team=state.away_team,
        league=state.league,
        elapsed_min=elapsed,
        status=state.status,
        score=f"{state.goals_home} - {state.goals_away}",
        # Remaining
        remaining_goals=round(rem_goals, 3),
        remaining_corners=round(rem_corners, 3),
        remaining_cards=round(rem_cards, 3),
        remaining_sot=round(rem_sot, 3),
        # Totals
        projected_total_goals=round(proj_goals, 2),
        projected_total_corners=round(proj_corners, 2),
        projected_total_cards=round(proj_cards, 2),
        projected_total_sot=round(proj_sot, 2),
        # Over probabilities
        prob_over_goals=prob_over_goals,
        prob_over_corners=prob_over_corners,
        prob_over_cards=prob_over_cards,
        prob_over_sot=prob_over_sot,
        # BTTS and moneyline
        btts_prob=round(btts, 4),
        moneyline=ml,
        # Pre-match comparison
        pre_match_total_goals=state.prior_goals,
        pre_match_total_corners=state.prior_corners,
        pre_match_total_cards=state.prior_cards,
        pre_match_btts_prob=state.prior_btts_prob,
        pre_match_moneyline=state.prior_moneyline,
        # Momentum
        momentum=mom,
        # xG
        live_xg_home=xg_home,
        live_xg_away=xg_away,
        xg_adjusted_moneyline=xg_ml,
        # Score transitions
        next_goal_probs=next_goal,
        # Fit quality
        poisson_fit={
            "goals": fit_goals,
            "corners": fit_corners,
            "cards": fit_cards,
            "sot": fit_sot,
        },
        # Alerts (populated by live_alerts.py)
        alerts=[],
        recent_events=recent_dicts,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
