"""
core/projections.py — All projection functions extracted from rag_cli_v2.py.

Pure computation functions that take team metadata / profile stats and return
projected numbers (goals, corners, cards, SoT, BTTS, moneyline, correct score,
goal difference).  These do NOT modify any external state.

Dependencies:
    - core.weights          (SCORING_WEIGHTS)
    - core.team_resolution  (_profile_meta, _recent_stats, get_blended_variance,
                              get_recent_team_fixture_rows, canonical_team_name,
                              _numeric)
    - prob_models           (over_prob, poisson_pmf, negbin_pmf, negbin_from_mean_var)
    - ml_edge               (ml_predict_total, get_elo_edge, get_dynamic_blend_weight)
    - referee_data          (get_referee_modifier, RefereeModifier)
    - knockout_context      (KnockoutContext)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Internal imports — try core.* first, fall back to flat for backward compat
# ---------------------------------------------------------------------------
try:
    from core.weights import SCORING_WEIGHTS
except ImportError:
    try:
        from weights import SCORING_WEIGHTS
    except ImportError:
        from rag_cli_v2 import SCORING_WEIGHTS  # type: ignore[import]

try:
    from core.team_resolution import (
        _profile_meta,
        _recent_stats,
        get_blended_variance,
        get_recent_team_fixture_rows,
        canonical_team_name,
        _numeric,
    )
except ImportError:
    try:
        from team_resolution import (
            _profile_meta,
            _recent_stats,
            get_blended_variance,
            get_recent_team_fixture_rows,
            canonical_team_name,
            _numeric,
        )
    except ImportError:
        from rag_cli_v2 import (  # type: ignore[import]
            _profile_meta,
            _recent_stats,
            get_blended_variance,
            get_recent_team_fixture_rows,
            canonical_team_name,
            _numeric,
        )

# prob_models
try:
    from prob_models import (
        over_prob, poisson_pmf, negbin_pmf, negbin_from_mean_var,
        dixon_coles_scoreline_prob, dixon_coles_btts_prob, dixon_coles_scoreline_matrix,
    )
except ImportError:
    from Scripts.rag_ingest.prob_models import (  # type: ignore[import]
        over_prob, poisson_pmf, negbin_pmf, negbin_from_mean_var,
        dixon_coles_scoreline_prob, dixon_coles_btts_prob, dixon_coles_scoreline_matrix,
    )

# ml_edge
try:
    from ml_edge import ml_predict_total, get_elo_edge, get_dynamic_blend_weight
    _HAS_ML = True
except ImportError:
    try:
        from Scripts.rag_ingest.ml_edge import ml_predict_total, get_elo_edge, get_dynamic_blend_weight
        _HAS_ML = True
    except ImportError:
        _HAS_ML = False
        def ml_predict_total(*a, **kw): return None
        def get_elo_edge(*a, **kw): return None
        def get_dynamic_blend_weight(*a, **kw): return 0.0

# referee_data
try:
    from referee_data import get_referee_modifier, RefereeModifier
    _REFEREE_AVAILABLE = True
except ImportError:
    try:
        from Scripts.rag_ingest.referee_data import get_referee_modifier, RefereeModifier
        _REFEREE_AVAILABLE = True
    except ImportError:
        _REFEREE_AVAILABLE = False
        from dataclasses import dataclass as _dc

        @_dc
        class RefereeModifier:
            multiplier: float = 1.0
            referee_name: str = None
            sample_size: int = 0
            confidence: float = 0.0
            strictness_ratio: float = 1.0
            avg_cards_per_match: float = 0.0
            avg_fouls_per_match: float = 0.0
            cards_per_foul: float = 0.0
            source: str = "unavailable"

        def get_referee_modifier(*args, **kwargs) -> "RefereeModifier":
            return RefereeModifier()

# knockout_context
try:
    from knockout_context import KnockoutContext
except ImportError:
    try:
        from Scripts.rag_ingest.knockout_context import KnockoutContext
    except ImportError:
        from dataclasses import dataclass as _dc_ko

        @_dc_ko
        class KnockoutContext:
            is_knockout: bool = False
            goals_modifier: float = 1.0
            corners_modifier: float = 1.0
            cards_modifier: float = 1.0
            sot_modifier: float = 1.0
            source: str = "none"

# league_context
try:
    from league_context import get_league_context
    _HAS_LEAGUE_CONTEXT = True
except ImportError:
    try:
        from Scripts.rag_ingest.league_context import get_league_context
        _HAS_LEAGUE_CONTEXT = True
    except ImportError:
        _HAS_LEAGUE_CONTEXT = False

        def get_league_context(*args, **kwargs):
            return None

# lineup_context card-risk fallback
try:
    from lineup_context import get_card_risk_profiles
    _HAS_LINEUP_RISK = True
except ImportError:
    try:
        from Scripts.rag_ingest.lineup_context import get_card_risk_profiles
        _HAS_LINEUP_RISK = True
    except ImportError:
        _HAS_LINEUP_RISK = False

        def get_card_risk_profiles(*args, **kwargs):
            return None, None


# ---- Module-level cache (mirrors the one in rag_cli_v2.py) ----
_team_recent_var_cache: Dict[tuple, Dict] = {}


# ===================================================================
#  Small helpers
# ===================================================================

def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _weighted_avg(values: List[Optional[float]], alpha: float = 0.85) -> Optional[float]:
    """Exponentially-weighted average.  Index 0 = most recent (highest weight)."""
    cleaned = [(i, x) for i, x in enumerate(values) if x is not None]
    if not cleaned:
        return None
    weights = [alpha ** i for i, _ in cleaned]
    total_w = sum(weights)
    return sum(w * x for w, (_, x) in zip(weights, cleaned)) / total_w


def _linear_slope(values: List[Optional[float]]) -> Optional[float]:
    """Compute linear slope of values (index 0 = most recent).
    Positive slope = improving (recent values higher than older values).
    Returns slope per match, or None if fewer than 4 valid points."""
    cleaned = [(i, x) for i, x in enumerate(values) if x is not None]
    if len(cleaned) < 4:
        return None
    n = len(cleaned)
    # Reverse so older = lower index (conventional regression direction)
    xs = [float(n - 1 - i) for i, _ in cleaned]
    ys = [x for _, x in cleaned]
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


def _stat_blend(stat: str) -> Tuple[float, float]:
    """Return (season_weight, recent_weight) for a given stat type.

    Uses stat-specific overrides from SCORING_WEIGHTS["projection"] if available,
    otherwise falls back to the generic blend_season/blend_recent.
    """
    pw = SCORING_WEIGHTS["projection"]
    s_key = f"blend_{stat}_season"
    r_key = f"blend_{stat}_recent"
    if s_key in pw and r_key in pw:
        return pw[s_key], pw[r_key]
    return pw["blend_season"], pw["blend_recent"]


def _venue_blend(venue_rate: Optional[float], overall_rate: Optional[float], blend_w: float) -> Optional[float]:
    """Blend venue-specific and overall rate.  Falls back gracefully."""
    if venue_rate is not None and overall_rate is not None:
        return blend_w * venue_rate + (1.0 - blend_w) * overall_rate
    if venue_rate is not None:
        return venue_rate
    return overall_rate


def _ml_blend_weight(stat: str) -> float:
    """Resolve ML blend weight for a stat. Uses dynamic R2-based weight if enabled,
    otherwise falls back to static SCORING_WEIGHTS["ml"]["blend_weight"].
    Stats in ml_skip_stats are forced to 0.0 regardless of dynamic_blend."""
    mw = SCORING_WEIGHTS["ml"]
    if stat in mw.get("skip_stats", set()):
        return 0.0
    if mw.get("dynamic_blend", False):
        return get_dynamic_blend_weight(stat)
    return mw.get("blend_weight", 0.0)


def _trend_adjustment(hr: Dict, ar: Dict, slope_key: str) -> float:
    """Compute trend adjustment from home/away recent slopes.
    Positive slope = team improving -> nudge projection up."""
    tw = SCORING_WEIGHTS.get("recency", {}).get("trend_weight", 0.0)
    if tw == 0:
        return 0.0
    h_slope = hr.get(slope_key)
    a_slope = ar.get(slope_key)
    adj = 0.0
    if h_slope is not None:
        adj += tw * h_slope
    if a_slope is not None:
        adj += tw * a_slope
    return adj


def _blend_total_with_divergence(stat: str, season_total: Optional[float], recent_total: Optional[float]) -> Optional[float]:
    """Match rag_cli_v2 total-stat blending: shift toward recent when form meaningfully diverges."""
    if season_total is not None and recent_total is not None:
        base_season_w, base_recent_w = _stat_blend(stat)
        divergence = abs(recent_total - season_total) / max(season_total, 1.0)
        if divergence > 0.20:
            shift = min(0.30, (divergence - 0.20) * 1.0)
            blend_season = base_season_w - shift
            blend_recent = base_recent_w + shift
        else:
            blend_season = base_season_w
            blend_recent = base_recent_w
        return blend_season * season_total + blend_recent * recent_total
    if season_total is not None:
        return season_total
    return recent_total


def _weighted_sum(components: List[Tuple[float, Optional[float]]]) -> Optional[float]:
    total = 0.0
    seen = False
    for weight, value in components:
        if value is None or weight == 0.0:
            continue
        total += weight * value
        seen = True
    return total if seen else None


def _weighted_component_blend(components: List[Tuple[Optional[float], float]]) -> Optional[float]:
    total_weight = 0.0
    blended = 0.0
    for value, weight in components:
        if value is None or weight == 0.0:
            continue
        blended += weight * value
        total_weight += weight
    if total_weight <= 0:
        return None
    return blended / total_weight


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _resolve_league_context(
    home: str,
    away: str,
    league: str,
    league_ctx=None,
    fixture_date: Optional[str] = None,
):
    if league_ctx is not None:
        return league_ctx
    if not _HAS_LEAGUE_CONTEXT:
        return None
    try:
        return get_league_context(home, away, league, target_date=fixture_date)
    except Exception:
        return None


def _profile_as_of(team: str, league: str, fixture_date: Optional[str] = None) -> Dict:
    """Keep legacy callers two-argument compatible while dated calls are safe."""
    if fixture_date:
        return _profile_meta(team, league, target_date=fixture_date)
    return _profile_meta(team, league)


def _extract_card_risk_context(home_risk, away_risk) -> Optional[Dict]:
    if home_risk is None or away_risk is None:
        return None
    n_home = int(getattr(home_risk, "n_players", 0) or 0)
    n_away = int(getattr(away_risk, "n_players", 0) or 0)
    if n_home <= 0 and n_away <= 0:
        return None
    return {
        "home_cards_per_90": float(getattr(home_risk, "team_starter_cards_per_90", 0.0) or 0.0),
        "away_cards_per_90": float(getattr(away_risk, "team_starter_cards_per_90", 0.0) or 0.0),
        "home_high_risk": int(getattr(home_risk, "high_card_risk_players", 0) or 0),
        "away_high_risk": int(getattr(away_risk, "high_card_risk_players", 0) or 0),
        "combined_risk_90": float(
            (getattr(home_risk, "team_starter_cards_per_90", 0.0) or 0.0)
            + (getattr(away_risk, "team_starter_cards_per_90", 0.0) or 0.0)
        ),
        "source": "expected_xi",
    }


def compute_stat_confidence(
    stat: str,
    home: str,
    away: str,
    league: str,
    ref_mod=None,
    fixture_date: Optional[str] = None,
) -> str:
    """Compute confidence bucket for any stat projection.

    Returns "high", "medium", or "low" based on:
    - Sample size (recent fixture count)
    - Referee data availability (for cards)
    - Stat-specific thresholds

    This extends the cards-specific confidence gating to all stats.
    """
    hr = _recent_stats(home, league, last_n=6, target_date=fixture_date)
    ar = _recent_stats(away, league, last_n=6, target_date=fixture_date)
    n_home = hr.get("n", 0)
    n_away = ar.get("n", 0)

    if stat == "cards":
        # Cards use the detail helper's confidence bucket (richer logic)
        ref_conf = 0.0
        if ref_mod is not None:
            ref_conf = getattr(ref_mod, "confidence", 0.0) if getattr(ref_mod, "source", "") == "profile" else 0.0
        ccf = SCORING_WEIGHTS.get("cards_confidence", {})
        if ref_conf >= ccf.get("referee_high_confidence", 0.70) and n_home >= 5 and n_away >= 5:
            return "high"
        elif ref_conf >= ccf.get("referee_medium_confidence", 0.35) or (n_home >= 5 and n_away >= 5):
            return "medium"
        return "low"

    # General stat confidence: based on sample size and profile data availability
    hm = _profile_as_of(home, league, fixture_date)
    am = _profile_as_of(away, league, fixture_date)
    has_profiles = bool(hm) and bool(am)

    if has_profiles and n_home >= 8 and n_away >= 8:
        return "high"
    elif has_profiles and n_home >= 4 and n_away >= 4:
        return "medium"
    return "low"


def _share_from_scores(
    team_score: Optional[float],
    opp_score: Optional[float],
    *,
    shrink: float,
    floor: float,
    ceiling: float,
) -> Optional[float]:
    if team_score is None and opp_score is None:
        return None
    team_base = max(team_score if team_score is not None else 0.0, 0.05)
    opp_base = max(opp_score if opp_score is not None else 0.0, 0.05)
    raw_share = team_base / (team_base + opp_base)
    shrunk = 0.5 + (raw_share - 0.5) * (1.0 - shrink)
    return _clamp(shrunk, floor, ceiling)


# ===================================================================
#  Per-team projection primitives
# ===================================================================

def projected_corners(home_meta: Dict, away_meta: Dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Blended projection with venue-specific rates:
    - own corners tendency (home-specific or away-specific when available)
    - opponent corners_against tendency
    """
    pw = SCORING_WEIGHTS["projection"]
    vb = pw.get("corners_venue_blend", 0.0)

    # Home team: prefer corners_home_pm, blended with overall corners_pm
    h_for_overall = safe_float(home_meta.get("corners_pm"))
    h_for_venue   = safe_float(home_meta.get("corners_home_pm"))
    h_for = _venue_blend(h_for_venue, h_for_overall, vb)

    # Away team: prefer corners_away_pm, blended with overall corners_pm
    a_for_overall = safe_float(away_meta.get("corners_pm"))
    a_for_venue   = safe_float(away_meta.get("corners_away_pm"))
    a_for = _venue_blend(a_for_venue, a_for_overall, vb)

    h_opp_allow = safe_float(away_meta.get("corners_against_pm"))
    a_opp_allow = safe_float(home_meta.get("corners_against_pm"))

    h_proj = None
    a_proj = None
    if h_for is not None and h_opp_allow is not None:
        h_proj = pw["corners_own"] * h_for + pw["corners_opp"] * h_opp_allow
    elif h_for is not None:
        h_proj = h_for
    elif h_opp_allow is not None:
        h_proj = h_opp_allow

    if a_for is not None and a_opp_allow is not None:
        a_proj = pw["corners_own"] * a_for + pw["corners_opp"] * a_opp_allow
    elif a_for is not None:
        a_proj = a_for
    elif a_opp_allow is not None:
        a_proj = a_opp_allow

    return h_proj, a_proj


def projected_cards(
    home_meta: Dict, away_meta: Dict,
    referee_modifier: float = 1.0,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Blended card projection with venue-specific rates and opponent induction:
    - own cards tendency (home-specific or away-specific when available)
    - opponent card-inducing tendency (opp_cards_induced_pm)
    - small aggression nudge
    - referee strictness modifier (multiplicative)
    """
    pw = SCORING_WEIGHTS["projection_cards"]
    vb = pw.get("venue_blend", 0.0)

    # Home team: prefer cards_home_pm, blended with overall cards_per_90_team
    h_for_overall = safe_float(home_meta.get("cards_per_90_team"))
    h_for_venue   = safe_float(home_meta.get("cards_home_pm"))
    h_for = _venue_blend(h_for_venue, h_for_overall, vb)

    # Away team: prefer cards_away_pm, blended with overall
    a_for_overall = safe_float(away_meta.get("cards_per_90_team"))
    a_for_venue   = safe_float(away_meta.get("cards_away_pm"))
    a_for = _venue_blend(a_for_venue, a_for_overall, vb)

    # Opponent card-inducing rates (cross-referenced)
    h_opp_allow = safe_float(away_meta.get("opp_cards_induced_pm"))
    a_opp_allow = safe_float(home_meta.get("opp_cards_induced_pm"))

    # Aggression nudge
    h_agg = safe_float(home_meta.get("aggression_index_norm"))
    a_agg = safe_float(away_meta.get("aggression_index_norm"))

    def blend(own, opp, agg):
        base = None
        if own is not None and opp is not None:
            base = pw["own"] * own + pw["opp"] * opp
        elif own is not None:
            base = own
        elif opp is not None:
            base = opp
        if base is not None and agg is not None:
            base += pw["agg_nudge"] * (agg * pw["agg_scale"])
        return base

    h_result = blend(h_for, h_opp_allow, h_agg)
    a_result = blend(a_for, a_opp_allow, a_agg)
    if referee_modifier != 1.0:
        if h_result is not None:
            h_result *= referee_modifier
        if a_result is not None:
            a_result *= referee_modifier
    return h_result, a_result


def projected_sot(home_meta: Dict, away_meta: Dict) -> Tuple[Optional[float], Optional[float]]:
    """Blended SoT projection: own sot_for_pm vs opponent sot_against_pm, with venue splits."""
    pw = SCORING_WEIGHTS["projection_sot"]
    proj_w = SCORING_WEIGHTS["projection"]
    vb = proj_w.get("sot_venue_blend", 0.50)

    h_for = safe_float(home_meta.get("sot_for_pm"))
    h_for_venue = safe_float(home_meta.get("sot_home_pm"))
    h_for = _venue_blend(h_for_venue, h_for, vb)

    a_for = safe_float(away_meta.get("sot_for_pm"))
    a_for_venue = safe_float(away_meta.get("sot_away_pm"))
    a_for = _venue_blend(a_for_venue, a_for, vb)

    h_opp_allow = safe_float(away_meta.get("sot_against_pm"))
    a_opp_allow = safe_float(home_meta.get("sot_against_pm"))

    def blend(own, opp):
        if own is not None and opp is not None:
            return pw["own"] * own + pw["opp"] * opp
        return own if own is not None else opp

    return blend(h_for, h_opp_allow), blend(a_for, a_opp_allow)


def projected_goals(home_meta: Dict, away_meta: Dict) -> Tuple[Optional[float], Optional[float]]:
    """Per-team goal projection blending own attack with opponent defense.
    Uses xG when available (60% xG + 40% actual goals) and venue splits."""
    pw = SCORING_WEIGHTS["projection"]
    own_w = pw.get("goals_own", 0.6)
    opp_w = pw.get("goals_opp", 0.4)
    vb = pw.get("goals_venue_blend", 0.50)
    xg_w = pw.get("xg_blend", 0.60)

    # Raw goals and xG per match
    h_gf_raw = safe_float(home_meta.get("goals_for_pm"))
    a_gf_raw = safe_float(away_meta.get("goals_for_pm"))
    # xG: compute per-match average from venue splits, or fall back to overall
    h_xg_home = safe_float(home_meta.get("xg_home_pm"))
    h_xg_away = safe_float(home_meta.get("xg_away_pm"))
    a_xg_home = safe_float(away_meta.get("xg_home_pm"))
    a_xg_away = safe_float(away_meta.get("xg_away_pm"))
    # Overall xG per match: average of home and away xG
    h_xg = ((h_xg_home + h_xg_away) / 2.0) if (h_xg_home is not None and h_xg_away is not None) else None
    a_xg = ((a_xg_home + a_xg_away) / 2.0) if (a_xg_home is not None and a_xg_away is not None) else None

    # Blend xG with actual goals when both available
    def _xg_blend(actual, xg):
        if actual is not None and xg is not None:
            return xg_w * xg + (1.0 - xg_w) * actual
        return actual  # fallback to actual if no xG

    h_gf = _xg_blend(h_gf_raw, h_xg)
    a_gf = _xg_blend(a_gf_raw, a_xg)

    # Venue-specific rates
    h_gf_venue = safe_float(home_meta.get("goals_home_pm"))
    a_gf_venue = safe_float(away_meta.get("goals_away_pm"))
    h_gf = _venue_blend(h_gf_venue, h_gf, vb)
    a_gf = _venue_blend(a_gf_venue, a_gf, vb)

    # goals_against_pm = avg goals conceded by that team
    a_ga = safe_float(away_meta.get("goals_against_pm"))
    h_ga = safe_float(home_meta.get("goals_against_pm"))

    h_proj = None
    if h_gf is not None and a_ga is not None:
        h_proj = own_w * h_gf + opp_w * a_ga
    elif h_gf is not None:
        h_proj = h_gf
    elif a_ga is not None:
        h_proj = a_ga

    a_proj = None
    if a_gf is not None and h_ga is not None:
        a_proj = own_w * a_gf + opp_w * h_ga
    elif a_gf is not None:
        a_proj = a_gf
    elif h_ga is not None:
        a_proj = h_ga

    # Home advantage adjustment: boost home, reduce away
    ha = pw.get("home_advantage", 0.0)
    if ha > 0:
        if h_proj is not None:
            h_proj += ha
        if a_proj is not None:
            a_proj = max(0.1, a_proj - ha)

    return h_proj, a_proj


def projected_team_goals(
    team: str,
    opponent: str,
    league: str,
    is_home: bool,
    league_ctx=None,
    fixture_date: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Per-team goal projection blending season + recent form.

    Returns (blended, season, recent).
    """
    pw = SCORING_WEIGHTS["projection"]
    home_name = team if is_home else opponent
    away_name = opponent if is_home else team
    league_ctx = _resolve_league_context(home_name, away_name, league,
                                          league_ctx=league_ctx, fixture_date=fixture_date)
    hm = _profile_as_of(home_name, league, fixture_date)
    am = _profile_as_of(away_name, league, fixture_date)

    h_proj, a_proj = projected_goals(hm, am)
    season_proj = h_proj if is_home else a_proj

    recent_stats = _recent_stats(team, league, last_n=6, target_date=fixture_date)
    recent_goals = recent_stats.get("goals_for_avg")

    if season_proj is not None and recent_goals is not None:
        blended = pw.get("blend_goals_season", 0.85) * season_proj + pw.get("blend_goals_recent", 0.15) * recent_goals
    elif season_proj is not None:
        blended = season_proj
    elif recent_goals is not None:
        blended = recent_goals
    else:
        return None, None, None

    if league_ctx is not None:
        adj = getattr(league_ctx, "adjustments", {}).get("goals", 1.0)
        blended *= adj

    return blended, season_proj, recent_goals


# ===================================================================
#  Match-total projections (blended season + recent + ML)
# ===================================================================

def projected_total_sot(
    home: str,
    away: str,
    league: str,
    knockout_ctx: Optional[KnockoutContext] = None,
    league_ctx=None,
    fixture_date: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    league_ctx = _resolve_league_context(home, away, league, league_ctx=league_ctx, fixture_date=fixture_date)
    hm = _profile_as_of(home, league, fixture_date)
    am = _profile_as_of(away, league, fixture_date)
    h_proj, a_proj = projected_sot(hm, am)
    season_total = (h_proj + a_proj) if (h_proj is not None and a_proj is not None) else None

    hr = _recent_stats(home, league, last_n=6, target_date=fixture_date)
    ar = _recent_stats(away, league, last_n=6, target_date=fixture_date)
    h_recent_for = hr.get("sot_for_avg")
    a_recent_for = ar.get("sot_for_avg")
    # Blend each team's recent SoT attack with opponent's recent SoT defensive concession
    if h_recent_for is not None and a_recent_for is not None:
        h_recent_proj = 0.6 * h_recent_for + 0.4 * ar.get("sot_against_avg", a_proj if a_proj is not None else a_recent_for)
        a_recent_proj = 0.6 * a_recent_for + 0.4 * hr.get("sot_against_avg", h_proj if h_proj is not None else h_recent_for)
        recent_total = h_recent_proj + a_recent_proj
    else:
        recent_total = None

    blended = _blend_total_with_divergence("sot", season_total, recent_total)
    if blended is None:
        return None, None, None

    # Trend adjustment
    blended += _trend_adjustment(hr, ar, "sot_for_slope")

    # ML blend
    ml_w = _ml_blend_weight("sot")
    if ml_w > 0:
        ml_proj = ml_predict_total(home, away, league, "sot", home_meta=hm, away_meta=am)
        if ml_proj is not None:
            blended = (1.0 - ml_w) * blended + ml_w * ml_proj

    # Knockout round context adjustment
    if knockout_ctx and knockout_ctx.is_knockout and knockout_ctx.sot_modifier != 1.0:
        blended *= knockout_ctx.sot_modifier

    if league_ctx is not None:
        adj = getattr(league_ctx, "adjustments", {}).get("sot", 1.0)
        if adj != 1.0:
            blended *= adj

    return blended, season_total, recent_total


def projected_total_corners(
    home: str,
    away: str,
    league: str,
    knockout_ctx: Optional[KnockoutContext] = None,
    league_ctx=None,
    fixture_date: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    league_ctx = _resolve_league_context(home, away, league, league_ctx=league_ctx, fixture_date=fixture_date)
    hm = _profile_as_of(home, league, fixture_date)
    am = _profile_as_of(away, league, fixture_date)
    h_proj, a_proj = projected_corners(hm, am)
    season_total = (h_proj + a_proj) if (h_proj is not None and a_proj is not None) else None

    hr = _recent_stats(home, league, last_n=6, target_date=fixture_date)
    ar = _recent_stats(away, league, last_n=6, target_date=fixture_date)
    h_recent_for = hr.get("corners_for_avg")
    a_recent_for = ar.get("corners_for_avg")
    # Blend each team's recent attack with opponent's recent defensive concession
    if h_recent_for is not None and a_recent_for is not None:
        h_recent_proj = 0.6 * h_recent_for + 0.4 * ar.get("corners_against_avg", a_proj if a_proj is not None else a_recent_for)
        a_recent_proj = 0.6 * a_recent_for + 0.4 * hr.get("corners_against_avg", h_proj if h_proj is not None else h_recent_for)
        recent_total = h_recent_proj + a_recent_proj
    else:
        recent_total = None

    blended = _blend_total_with_divergence("corners", season_total, recent_total)
    if blended is None:
        return None, None, None

    # Trend adjustment
    blended += _trend_adjustment(hr, ar, "corners_for_slope")

    # ML blend
    ml_w = _ml_blend_weight("corners")
    if ml_w > 0:
        ml_proj = ml_predict_total(home, away, league, "corners", home_meta=hm, away_meta=am)
        if ml_proj is not None:
            blended = (1.0 - ml_w) * blended + ml_w * ml_proj

    # Knockout round context adjustment
    if knockout_ctx and knockout_ctx.is_knockout and knockout_ctx.corners_modifier != 1.0:
        blended *= knockout_ctx.corners_modifier

    # League context adjustment
    if league_ctx is not None:
        adj = getattr(league_ctx, "adjustments", {}).get("corners", 1.0)
        if adj != 1.0:
            blended *= adj

    return blended, season_total, recent_total


def projected_total_cards_detail(
    home: str,
    away: str,
    league: str,
    knockout_ctx: Optional[KnockoutContext] = None,
    ref_mod: Optional[RefereeModifier] = None,
    lineup_ctx=None,
    league_ctx=None,
    fixture_date: Optional[str] = None,
) -> Dict:
    """Rich cards projection returning a detail dict.

    Returns:
        {
            "total_cards": float | None,
            "total_yellows": float | None,
            "total_reds": float | None,
            "season_total": float | None,
            "recent_total": float | None,
            "referee_modifier": RefereeModifier,
            "lineup_card_context": dict | None,
            "uncertainty": float,
            "confidence_bucket": str,     # "high", "medium", "low"
            "warnings": list[str],
        }
    """
    pw = SCORING_WEIGHTS["projection"]
    rw = SCORING_WEIGHTS["referee"]
    ctm = SCORING_WEIGHTS.get("cards_total_model", {})
    cyr = SCORING_WEIGHTS.get("cards_yellow_red_split", {})
    ccf = SCORING_WEIGHTS.get("cards_confidence", {})
    league_ctx = _resolve_league_context(home, away, league, league_ctx=league_ctx, fixture_date=fixture_date)
    hm = _profile_as_of(home, league, fixture_date)
    am = _profile_as_of(away, league, fixture_date)

    warnings: list = []

    if ref_mod is None:
        ref_mod = get_referee_modifier(home, away, league,
                                       weight=rw.get("modifier_weight", 0.25))

    # ---- Yellow-card base model ----
    h_proj, a_proj = projected_cards(hm, am, referee_modifier=ref_mod.multiplier)
    season_total = (h_proj + a_proj) if (h_proj is not None and a_proj is not None) else None

    hr = _recent_stats(home, league, last_n=6, target_date=fixture_date)
    ar = _recent_stats(away, league, last_n=6, target_date=fixture_date)

    # Yellow/red split from season profiles
    h_yellows_pm = safe_float((hm or {}).get("yellows_pm"))
    a_yellows_pm = safe_float((am or {}).get("yellows_pm"))
    h_reds_pm = safe_float((hm or {}).get("reds_pm"))
    a_reds_pm = safe_float((am or {}).get("reds_pm"))

    # Season yellow total
    season_yellows = None
    season_reds = None
    if h_yellows_pm is not None and a_yellows_pm is not None:
        season_yellows = h_yellows_pm + a_yellows_pm
    if h_reds_pm is not None and a_reds_pm is not None:
        season_reds = h_reds_pm + a_reds_pm

    # Recent yellow/red averages
    h_recent_yellows = hr.get("yellows_avg")
    a_recent_yellows = ar.get("yellows_avg")
    h_recent_reds = hr.get("reds_avg")
    a_recent_reds = ar.get("reds_avg")
    recent_yellows = None
    recent_reds = None
    if h_recent_yellows is not None and a_recent_yellows is not None:
        recent_yellows = h_recent_yellows + a_recent_yellows
    if h_recent_reds is not None and a_recent_reds is not None:
        recent_reds = h_recent_reds + a_recent_reds

    # Recent cards total (existing logic)
    h_recent_cards = hr.get("cards_avg")
    a_recent_cards = ar.get("cards_avg")
    h_recent_opp_induced = ar.get("cards_induced_avg")
    a_recent_opp_induced = hr.get("cards_induced_avg")
    if h_recent_cards is not None and a_recent_cards is not None:
        h_recent_proj = 0.6 * h_recent_cards + 0.4 * (
            h_recent_opp_induced if h_recent_opp_induced is not None else a_recent_cards
        )
        a_recent_proj = 0.6 * a_recent_cards + 0.4 * (
            a_recent_opp_induced if a_recent_opp_induced is not None else h_recent_cards
        )
        recent_total = h_recent_proj + a_recent_proj
    else:
        recent_total = None

    blended = _blend_total_with_divergence("cards", season_total, recent_total)
    if blended is None:
        return {
            "total_cards": None, "total_yellows": None, "total_reds": None,
            "season_total": None, "recent_total": None,
            "referee_modifier": ref_mod, "lineup_card_context": None,
            "uncertainty": 999.0, "confidence_bucket": "low", "warnings": ["insufficient data"],
        }

    # Trend adjustment
    blended += _trend_adjustment(hr, ar, "cards_slope")

    # ---- Red-card adjustment ----
    # Estimate reds separately, then compose total = yellows_base + reds_add
    reds_base = ctm.get("reds_base_rate", 0.18)
    reds_cap = ctm.get("reds_cap", 0.60)
    if season_reds is not None and recent_reds is not None:
        reds_est = 0.70 * season_reds + 0.30 * recent_reds
    elif season_reds is not None:
        reds_est = season_reds
    elif recent_reds is not None:
        reds_est = recent_reds
    else:
        reds_est = reds_base
    reds_est = min(reds_est, reds_cap)

    # Yellow estimate: total cards minus reds (but floor at 0)
    yellows_est = max(0.0, blended - reds_est)

    # Rebuild total: yellows_base_weight * yellows + reds_add_weight * reds
    yellows_w = ctm.get("yellows_base_weight", 0.82)
    reds_w = ctm.get("reds_add_weight", 0.18)
    # The blended total is already the primary anchor; the yellow/red split
    # provides transparency without changing the total significantly
    total_yellows = yellows_est
    total_reds = reds_est

    # ---- Referee component (existing logic, unchanged) ----
    if (ref_mod.source == "profile" and ref_mod.avg_cards_per_match > 0
            and ref_mod.sample_size >= rw.get("min_sample_size", 5)):
        ref_anchor_w = rw.get("anchor_weight", 0.15) * ref_mod.confidence
        blended = (1.0 - ref_anchor_w) * blended + ref_anchor_w * ref_mod.avg_cards_per_match

    # Foul-based card estimate
    cpf = ref_mod.cards_per_foul if ref_mod.source == "profile" else 0.0
    if cpf > 0 and ref_mod.sample_size >= rw.get("min_sample_size", 5):
        pc = SCORING_WEIGHTS["projection_cards"]
        h_fouls_s = safe_float(hm.get("fouls_per_90_team"))
        a_fouls_s = safe_float(am.get("fouls_per_90_team"))
        h_fouls_r = hr.get("fouls_avg")
        a_fouls_r = ar.get("fouls_avg")
        def _foul_bl(s, r):
            if s is not None and r is not None:
                return pw["blend_season"] * s + pw["blend_recent"] * r
            return s if s is not None else r
        h_fouls = _foul_bl(h_fouls_s, h_fouls_r)
        a_fouls = _foul_bl(a_fouls_s, a_fouls_r)
        if h_fouls is not None and a_fouls is not None:
            foul_based_total = (h_fouls + a_fouls) * cpf
            foul_w = pc.get("foul_card_blend", 0.15) * ref_mod.confidence
            blended = (1.0 - foul_w) * blended + foul_w * foul_based_total

    # ---- Recent induced-card component ----
    induced_w = ctm.get("induced_weight", 0.12)
    if induced_w > 0 and h_recent_opp_induced is not None and a_recent_opp_induced is not None:
        induced_signal = h_recent_opp_induced + a_recent_opp_induced
        blended = (1.0 - induced_w) * blended + induced_w * induced_signal

    # ---- Lineup card-risk component ----
    lineup_card_context = None
    lineup_risk_w = ctm.get("lineup_risk_weight", 0.08)
    lineup_risk_cap = ctm.get("lineup_risk_cap", 0.25)
    h_risk = getattr(lineup_ctx, "home_card_risk", None) if lineup_ctx is not None else None
    a_risk = getattr(lineup_ctx, "away_card_risk", None) if lineup_ctx is not None else None
    if (h_risk is None or a_risk is None) and _HAS_LINEUP_RISK:
        try:
            h_risk, a_risk = get_card_risk_profiles(home, away, league)
        except Exception:
            h_risk, a_risk = None, None
    lineup_card_context = _extract_card_risk_context(h_risk, a_risk)
    if lineup_card_context is not None:
        if getattr(lineup_ctx, "source", "") == "lineups":
            lineup_card_context["source"] = "confirmed_xi"
        combined_risk_90 = lineup_card_context["combined_risk_90"]
        if combined_risk_90 > 0:
            risk_delta = combined_risk_90 - blended
            adj = max(-lineup_risk_cap, min(lineup_risk_cap, lineup_risk_w * risk_delta))
            blended += adj

    # ---- ML blend ----
    ml_w = _ml_blend_weight("cards")
    if ml_w > 0:
        ml_proj = ml_predict_total(home, away, league, "cards", home_meta=hm, away_meta=am)
        if ml_proj is not None:
            blended = (1.0 - ml_w) * blended + ml_w * ml_proj

    # ---- Knockout adjustment ----
    if knockout_ctx and knockout_ctx.is_knockout and knockout_ctx.cards_modifier != 1.0:
        blended *= knockout_ctx.cards_modifier

    if league_ctx is not None:
        adj = getattr(league_ctx, "adjustments", {}).get("cards", 1.0)
        if adj != 1.0:
            blended *= adj

    # ---- Re-derive yellow/red from final blended total ----
    if season_yellows is not None and season_total is not None and season_total > 0:
        yellow_ratio = season_yellows / season_total
    else:
        yellow_ratio = cyr.get("default_yellow_ratio", 0.92)
    yellow_ratio = max(cyr.get("min_yellow_ratio", 0.80),
                       min(cyr.get("max_yellow_ratio", 0.98), yellow_ratio))
    total_yellows = blended * yellow_ratio
    total_reds = blended * (1.0 - yellow_ratio)

    # ---- Uncertainty & confidence ----
    uncertainty = ccf.get("uncertainty_base", 0.8)
    if ref_mod.source != "profile":
        uncertainty += ccf.get("uncertainty_no_referee", 0.3)
        warnings.append("No referee assigned — cards projection has higher uncertainty")
    if lineup_card_context is None:
        uncertainty += ccf.get("uncertainty_no_lineup", 0.1)
    n_home = hr.get("n", 0)
    n_away = ar.get("n", 0)
    if n_home < 10 or n_away < 10:
        uncertainty += ccf.get("uncertainty_low_sample", 0.2)
        warnings.append("Low sample size for one or both teams")

    ref_conf = ref_mod.confidence if ref_mod.source == "profile" else 0.0
    if ref_conf >= ccf.get("referee_high_confidence", 0.70) and n_home >= 5 and n_away >= 5:
        confidence_bucket = "high"
    elif ref_conf >= ccf.get("referee_medium_confidence", 0.35) or (n_home >= 5 and n_away >= 5):
        confidence_bucket = "medium"
    else:
        confidence_bucket = "low"

    return {
        "total_cards": round(blended, 3),
        "total_yellows": round(total_yellows, 3),
        "total_reds": round(total_reds, 3),
        "season_total": round(season_total, 3) if season_total is not None else None,
        "recent_total": round(recent_total, 3) if recent_total is not None else None,
        "referee_modifier": ref_mod,
        "lineup_card_context": lineup_card_context,
        "uncertainty": round(uncertainty, 3),
        "confidence_bucket": confidence_bucket,
        "warnings": warnings,
    }


def projected_total_cards(
    home: str,
    away: str,
    league: str,
    knockout_ctx: Optional[KnockoutContext] = None,
    ref_mod: Optional[RefereeModifier] = None,
    lineup_ctx=None,
    league_ctx=None,
    fixture_date: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float], RefereeModifier]:
    """Backward-compatible wrapper — returns (blended, season_total, recent_total, ref_mod)."""
    detail = projected_total_cards_detail(home, away, league,
                                          knockout_ctx=knockout_ctx, ref_mod=ref_mod,
                                          lineup_ctx=lineup_ctx, league_ctx=league_ctx,
                                          fixture_date=fixture_date)
    return (
        detail["total_cards"],
        detail["season_total"],
        detail["recent_total"],
        detail["referee_modifier"],
    )


# ---- Per-team blended projections (for team totals lines) ----

def projected_team_corners_detail(
    team: str,
    opponent: str,
    league: str,
    is_home: bool,
    league_ctx=None,
    fixture_date: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    """Per-team corner projection components.

    Architecture mirrors projected_team_cards:
      1. Get match total from projected_total_corners()
      2. Compute team share via season + recent feature scoring
      3. share_anchor = match_total * share
      4. stabilizer = direct per-team season/recent tendency
      5. final = anchor_weight * share_anchor + stabilizer_weight * stabilizer
    """
    tlw = SCORING_WEIGHTS.get("team_lines", {})
    csw = tlw.get("corners_share", {})
    cdw = tlw.get("corners_direct", {})
    cow = tlw.get("corners_output", {})
    home_name = team if is_home else opponent
    away_name = opponent if is_home else team
    league_ctx = _resolve_league_context(home_name, away_name, league,
                                          league_ctx=league_ctx, fixture_date=fixture_date)
    hm = _profile_as_of(team if is_home else opponent, league, fixture_date)
    am = _profile_as_of(opponent if is_home else team, league, fixture_date)
    team_meta = hm if is_home else am
    opp_meta = am if is_home else hm

    # ---- Step 1: Get match total (the anchor) ----
    total_blended, season_total, recent_total = projected_total_corners(
        home_name, away_name, league,
        league_ctx=league_ctx, fixture_date=fixture_date,
    )

    # ---- Step 2: Per-team base data for share + stabilizer ----
    h_proj, a_proj = projected_corners(hm, am)
    season_direct = h_proj if is_home else a_proj

    venue = "home" if is_home else "away"
    opp_venue = "away" if is_home else "home"
    recent_stats = _recent_stats(team, league, last_n=6, venue=venue, target_date=fixture_date)
    if recent_stats.get("corners_for_avg") is None:
        recent_stats = _recent_stats(team, league, last_n=6, target_date=fixture_date)
    opp_recent = _recent_stats(opponent, league, last_n=6, venue=opp_venue, target_date=fixture_date)
    if opp_recent.get("corners_against_avg") is None:
        opp_recent = _recent_stats(opponent, league, last_n=6, target_date=fixture_date)

    recent_team_for = recent_stats.get("corners_for_avg")
    recent_opp_allow = opp_recent.get("corners_against_avg")
    recent_own_w = tlw.get("corners_recent_own", 0.50)
    if recent_team_for is not None and recent_opp_allow is not None:
        recent_direct = recent_own_w * recent_team_for + (1.0 - recent_own_w) * recent_opp_allow
    else:
        recent_direct = recent_team_for if recent_team_for is not None else recent_opp_allow

    # ---- Step 3: Team share via feature scoring ----
    vb = SCORING_WEIGHTS["projection"].get("corners_venue_blend", 0.50)

    def _season_corner_share_score(meta: Dict, other_meta: Dict, home_side: bool) -> Optional[float]:
        return _weighted_sum([
            (csw.get("season_own", 0.35), _venue_blend(
                safe_float(meta.get("corners_home_pm" if home_side else "corners_away_pm")),
                safe_float(meta.get("corners_pm")),
                vb,
            )),
            (csw.get("season_opp_allow", 0.40), _venue_blend(
                safe_float(other_meta.get("corners_against_away_pm" if home_side else "corners_against_home_pm")),
                safe_float(other_meta.get("corners_against_pm")),
                vb,
            )),
            (csw.get("season_control", 0.10), safe_float(meta.get("control_index"))),
            (csw.get("season_dominance", 0.08), safe_float(meta.get("dominance_index"))),
            (csw.get("season_possession", 0.07), safe_float(meta.get("possession"))),
        ])

    def _recent_corner_share_score(team_recent: Dict, opp_rec: Dict) -> Optional[float]:
        return _weighted_sum([
            (csw.get("recent_own", 0.40), team_recent.get("corners_for_avg")),
            (csw.get("recent_opp_allow", 0.35), opp_rec.get("corners_against_avg")),
            (csw.get("recent_shots", 0.05), team_recent.get("shots_for_avg")),
            (csw.get("recent_sot", 0.05), team_recent.get("sot_for_avg")),
            (csw.get("recent_control", 0.08), team_recent.get("control_avg")),
            (csw.get("recent_possession", 0.07), team_recent.get("possession_avg")),
        ])

    team_season_score = _season_corner_share_score(team_meta, opp_meta, is_home)
    opp_season_score = _season_corner_share_score(opp_meta, team_meta, not is_home)
    season_share = _share_from_scores(
        team_season_score, opp_season_score,
        shrink=csw.get("shrink", 0.06),
        floor=csw.get("floor", 0.28),
        ceiling=csw.get("ceiling", 0.72),
    )

    team_recent_score = _recent_corner_share_score(recent_stats, opp_recent)
    opp_recent_score = _recent_corner_share_score(opp_recent, recent_stats)
    recent_share = _share_from_scores(
        team_recent_score, opp_recent_score,
        shrink=csw.get("shrink", 0.06),
        floor=csw.get("floor", 0.28),
        ceiling=csw.get("ceiling", 0.72),
    )

    share_blended = _weighted_component_blend([
        (season_share, csw.get("season_blend", 0.65)),
        (recent_share, csw.get("recent_blend", 0.35)),
    ])
    share_anchor = (total_blended * share_blended
                    if (total_blended is not None and share_blended is not None) else None)

    # ---- Step 4: Stabilizer (direct per-team tendency) ----
    stabilizer = _weighted_component_blend([
        (season_direct, cdw.get("season", 0.65)),
        (recent_direct, cdw.get("recent", 0.35)),
    ])

    # ---- Step 5: Final blend ----
    blended = _weighted_component_blend([
        (share_anchor, cow.get("match_anchor_weight", 0.55)),
        (stabilizer, cow.get("direct_stabilizer_weight", 0.45)),
    ])
    # Backward-compatible season_proj / recent_proj
    season_proj = (season_total * season_share
                   if (season_total is not None and season_share is not None) else season_direct)
    recent_proj = (recent_total * recent_share
                   if (recent_total is not None and recent_share is not None) else recent_direct)

    # League context already applied inside projected_total_corners — no re-application

    return {
        "match_total": total_blended,
        "season_total": season_total,
        "recent_total": recent_total,
        "season_share": season_share,
        "recent_share": recent_share,
        "share_blended": share_blended,
        "share_anchor": share_anchor,
        "season_direct": season_direct,
        "recent_direct": recent_direct,
        "stabilizer": stabilizer,
        "final": blended,
        "season_proj": season_proj,
        "recent_proj": recent_proj,
    }


def projected_team_corners(
    team: str,
    opponent: str,
    league: str,
    is_home: bool,
    league_ctx=None,
    fixture_date: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Per-team corner projection. Returns (final, season_component, recent_component)."""
    detail = projected_team_corners_detail(
        team, opponent, league, is_home,
        league_ctx=league_ctx,
        fixture_date=fixture_date,
    )
    return detail.get("final"), detail.get("season_proj"), detail.get("recent_proj")


def projected_team_cards(
    team: str,
    opponent: str,
    league: str,
    is_home: bool,
    ref_mod: "RefereeModifier" = None,
    lineup_ctx=None,
    league_ctx=None,
    fixture_date: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Per-team card projection from match-total anchor + team share + stabilizer."""
    tlw = SCORING_WEIGHTS.get("team_lines", {})
    cdw = tlw.get("cards_direct", {})
    cow = tlw.get("cards_output", {})
    csw = tlw.get("cards_share", {})
    csm = SCORING_WEIGHTS.get("cards_share_model", {})
    hm = _profile_as_of(team if is_home else opponent, league, fixture_date)
    am = _profile_as_of(opponent if is_home else team, league, fixture_date)
    team_meta = hm if is_home else am
    opp_meta = am if is_home else hm

    home_name = team if is_home else opponent
    away_name = opponent if is_home else team
    league_ctx = _resolve_league_context(home_name, away_name, league, league_ctx=league_ctx, fixture_date=fixture_date)

    vb = SCORING_WEIGHTS["projection_cards"].get("venue_blend", 0.0)
    own_overall = safe_float(team_meta.get("cards_per_90_team"))
    own_venue = safe_float(team_meta.get("cards_home_pm" if is_home else "cards_away_pm"))
    own_rate = _venue_blend(own_venue, own_overall, vb)
    opp_induced = safe_float(
        opp_meta.get("cards_induced_away_pm" if is_home else "cards_induced_home_pm")
    )
    if opp_induced is None:
        opp_induced = safe_float(opp_meta.get("opp_cards_induced_pm"))
    if own_rate is not None and opp_induced is not None:
        season_direct = SCORING_WEIGHTS["projection_cards"]["own"] * own_rate + SCORING_WEIGHTS["projection_cards"]["opp"] * opp_induced
    else:
        season_direct = own_rate if own_rate is not None else opp_induced

    recent_stats = _recent_stats(team, league, last_n=6, target_date=fixture_date)
    opp_recent = _recent_stats(opponent, league, last_n=6, target_date=fixture_date)
    recent_cards = recent_stats.get("cards_avg")
    recent_opp_induced = opp_recent.get("cards_induced_avg")
    recent_own_w = tlw.get("cards_recent_own", 0.50)
    if recent_cards is not None and recent_opp_induced is not None:
        recent_direct = recent_own_w * recent_cards + (1.0 - recent_own_w) * recent_opp_induced
    else:
        recent_direct = recent_cards if recent_cards is not None else recent_opp_induced

    stabilizer = _weighted_component_blend([
        (season_direct, cdw.get("season", 0.65)),
        (recent_direct, cdw.get("recent", 0.35)),
    ])

    # Use match-total anchor from detail helper
    total_blended, season_total, recent_total, ref_mod = projected_total_cards(
        home_name, away_name, league, ref_mod=ref_mod,
        lineup_ctx=lineup_ctx, league_ctx=league_ctx, fixture_date=fixture_date
    )

    def _season_share_score(meta: Dict, other_meta: Dict, home_side: bool) -> Optional[float]:
        opp_control = safe_float(other_meta.get("control_index"))
        own_fouls = _venue_blend(
            safe_float(meta.get("fouls_home_pm" if home_side else "fouls_away_pm")),
            safe_float(meta.get("fouls_per_90_team")),
            vb,
        )
        return _weighted_sum([
            (csw.get("season_own", 0.34), _venue_blend(
                safe_float(meta.get("cards_home_pm" if home_side else "cards_away_pm")),
                safe_float(meta.get("cards_per_90_team")),
                vb,
            )),
            (csw.get("season_opp_induced", 0.43), _venue_blend(
                safe_float(other_meta.get("cards_induced_away_pm" if home_side else "cards_induced_home_pm")),
                safe_float(other_meta.get("opp_cards_induced_pm")),
                vb,
            )),
            (csw.get("season_fouls", 0.06), own_fouls),
            (csw.get("season_aggression", 0.02), safe_float(meta.get("aggression_index_norm"))),
            (csw.get("season_cards_per_foul", 0.05), safe_float(meta.get("cards_per_foul_team"))),
            (csw.get("season_opp_control", 0.97), opp_control),
        ])

    def _recent_share_score(team_recent: Dict, other_recent: Dict) -> Optional[float]:
        return _weighted_sum([
            (csw.get("recent_own", 0.41), team_recent.get("cards_avg")),
            (csw.get("recent_opp_induced", 0.59), other_recent.get("cards_induced_avg")),
            (csw.get("recent_fouls", 0.10), team_recent.get("fouls_avg")),
            (csw.get("recent_aggression", 0.46), team_recent.get("aggression_avg")),
            (csw.get("recent_low_control", 1.47), 1.0 - team_recent.get("control_avg") if team_recent.get("control_avg") is not None else None),
            (csw.get("recent_opp_control", 0.84), other_recent.get("control_avg")),
        ])

    team_season_score = _season_share_score(team_meta, opp_meta, is_home)
    opp_season_score = _season_share_score(opp_meta, team_meta, not is_home)
    season_share = _share_from_scores(
        team_season_score,
        opp_season_score,
        shrink=csw.get("shrink", 0.06),
        floor=csw.get("floor", 0.26),
        ceiling=csw.get("ceiling", 0.71),
    )

    team_recent_score = _recent_share_score(recent_stats, opp_recent)
    opp_recent_score = _recent_share_score(opp_recent, recent_stats)
    recent_share = _share_from_scores(
        team_recent_score,
        opp_recent_score,
        shrink=csw.get("shrink", 0.06),
        floor=csw.get("floor", 0.26),
        ceiling=csw.get("ceiling", 0.71),
    )

    share_blended = _weighted_component_blend([
        (season_share, csw.get("season_blend", 0.68)),
        (recent_share, csw.get("recent_blend", 0.32)),
    ])
    share_anchor = total_blended * share_blended if (total_blended is not None and share_blended is not None) else None

    season_proj = season_total * season_share if (season_total is not None and season_share is not None) else season_direct
    recent_proj = recent_total * recent_share if (recent_total is not None and recent_share is not None) else recent_direct

    # Use cards_share_model weights if available, fall back to cards_output
    match_anchor_w = csm.get("match_anchor_weight", cow.get("share_weight", 0.50))
    stabilizer_w = csm.get("direct_stabilizer_weight", cow.get("stabilizer_weight", 0.50))
    blended = _weighted_component_blend([
        (share_anchor, match_anchor_w),
        (stabilizer, stabilizer_w),
    ])
    if blended is None:
        return None, None, None

    return blended, season_proj, recent_proj


def projected_total_goals(
    home: str,
    away: str,
    league: str,
    knockout_ctx: Optional[KnockoutContext] = None,
    league_ctx=None,
    fixture_date: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    league_ctx = _resolve_league_context(home, away, league, league_ctx=league_ctx, fixture_date=fixture_date)
    pw = SCORING_WEIGHTS["projection"]
    hm = _profile_as_of(home, league, fixture_date)
    am = _profile_as_of(away, league, fixture_date)

    h_proj, a_proj = projected_goals(hm, am)
    season_total = (h_proj + a_proj) if (h_proj is not None and a_proj is not None) else None

    hr = _recent_stats(home, league, last_n=6, target_date=fixture_date)
    ar = _recent_stats(away, league, last_n=6, target_date=fixture_date)
    h_xg = hr.get("xg_for_avg")
    a_xg = ar.get("xg_for_avg")
    recent_total = (h_xg + a_xg) if (h_xg is not None and a_xg is not None) else None

    blended = _blend_total_with_divergence("goals", season_total, recent_total)
    if blended is None:
        return None, None, None

    # Trend adjustment
    blended += _trend_adjustment(hr, ar, "goals_slope")

    # ML blend
    ml_w = _ml_blend_weight("goals")
    if ml_w > 0:
        ml_proj = ml_predict_total(home, away, league, "goals", home_meta=hm, away_meta=am)
        if ml_proj is not None:
            blended = (1.0 - ml_w) * blended + ml_w * ml_proj

    # Knockout round context adjustment
    if knockout_ctx and knockout_ctx.is_knockout and knockout_ctx.goals_modifier != 1.0:
        blended *= knockout_ctx.goals_modifier

    if league_ctx is not None:
        adj = getattr(league_ctx, "adjustments", {}).get("goals", 1.0)
        if adj != 1.0:
            blended *= adj

    return blended, season_total, recent_total


def _goal_market_team_projections(
    home: str,
    away: str,
    league: str,
    league_ctx=None,
    fixture_date: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Canonical team-goal split for BTTS / scoreline / moneyline / spreads."""
    bs, br = _stat_blend("goals")
    hm = _profile_as_of(home, league, fixture_date)
    am = _profile_as_of(away, league, fixture_date)
    h_season, a_season = projected_goals(hm, am)

    hr = _recent_stats(home, league, last_n=6, target_date=fixture_date)
    ar = _recent_stats(away, league, last_n=6, target_date=fixture_date)
    h_xg = hr.get("xg_for_avg")
    a_xg = ar.get("xg_for_avg")

    def _blend(season_val, recent_val):
        if season_val is not None and recent_val is not None:
            return bs * season_val + br * recent_val
        return season_val if season_val is not None else recent_val

    h_proj = _blend(h_season, h_xg)
    a_proj = _blend(a_season, a_xg)

    total_blended, _, _ = projected_total_goals(
        home,
        away,
        league,
        league_ctx=league_ctx,
        fixture_date=fixture_date,
    )
    if h_proj is not None and a_proj is not None and total_blended is not None:
        raw_total = h_proj + a_proj
        if raw_total > 0:
            scale = total_blended / raw_total
            h_proj *= scale
            a_proj *= scale

    return h_proj, a_proj, h_season, a_season


# ===================================================================
#  Derived projections (BTTS, correct score, moneyline, goal diff)
# ===================================================================

def projected_btts_prob(home: str, away: str, league: str,
                        league_ctx=None,
                        fixture_date: Optional[str] = None) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Project P(BTTS Yes) using Dixon-Coles bivariate Poisson.
    Returns (p_btts_yes, h_proj, a_proj, h_season, a_season).
    """
    h_proj, a_proj, h_season, a_season = _goal_market_team_projections(
        home, away, league, league_ctx=league_ctx, fixture_date=fixture_date,
    )

    if h_proj is None or a_proj is None:
        return None, None, None, h_season, a_season

    # Dixon-Coles bivariate Poisson for BTTS (accounts for goal correlation)
    p_btts_yes = dixon_coles_btts_prob(h_proj, a_proj)

    return p_btts_yes, h_proj, a_proj, h_season, a_season


def projected_correct_score_probs(
    home: str, away: str, league: str, max_goals: int = 6,
    league_ctx=None,
    fixture_date: Optional[str] = None,
) -> Tuple[Optional[Dict[Tuple[int, int], float]], Optional[float], Optional[float]]:
    """
    Compute P(Home=i, Away=j) for all i,j in [0, max_goals] using Dixon-Coles
    bivariate Poisson (rho correction for low-scoring outcomes).
    Returns (prob_matrix, h_proj, a_proj).
    """
    h_proj, a_proj, _, _ = _goal_market_team_projections(
        home, away, league, league_ctx=league_ctx, fixture_date=fixture_date,
    )
    if h_proj is None or a_proj is None:
        return None, None, None

    matrix = dixon_coles_scoreline_matrix(h_proj, a_proj, rho=-0.10, max_goals=max_goals)
    probs: Dict[Tuple[int, int], float] = {}
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            probs[(i, j)] = matrix[i][j]

    return probs, h_proj, a_proj


def projected_moneyline_probs(
    home: str, away: str, league: str,
    league_ctx=None,
    fixture_date: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Compute P(Home Win), P(Draw), P(Away Win) from the Poisson/NegBin scoreline matrix.
    Returns (p_home, p_draw, p_away, h_proj, a_proj).
    """
    probs, h_proj, a_proj = projected_correct_score_probs(
        home, away, league, league_ctx=league_ctx, fixture_date=fixture_date,
    )
    if probs is None:
        return None, None, None, None, None

    p_home = sum(p for (h, a), p in probs.items() if h > a)
    p_draw = sum(p for (h, a), p in probs.items() if h == a)
    p_away = sum(p for (h, a), p in probs.items() if h < a)

    return p_home, p_draw, p_away, h_proj, a_proj


def projected_goal_difference(home: str, away: str, league: str,
                              league_ctx=None,
                              fixture_date: Optional[str] = None) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Project expected goal difference (home - away). Positive = home favored.
    Returns (blended_diff, season_diff, recent_diff).
    """
    h_proj, a_proj, h_season, a_season = _goal_market_team_projections(
        home, away, league, league_ctx=league_ctx, fixture_date=fixture_date,
    )
    if h_proj is None or a_proj is None:
        return None, None, None

    probs, _, _ = projected_correct_score_probs(
        home, away, league, max_goals=7, league_ctx=league_ctx, fixture_date=fixture_date,
    )
    blended_diff = sum((h - a) * p for (h, a), p in (probs or {}).items()) if probs else (h_proj - a_proj)
    season_diff = (h_season - a_season) if (h_season is not None and a_season is not None) else None

    hr = _recent_stats(home, league, last_n=6, target_date=fixture_date)
    ar = _recent_stats(away, league, last_n=6, target_date=fixture_date)
    h_xg = hr.get("xg_for_avg")
    a_xg = ar.get("xg_for_avg")
    recent_diff = (h_xg - a_xg) if (h_xg is not None and a_xg is not None) else None

    return blended_diff, season_diff, recent_diff


# ---------------------------------------------------------------------------
#  Dynamic margin standard deviation for spread cover probability
# ---------------------------------------------------------------------------

_EUROPEAN_COMPETITIONS = {"UCL", "UEL", "UECL"}
_EURO_COMP_UPLIFT = 0.15  # extra std for cross-confederation matchups
_MARGIN_STD_FLOOR = 1.15
_MARGIN_STD_CAP = 1.85


def compute_margin_std(home: str, away: str, league: str,
                       fixture_date: Optional[str] = None) -> float:
    """Compute match-specific goal-difference standard deviation.

    Uses blended (season 70% + recent 30%) goals variance from both teams.
    Match-level variance of (home_goals - away_goals) ≈ Var(home) + Var(away)
    assuming independence.  European competitions get an uplift because
    cross-confederation matchups and knockout formats produce higher variance.

    Falls back to the static default (1.25) when data is unavailable.
    """
    default = SCORING_WEIGHTS["prob"].get("margin_std_default", 1.25)

    h_var = get_blended_variance(home, league, "goals_var", fixture_date=fixture_date)
    a_var = get_blended_variance(away, league, "goals_var", fixture_date=fixture_date)

    if h_var is None and a_var is None:
        std = default
    else:
        # If only one side is available, double it as a rough proxy
        if h_var is None:
            match_var = 2.0 * a_var
        elif a_var is None:
            match_var = 2.0 * h_var
        else:
            match_var = h_var + a_var
        std = match_var ** 0.5

    # European competition uplift
    if league in _EUROPEAN_COMPETITIONS:
        std += _EURO_COMP_UPLIFT

    return max(_MARGIN_STD_FLOOR, min(_MARGIN_STD_CAP, std))
