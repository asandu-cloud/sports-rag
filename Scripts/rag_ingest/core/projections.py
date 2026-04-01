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
        dixon_coles_scoreline_prob, dixon_coles_btts_prob,
    )
except ImportError:
    from Scripts.rag_ingest.prob_models import (  # type: ignore[import]
        over_prob, poisson_pmf, negbin_pmf, negbin_from_mean_var,
        dixon_coles_scoreline_prob, dixon_coles_btts_prob,
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


# ===================================================================
#  Match-total projections (blended season + recent + ML)
# ===================================================================

def projected_total_sot(home: str, away: str, league: str, knockout_ctx: Optional[KnockoutContext] = None) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    hm = _profile_meta(home, league)
    am = _profile_meta(away, league)
    h_proj, a_proj = projected_sot(hm, am)
    season_total = (h_proj + a_proj) if (h_proj is not None and a_proj is not None) else None

    hr = _recent_stats(home, league, last_n=6)
    ar = _recent_stats(away, league, last_n=6)
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

    return blended, season_total, recent_total


def projected_total_corners(home: str, away: str, league: str, knockout_ctx: Optional[KnockoutContext] = None) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    hm = _profile_meta(home, league)
    am = _profile_meta(away, league)
    h_proj, a_proj = projected_corners(hm, am)
    season_total = (h_proj + a_proj) if (h_proj is not None and a_proj is not None) else None

    hr = _recent_stats(home, league, last_n=6)
    ar = _recent_stats(away, league, last_n=6)
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

    return blended, season_total, recent_total


def projected_total_cards(home: str, away: str, league: str, knockout_ctx: Optional[KnockoutContext] = None) -> Tuple[Optional[float], Optional[float], Optional[float], RefereeModifier]:
    pw = SCORING_WEIGHTS["projection"]
    rw = SCORING_WEIGHTS["referee"]
    hm = _profile_meta(home, league)
    am = _profile_meta(away, league)

    ref_mod = get_referee_modifier(home, away, league,
                                   weight=rw.get("modifier_weight", 0.25))

    # Season projection (referee modifier applied inside projected_cards)
    h_proj, a_proj = projected_cards(hm, am, referee_modifier=ref_mod.multiplier)
    season_total = (h_proj + a_proj) if (h_proj is not None and a_proj is not None) else None

    hr = _recent_stats(home, league, last_n=6)
    ar = _recent_stats(away, league, last_n=6)
    h_recent_cards = hr.get("cards_avg")
    a_recent_cards = ar.get("cards_avg")
    # Blend each team's recent cards with opponent's recent card-inducing tendency
    # For cards, opponent "defensive" form = how many cards opponents get against them
    if h_recent_cards is not None and a_recent_cards is not None:
        # Use opponent's cards_avg as proxy for cards conceded/induced context
        h_recent_proj = 0.6 * h_recent_cards + 0.4 * a_recent_cards
        a_recent_proj = 0.6 * a_recent_cards + 0.4 * h_recent_cards
        recent_total = h_recent_proj + a_recent_proj
    else:
        recent_total = None

    blended = _blend_total_with_divergence("cards", season_total, recent_total)
    if blended is None:
        return None, None, None, ref_mod

    # Trend adjustment
    blended += _trend_adjustment(hr, ar, "cards_slope")

    # Referee anchor: blend the referee's own avg cards/match into the projection.
    # This gives the referee direct influence beyond the multiplicative modifier,
    # especially for strict/lenient refs with enough sample.
    if (ref_mod.source == "profile" and ref_mod.avg_cards_per_match > 0
            and ref_mod.sample_size >= rw.get("min_sample_size", 5)):
        ref_anchor_w = rw.get("anchor_weight", 0.15) * ref_mod.confidence
        blended = (1.0 - ref_anchor_w) * blended + ref_anchor_w * ref_mod.avg_cards_per_match

    # Foul-based card estimate: predicted_fouls x referee cards_per_foul
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

    # ML blend
    ml_w = _ml_blend_weight("cards")
    if ml_w > 0:
        ml_proj = ml_predict_total(home, away, league, "cards", home_meta=hm, away_meta=am)
        if ml_proj is not None:
            blended = (1.0 - ml_w) * blended + ml_w * ml_proj

    # Knockout round context adjustment
    if knockout_ctx and knockout_ctx.is_knockout and knockout_ctx.cards_modifier != 1.0:
        blended *= knockout_ctx.cards_modifier

    return blended, season_total, recent_total, ref_mod


# ---- Per-team blended projections (for team totals lines) ----

def projected_team_corners(
    team: str, opponent: str, league: str, is_home: bool,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Per-team blended corner projection. Returns (blended, season, recent)."""
    pw = SCORING_WEIGHTS["projection"]
    hm = _profile_meta(team if is_home else opponent, league)
    am = _profile_meta(opponent if is_home else team, league)

    h_proj, a_proj = projected_corners(hm, am)
    season_proj = h_proj if is_home else a_proj

    recent_stats = _recent_stats(team, league, last_n=6)
    recent_corners = recent_stats.get("corners_for_avg")

    if season_proj is not None and recent_corners is not None:
        blended = pw["blend_season"] * season_proj + pw["blend_recent"] * recent_corners
    elif season_proj is not None:
        blended = season_proj
    elif recent_corners is not None:
        blended = recent_corners
    else:
        return None, None, None

    # ML proportional split
    ml_w = _ml_blend_weight("corners")
    if ml_w > 0:
        home_name = team if is_home else opponent
        away_name = opponent if is_home else team
        ml_proj = ml_predict_total(home_name, away_name, league, "corners",
                                   home_meta=hm, away_meta=am)
        if ml_proj is not None and h_proj is not None and a_proj is not None:
            total_season = h_proj + a_proj
            if total_season > 0:
                team_share = (h_proj if is_home else a_proj) / total_season
                blended = (1.0 - ml_w) * blended + ml_w * (ml_proj * team_share)

    return blended, season_proj, recent_corners


def projected_team_cards(
    team: str, opponent: str, league: str, is_home: bool,
    ref_mod: "RefereeModifier" = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Per-team blended card projection. Returns (blended, season, recent)."""
    pw = SCORING_WEIGHTS["projection"]
    rw = SCORING_WEIGHTS["referee"]
    hm = _profile_meta(team if is_home else opponent, league)
    am = _profile_meta(opponent if is_home else team, league)

    if ref_mod is None:
        home_name = team if is_home else opponent
        away_name = opponent if is_home else team
        ref_mod = get_referee_modifier(home_name, away_name, league,
                                       weight=rw.get("modifier_weight", 0.25))

    # Use referee_modifier=1.0 here — referee signal comes through anchor and
    # foul-based estimate below, not multiplicatively (avoids triple-counting).
    h_proj, a_proj = projected_cards(hm, am, referee_modifier=1.0)
    season_proj = h_proj if is_home else a_proj

    recent_stats = _recent_stats(team, league, last_n=6)
    recent_cards = recent_stats.get("cards_avg")

    if season_proj is not None and recent_cards is not None:
        blended = pw["blend_season"] * season_proj + pw["blend_recent"] * recent_cards
    elif season_proj is not None:
        blended = season_proj
    elif recent_cards is not None:
        blended = recent_cards
    else:
        return None, None, None

    # Referee anchor scaled to per-team (divide match avg by 2)
    if (ref_mod.source == "profile" and ref_mod.avg_cards_per_match > 0
            and ref_mod.sample_size >= rw.get("min_sample_size", 5)):
        ref_anchor_w = rw.get("anchor_weight", 0.15) * ref_mod.confidence
        per_team_ref_avg = ref_mod.avg_cards_per_match / 2.0
        blended = (1.0 - ref_anchor_w) * blended + ref_anchor_w * per_team_ref_avg

    # Foul-based card estimate: this team's predicted fouls x ref cards_per_foul
    cpf = ref_mod.cards_per_foul if ref_mod.source == "profile" else 0.0
    if cpf > 0 and ref_mod.sample_size >= rw.get("min_sample_size", 5):
        pc = SCORING_WEIGHTS["projection_cards"]
        team_meta = hm if is_home else am
        fouls_s = safe_float(team_meta.get("fouls_per_90_team"))
        fouls_r = recent_stats.get("fouls_avg")
        if fouls_s is not None and fouls_r is not None:
            team_fouls = pw["blend_season"] * fouls_s + pw["blend_recent"] * fouls_r
        else:
            team_fouls = fouls_s if fouls_s is not None else fouls_r
        if team_fouls is not None:
            foul_based_cards = team_fouls * cpf
            foul_w = pc.get("foul_card_blend", 0.15) * ref_mod.confidence
            blended = (1.0 - foul_w) * blended + foul_w * foul_based_cards

    # ML proportional split
    ml_w = _ml_blend_weight("cards")
    if ml_w > 0:
        home_name = team if is_home else opponent
        away_name = opponent if is_home else team
        ml_proj = ml_predict_total(home_name, away_name, league, "cards",
                                   home_meta=hm, away_meta=am)
        if ml_proj is not None and h_proj is not None and a_proj is not None:
            total_season = h_proj + a_proj
            if total_season > 0:
                team_share = (h_proj if is_home else a_proj) / total_season
                blended = (1.0 - ml_w) * blended + ml_w * (ml_proj * team_share)

    return blended, season_proj, recent_cards


def projected_total_goals(home: str, away: str, league: str, knockout_ctx: Optional[KnockoutContext] = None) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    pw = SCORING_WEIGHTS["projection"]
    hm = _profile_meta(home, league)
    am = _profile_meta(away, league)

    h_proj, a_proj = projected_goals(hm, am)
    season_total = (h_proj + a_proj) if (h_proj is not None and a_proj is not None) else None

    hr = _recent_stats(home, league, last_n=6)
    ar = _recent_stats(away, league, last_n=6)
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

    return blended, season_total, recent_total


# ===================================================================
#  Derived projections (BTTS, correct score, moneyline, goal diff)
# ===================================================================

def projected_btts_prob(home: str, away: str, league: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Project P(BTTS Yes) using Dixon-Coles bivariate Poisson.
    Returns (p_btts_yes, h_proj, a_proj, h_season, a_season).
    """
    bs, br = _stat_blend("goals")
    hm = _profile_meta(home, league)
    am = _profile_meta(away, league)

    h_season, a_season = projected_goals(hm, am)

    hr = _recent_stats(home, league, last_n=6)
    ar = _recent_stats(away, league, last_n=6)
    h_xg = hr.get("xg_for_avg")
    a_xg = ar.get("xg_for_avg")

    def _blend(season_val, recent_val):
        if season_val is not None and recent_val is not None:
            return bs * season_val + br * recent_val
        return season_val if season_val is not None else recent_val

    h_proj = _blend(h_season, h_xg)
    a_proj = _blend(a_season, a_xg)

    if h_proj is None or a_proj is None:
        return None, None, None, h_season, a_season

    # ML blend: split ML total proportionally between home/away
    ml_w = _ml_blend_weight("goals")
    if ml_w > 0:
        ml_proj = ml_predict_total(home, away, league, "goals", home_meta=hm, away_meta=am)
        if ml_proj is not None and (h_proj + a_proj) > 0:
            ratio_h = h_proj / (h_proj + a_proj)
            h_proj = (1.0 - ml_w) * h_proj + ml_w * (ml_proj * ratio_h)
            a_proj = (1.0 - ml_w) * a_proj + ml_w * (ml_proj * (1.0 - ratio_h))

    # Per-team variance for distribution selection (Bayesian blend: season + recent)
    h_var = get_blended_variance(home, league, "goals_var")
    a_var = get_blended_variance(away, league, "goals_var")

    # Dixon-Coles bivariate Poisson for BTTS (accounts for goal correlation)
    p_btts_yes = dixon_coles_btts_prob(h_proj, a_proj)

    return p_btts_yes, h_proj, a_proj, h_season, a_season


def projected_correct_score_probs(
    home: str, away: str, league: str, max_goals: int = 6,
) -> Tuple[Optional[Dict[Tuple[int, int], float]], Optional[float], Optional[float]]:
    """
    Compute P(Home=i, Away=j) for all i,j in [0, max_goals] using Dixon-Coles
    bivariate Poisson (rho correction for low-scoring outcomes).
    Returns (prob_matrix, h_proj, a_proj).
    """
    bs, br = _stat_blend("goals")
    hm = _profile_meta(home, league)
    am = _profile_meta(away, league)
    h_season, a_season = projected_goals(hm, am)

    hr = _recent_stats(home, league, last_n=6)
    ar = _recent_stats(away, league, last_n=6)
    h_xg = hr.get("xg_for_avg")
    a_xg = ar.get("xg_for_avg")

    def _blend(season_val, recent_val):
        if season_val is not None and recent_val is not None:
            return bs * season_val + br * recent_val
        return season_val if season_val is not None else recent_val

    h_proj = _blend(h_season, h_xg)
    a_proj = _blend(a_season, a_xg)
    if h_proj is None or a_proj is None:
        return None, None, None

    # ML blend (same pipeline as projected_btts_prob)
    ml_w = _ml_blend_weight("goals")
    if ml_w > 0:
        ml_proj = ml_predict_total(home, away, league, "goals", home_meta=hm, away_meta=am)
        if ml_proj is not None and (h_proj + a_proj) > 0:
            ratio_h = h_proj / (h_proj + a_proj)
            h_proj = (1.0 - ml_w) * h_proj + ml_w * (ml_proj * ratio_h)
            a_proj = (1.0 - ml_w) * a_proj + ml_w * (ml_proj * (1.0 - ratio_h))

    # Dixon-Coles bivariate Poisson with rho correction
    probs: Dict[Tuple[int, int], float] = {}
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            probs[(i, j)] = dixon_coles_scoreline_prob(i, j, h_proj, a_proj, rho=-0.10)

    return probs, h_proj, a_proj


def projected_moneyline_probs(
    home: str, away: str, league: str,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Compute P(Home Win), P(Draw), P(Away Win) from the Poisson/NegBin scoreline matrix.
    Returns (p_home, p_draw, p_away, h_proj, a_proj).
    """
    probs, h_proj, a_proj = projected_correct_score_probs(home, away, league)
    if probs is None:
        return None, None, None, None, None

    p_home = sum(p for (h, a), p in probs.items() if h > a)
    p_draw = sum(p for (h, a), p in probs.items() if h == a)
    p_away = sum(p for (h, a), p in probs.items() if h < a)

    return p_home, p_draw, p_away, h_proj, a_proj


def projected_goal_difference(home: str, away: str, league: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Project expected goal difference (home - away). Positive = home favored.
    Returns (blended_diff, season_diff, recent_diff).
    """
    pw = SCORING_WEIGHTS["projection_spreads"]
    hm = _profile_meta(home, league)
    am = _profile_meta(away, league)

    h_g = safe_float(hm.get("goals_for_pm"))
    a_g = safe_float(am.get("goals_for_pm"))
    h_form = safe_float(hm.get("form_index_team"))
    a_form = safe_float(am.get("form_index_team"))
    h_dom = safe_float(hm.get("dominance_index"))
    a_dom = safe_float(am.get("dominance_index"))

    # Home advantage for goal difference
    proj_w = SCORING_WEIGHTS["projection"]
    ha = proj_w.get("home_advantage", 0.0) * 2  # full spread: +ha for home, -ha for away = 2*ha diff

    season_diff = None
    if h_g is not None and a_g is not None:
        season_diff = (h_g - a_g) + ha
        if h_form is not None and a_form is not None:
            season_diff += (h_form - a_form) * pw["form_w"]
        if h_dom is not None and a_dom is not None:
            season_diff += (h_dom - a_dom) * pw["dominance_w"]

    hr = _recent_stats(home, league, last_n=6)
    ar = _recent_stats(away, league, last_n=6)
    h_xg = hr.get("xg_for_avg")
    a_xg = ar.get("xg_for_avg")
    recent_diff = (h_xg - a_xg) if (h_xg is not None and a_xg is not None) else None

    if season_diff is not None and recent_diff is not None:
        return (pw["blend_season"] * season_diff + pw["blend_recent"] * recent_diff), season_diff, recent_diff
    if season_diff is not None:
        return season_diff, season_diff, None
    if recent_diff is not None:
        return recent_diff, None, recent_diff
    return None, None, None
