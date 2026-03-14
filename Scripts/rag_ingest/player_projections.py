"""Player prop projection engine.

Deterministic per-player projections for goals, SoT, assists, cards.
Combines player rates with team context and opponent strength.
Uses Poisson/NegBin from prob_models.py for probability distributions.

No external dependencies beyond existing RAG infrastructure.
Graceful degradation: returns None when data insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from prob_models import over_prob
except ImportError:
    from Scripts.rag_ingest.prob_models import over_prob


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PlayerProjection:
    """Point estimate + probability distribution for a single player stat."""
    player_name: str
    player_id: int
    team: str
    stat: str                         # "goals" | "sot" | "assists" | "cards"
    projection: float                 # expected value (e.g., 0.42 goals)
    minutes_projection: float         # expected minutes (e.g., 82.0)
    minutes_risk: str                 # "nailed_on" | "likely_starter" | "rotation" | "bench"
    start_probability: float          # from start_rate
    prob_over_0_5: Optional[float]    # P(X >= 1)
    prob_over_1_5: Optional[float]    # P(X >= 2)
    prob_over_2_5: Optional[float]    # P(X >= 3)
    confidence: str                   # "high" | "medium" | "low"
    evidence: Dict                    # raw numbers for rendering
    opponent_adjustment: float        # multiplier applied
    data_quality: str                 # "full" | "limited" | "insufficient"


@dataclass
class PlayerPropRecommendation:
    """Final recommendation for a player prop market."""
    projection: PlayerProjection
    recommended_line: float           # e.g., 0.5
    recommended_side: str             # "over" | "under"
    model_prob: float                 # P(outcome) for recommended side
    rationale: List[str]              # human-readable evidence lines


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLAYER_PROP_WEIGHTS = {
    "season_blend": 0.75,
    "recent_blend": 0.25,
    "team_share_weight": 0.30,
    "opponent_defense_weight": 0.40,

    # Opponent adjustment bounds
    "max_opp_multiplier": 1.40,
    "min_opp_multiplier": 0.65,

    # Minutes risk
    "rotation_minutes_discount": 0.70,
    "bench_minutes_discount": 0.25,
    "min_start_rate_for_rec": 0.40,

    # Confidence thresholds
    "confidence_high_prob": 0.60,
    "confidence_medium_prob": 0.45,
    "min_qualified_apps": 5,
    "min_qualified_apps_limited": 3,
    "min_total_minutes": 180,
}

STAT_LINES = {
    "goals": [0.5, 1.5, 2.5],
    "sot": [0.5, 1.5, 2.5, 3.5],
    "assists": [0.5, 1.5],
    "cards": [0.5, 1.5],
}

LEAGUE_DEFAULTS = {
    "goals": 1.30,
    "sot": 3.80,
    "cards": 1.80,
    "assists": 0.85,
}


# ---------------------------------------------------------------------------
# Core projection
# ---------------------------------------------------------------------------

def projected_player_stat(
    player_meta: Dict,
    team_meta: Dict,
    opponent_meta: Dict,
    stat: str,
    league: str,
    is_home: bool,
    recent_fixtures: Optional[List[Dict]] = None,
) -> Optional[PlayerProjection]:
    """Compute expected value for a player stat in a specific fixture.

    Pipeline:
    1. Player's own per-90 rate (starter appearances preferred)
    2. Adjust for expected minutes (starter/rotation/bench)
    3. Scale by team context (team output vs league average)
    4. Adjust for opponent defensive quality
    5. Blend season rate with recent form
    6. Feed to Poisson for probability distribution
    """
    pw = PLAYER_PROP_WEIGHTS

    # --- Data quality gate ---
    role = player_meta.get("minutes_risk") or player_meta.get("role", "bench")
    start_rate = _safe_float(player_meta.get("start_rate"), 0.0)
    total_minutes = int(_safe_float(player_meta.get("total_minutes")
                                     or player_meta.get("minutes"), 0))
    qualified_apps = int(_safe_float(player_meta.get("qualified_appearances")
                                     or player_meta.get("appearances_as_starter"), 0))
    total_apps = int(_safe_float(player_meta.get("appearances_total")
                                 or player_meta.get("total_appearances"), 0))

    if total_minutes < pw["min_total_minutes"]:
        return None
    if qualified_apps < pw["min_qualified_apps_limited"]:
        return None

    data_quality = "full" if qualified_apps >= pw["min_qualified_apps"] else "limited"

    # --- Player's own per-90 rate (prefer starter-only rates) ---
    starter_rate_field = _stat_to_starter_rate_field(stat)
    base_rate = _safe_float(player_meta.get(starter_rate_field))

    # Fallback to overall rate if starter rate unavailable
    if base_rate is None or base_rate < 0:
        rate_field = _stat_to_rate_field(stat)
        base_rate = _safe_float(player_meta.get(rate_field))

    if base_rate is None or base_rate < 0:
        return None

    # --- Minutes projection ---
    avg_minutes = _safe_float(player_meta.get("avg_minutes_per_appearance"), 70.0)

    if role in ("nailed_on", "starter"):
        expected_minutes = min(avg_minutes, 90.0)
    elif role in ("likely_starter",):
        expected_minutes = min(avg_minutes, 85.0)
    elif role == "rotation":
        expected_minutes = avg_minutes * pw["rotation_minutes_discount"]
    else:  # bench
        expected_minutes = avg_minutes * pw["bench_minutes_discount"]

    # --- Scale rate to expected minutes ---
    minutes_factor = expected_minutes / 90.0
    season_projection = base_rate * minutes_factor

    # --- Recent form blend ---
    recent_rate = _compute_recent_rate(recent_fixtures, stat, min_minutes=30)
    if recent_rate is not None:
        recent_projection = recent_rate * minutes_factor
        blended = (pw["season_blend"] * season_projection
                   + pw["recent_blend"] * recent_projection)
    else:
        blended = season_projection

    # --- Team context adjustment ---
    team_share = _team_share_adjustment(team_meta, stat)
    blended *= team_share

    # --- Opponent adjustment ---
    opp_multiplier = _opponent_adjustment(opponent_meta, stat)
    final_projection = blended * opp_multiplier
    final_projection = max(0.01, final_projection)

    # --- Poisson probabilities ---
    variance = _player_stat_variance(recent_fixtures, stat) if recent_fixtures else None
    prob_over_05 = over_prob(final_projection, 0.5, variance)
    prob_over_15 = over_prob(final_projection, 1.5, variance)
    prob_over_25 = (over_prob(final_projection, 2.5, variance)
                    if stat in ("goals", "sot") else None)

    # --- Confidence ---
    confidence = _player_confidence(prob_over_05, data_quality, role, qualified_apps)

    return PlayerProjection(
        player_name=player_meta.get("player_name", "Unknown"),
        player_id=int(_safe_float(player_meta.get("player_id"), 0)),
        team=player_meta.get("team", "Unknown"),
        stat=stat,
        projection=round(final_projection, 3),
        minutes_projection=round(expected_minutes, 1),
        minutes_risk=role,
        start_probability=round(start_rate, 2),
        prob_over_0_5=round(prob_over_05, 4) if prob_over_05 is not None else None,
        prob_over_1_5=round(prob_over_15, 4) if prob_over_15 is not None else None,
        prob_over_2_5=round(prob_over_25, 4) if prob_over_25 is not None else None,
        confidence=confidence,
        evidence={
            "base_rate_per_90": round(base_rate, 3),
            "expected_minutes": round(expected_minutes, 1),
            "season_projection": round(season_projection, 3),
            "recent_rate_per_90": round(recent_rate, 3) if recent_rate is not None else None,
            "team_share_factor": round(team_share, 3),
            "opponent_multiplier": round(opp_multiplier, 3),
            "qualified_apps": qualified_apps,
            "total_apps": total_apps,
        },
        opponent_adjustment=round(opp_multiplier, 3),
        data_quality=data_quality,
    )


def recommend_player_prop(
    projection: PlayerProjection,
) -> Optional[PlayerPropRecommendation]:
    """Pick the best line and side for a player prop recommendation.

    Returns None if the player doesn't meet minimum start rate.
    """
    if projection is None:
        return None

    pw = PLAYER_PROP_WEIGHTS
    if projection.start_probability < pw["min_start_rate_for_rec"]:
        return None

    lines = STAT_LINES.get(projection.stat, [0.5])
    best_line = None
    best_side = None
    best_prob = 0.0

    lam = projection.projection
    for line in lines:
        p_over = over_prob(lam, line)
        p_under = 1.0 - p_over

        if p_over >= 0.55 and p_over > best_prob:
            best_line = line
            best_side = "over"
            best_prob = p_over
        elif p_under >= 0.60 and p_under > best_prob:
            best_line = line
            best_side = "under"
            best_prob = p_under

    if best_line is None:
        p_over_05 = projection.prob_over_0_5 or over_prob(lam, 0.5)
        best_line = 0.5
        if p_over_05 >= 0.50:
            best_side = "over"
            best_prob = p_over_05
        else:
            best_side = "under"
            best_prob = 1.0 - p_over_05

    rationale = _build_rationale(projection, best_line, best_side, best_prob)

    return PlayerPropRecommendation(
        projection=projection,
        recommended_line=best_line,
        recommended_side=best_side,
        model_prob=round(best_prob, 4),
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Opponent & team adjustments
# ---------------------------------------------------------------------------

def _opponent_adjustment(opponent_meta: Dict, stat: str) -> float:
    """Opponent defensive quality multiplier.

    Leaky defense (high goals_against_pm) → multiplier > 1.0.
    Stingy defense → multiplier < 1.0.
    """
    pw = PLAYER_PROP_WEIGHTS

    if stat in ("goals", "assists"):
        opp_concedes = _safe_float(opponent_meta.get("goals_against_pm"))
        league_avg = LEAGUE_DEFAULTS.get("goals", 1.30)
    elif stat == "sot":
        opp_concedes = _safe_float(opponent_meta.get("sot_against_pm"))
        league_avg = LEAGUE_DEFAULTS.get("sot", 3.80)
    elif stat == "cards":
        opp_concedes = _safe_float(opponent_meta.get("opp_cards_induced_pm"))
        league_avg = LEAGUE_DEFAULTS.get("cards", 1.80)
    else:
        return 1.0

    if opp_concedes is None or opp_concedes <= 0 or league_avg <= 0:
        return 1.0

    raw = opp_concedes / league_avg
    clamped = max(pw["min_opp_multiplier"], min(pw["max_opp_multiplier"], raw))
    return 1.0 + pw["opponent_defense_weight"] * (clamped - 1.0)


def _team_share_adjustment(team_meta: Dict, stat: str) -> float:
    """Team output context adjustment.

    High-scoring team → boost attackers. Low-scoring → penalize.
    """
    pw = PLAYER_PROP_WEIGHTS

    if stat in ("goals", "assists"):
        team_output = _safe_float(team_meta.get("goals_for_pm"))
        league_avg = LEAGUE_DEFAULTS.get("goals", 1.30)
    elif stat == "sot":
        team_output = _safe_float(team_meta.get("sot_for_pm"))
        league_avg = LEAGUE_DEFAULTS.get("sot", 3.80)
    elif stat == "cards":
        team_output = _safe_float(team_meta.get("cards_per_90_team"))
        league_avg = LEAGUE_DEFAULTS.get("cards", 1.80)
    else:
        return 1.0

    if team_output is None or team_output <= 0 or league_avg <= 0:
        return 1.0

    team_relative = team_output / league_avg
    return 1.0 + pw["team_share_weight"] * (team_relative - 1.0)


# ---------------------------------------------------------------------------
# Recent form
# ---------------------------------------------------------------------------

def _compute_recent_rate(
    recent_fixtures: Optional[List[Dict]],
    stat: str,
    min_minutes: int = 30,
) -> Optional[float]:
    """Per-90 rate from recent fixture docs with exponential decay."""
    if not recent_fixtures:
        return None

    ALPHA = 0.85
    stat_field = _stat_to_raw_field(stat)

    qualified = []
    for f in recent_fixtures:
        meta = f if isinstance(f, dict) and "minutes" in f else f.get("meta", f)
        mins = int(_safe_float(meta.get("minutes"), 0))
        if mins >= min_minutes:
            val = _safe_float(meta.get(stat_field), 0.0)
            qualified.append((mins, val))

    if len(qualified) < 2:
        return None

    total_weighted_rate = 0.0
    total_weight = 0.0
    for i, (mins, val) in enumerate(qualified[:6]):
        w = ALPHA ** i
        rate_per_90 = (val / mins) * 90.0 if mins > 0 else 0.0
        total_weighted_rate += w * rate_per_90
        total_weight += w

    return total_weighted_rate / total_weight if total_weight > 0 else None


def _player_stat_variance(
    recent_fixtures: Optional[List[Dict]],
    stat: str,
    min_minutes: int = 30,
) -> Optional[float]:
    """Per-match variance for Poisson vs NegBin selection."""
    if not recent_fixtures:
        return None

    stat_field = _stat_to_raw_field(stat)
    values = []
    for f in recent_fixtures:
        meta = f if isinstance(f, dict) and "minutes" in f else f.get("meta", f)
        mins = int(_safe_float(meta.get("minutes"), 0))
        if mins >= min_minutes:
            values.append(_safe_float(meta.get(stat_field), 0.0))

    if len(values) < 4:
        return None

    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return var


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def _player_confidence(
    prob_over_05: Optional[float],
    data_quality: str,
    role: str,
    qualified_apps: int,
) -> str:
    """Map projections to confidence levels.

    Model-only (no bookmaker comparison), so confidence is based on:
    model probability strength, data quality, and minutes risk.
    """
    pw = PLAYER_PROP_WEIGHTS

    if prob_over_05 is None:
        return "low"

    if prob_over_05 >= pw["confidence_high_prob"]:
        conf = "high"
    elif prob_over_05 >= pw["confidence_medium_prob"]:
        conf = "medium"
    else:
        conf = "low"

    # Downgrade for data quality
    if data_quality == "limited" and conf == "high":
        conf = "medium"

    # Downgrade for rotation/bench
    if role == "rotation" and conf == "high":
        conf = "medium"
    elif role == "bench":
        conf = "low"

    return conf


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_player_prop_answer(
    league: str,
    recommendations: List[PlayerPropRecommendation],
) -> str:
    """Render player prop recommendations in structured format."""
    lines = [f"Player Prop Analysis — {league}", "=" * 50, ""]

    if not recommendations:
        lines.append("No player prop recommendations meet confidence thresholds.")
        lines.append("This can happen when player data is sparse or all candidates "
                     "are rotation/bench players.")
        return "\n".join(lines)

    for rec in recommendations:
        p = rec.projection
        lines.append(f"  {p.player_name} ({p.team}) — {p.stat.upper()}")
        lines.append(f"    Role: {p.minutes_risk} | Start rate: {p.start_probability:.0%} | "
                     f"Expected minutes: {p.minutes_projection:.0f}")
        lines.append(f"    Projection: {p.projection:.2f} {p.stat} | "
                     f"P(Over 0.5): {p.prob_over_0_5:.0%}"
                     + (f" | P(Over 1.5): {p.prob_over_1_5:.0%}" if p.prob_over_1_5 else ""))
        opp_sign = "+" if p.opponent_adjustment > 1 else ""
        lines.append(f"    Opponent adj: {opp_sign}"
                     f"{(p.opponent_adjustment - 1) * 100:.0f}%")
        lines.append(f"    >>> Recommended: {rec.recommended_side.upper()} "
                     f"{rec.recommended_line} | Model P: {rec.model_prob:.0%} | "
                     f"Confidence: {p.confidence.upper()}")
        if p.data_quality != "full":
            lines.append(f"    [!] Data quality: {p.data_quality} "
                         f"({p.evidence.get('qualified_apps', '?')} qualified apps)")
        for r in rec.rationale:
            lines.append(f"      - {r}")
        lines.append("")

    lines.append("Method: deterministic player projection "
                 "(season rate + recent form + opponent defense + minutes risk)")
    lines.append("Note: Player prop odds not available from feed — "
                 "projections are model-only (no value edge vs market).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stat_to_rate_field(stat: str) -> str:
    return {"goals": "goals_per_90", "sot": "sot_per_90",
            "assists": "assists_per_90", "cards": "cards_per_90"}.get(stat, f"{stat}_per_90")


def _stat_to_starter_rate_field(stat: str) -> str:
    return {"goals": "starter_goals_per_90", "sot": "starter_sot_per_90",
            "assists": "starter_assists_per_90", "cards": "starter_cards_per_90"
            }.get(stat, f"starter_{stat}_per_90")


def _stat_to_raw_field(stat: str) -> str:
    return {"goals": "goals", "sot": "shots_on", "assists": "assists",
            "cards": "cards_total"}.get(stat, stat)


def _build_rationale(
    projection: PlayerProjection,
    line: float,
    side: str,
    prob: float,
) -> List[str]:
    ev = projection.evidence
    lines = []
    lines.append(f"Season rate: {ev['base_rate_per_90']:.2f} {projection.stat}/90 "
                 f"(from {ev.get('qualified_apps', '?')} qualified apps)")
    if ev.get("recent_rate_per_90") is not None:
        lines.append(f"Recent form: {ev['recent_rate_per_90']:.2f} "
                     f"{projection.stat}/90 (last 6)")
    lines.append(f"Team factor: {ev['team_share_factor']:.2f}x | "
                 f"Opponent factor: {ev['opponent_multiplier']:.2f}x")
    lines.append(f"Minutes risk: {projection.minutes_risk} "
                 f"({projection.start_probability:.0%} start rate, "
                 f"~{projection.minutes_projection:.0f} min expected)")
    return lines


def _safe_float(val, default=None):
    try:
        v = float(val)
        return v if v == v else default  # NaN check
    except (TypeError, ValueError):
        return default
