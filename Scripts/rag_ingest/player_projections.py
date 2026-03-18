"""Player prop projection engine.

Deterministic per-player projections for goals, SoT, assists, cards.
Combines player rates with team context, opponent strength, position
relevance, home/away splits, recent form streaks, and minutes risk.
Uses Poisson/NegBin from prob_models.py for probability distributions.

No external dependencies beyond existing RAG infrastructure.
Graceful degradation: returns None when data insufficient.
"""

from __future__ import annotations

from collections import defaultdict
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
    position: str                     # "F" | "M" | "D" | "G"
    fixture: str                      # "Arsenal vs Chelsea"
    projection: float                 # expected value (e.g., 0.42 goals)
    minutes_projection: float         # expected minutes (e.g., 82.0)
    minutes_risk: str                 # "nailed_on" | "likely_starter" | "rotation" | "bench"
    start_probability: float          # from start_rate
    recent_role_trend: str            # "starting" | "benched" | "mixed"
    prob_over_0_5: Optional[float]    # P(X >= 1)
    prob_over_1_5: Optional[float]    # P(X >= 2)
    prob_over_2_5: Optional[float]    # P(X >= 3)
    confidence: str                   # "high" | "medium" | "low"
    evidence: Dict                    # raw numbers for rendering
    opponent_adjustment: float        # multiplier applied
    data_quality: str                 # "full" | "limited" | "insufficient"
    recent_form_values: List[int]     # last N raw stat values (e.g., [1, 0, 2, 0, 1])
    is_home: bool                     # True if playing at home


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
    "season_blend": 0.70,
    "recent_blend": 0.30,
    "team_share_weight": 0.30,
    "opponent_defense_weight": 0.40,

    # Opponent adjustment bounds
    "max_opp_multiplier": 1.40,
    "min_opp_multiplier": 0.65,

    # Home/away adjustment
    "home_boost": 1.06,
    "away_discount": 0.94,

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

# Position relevance multipliers per stat — prevents absurd recs like
# "Defender to score 2+ goals" while boosting natural fits.
POSITION_RELEVANCE = {
    "goals":   {"F": 1.0, "M": 0.85, "D": 0.40, "G": 0.0},
    "sot":     {"F": 1.0, "M": 0.90, "D": 0.45, "G": 0.0},
    "assists": {"F": 0.85, "M": 1.0, "D": 0.55, "G": 0.0},
    "cards":   {"F": 0.80, "M": 1.0, "D": 1.0, "G": 0.15},
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

# Stat display labels
STAT_LABELS = {
    "goals": "Anytime Goalscorer",
    "sot": "Shots on Target",
    "assists": "Assists",
    "cards": "To Be Booked",
}

STAT_EMOJIS = {
    "goals": "\u26bd",
    "sot": "\U0001f3af",
    "assists": "\U0001f3c3",
    "cards": "\U0001f7e8",
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
    fixture_label: str = "",
) -> Optional[PlayerProjection]:
    """Compute expected value for a player stat in a specific fixture.

    Pipeline:
    1. Position gate — filter goalkeepers, apply position relevance
    2. Player's own per-90 rate (starter appearances preferred)
    3. Adjust for expected minutes (starter/rotation/bench)
    4. Home/away venue adjustment
    5. Scale by team context (team output vs league average)
    6. Adjust for opponent defensive quality
    7. Blend season rate with recent form (70/30)
    8. Apply position relevance scaling
    9. Feed to Poisson/NegBin for probability distribution
    """
    pw = PLAYER_PROP_WEIGHTS

    # --- Position gate ---
    position = (player_meta.get("position") or "").strip().upper()
    if not position:
        position = "M"  # default to midfielder if unknown
    # Normalize position variants
    if position in ("GK", "G"):
        position = "G"
    elif position in ("D", "DEF"):
        position = "D"
    elif position in ("M", "MF", "MID"):
        position = "M"
    elif position in ("F", "FW", "ST", "ATT", "A"):
        position = "F"

    pos_relevance = POSITION_RELEVANCE.get(stat, {}).get(position, 0.5)
    if pos_relevance <= 0.0:
        return None  # e.g., goalkeeper for goals/sot/assists

    # --- Data quality gate ---
    role = player_meta.get("minutes_risk") or player_meta.get("role", "bench")
    start_rate = _safe_float(player_meta.get("start_rate"), 0.0)
    total_minutes = int(_safe_float(player_meta.get("total_minutes")
                                     or player_meta.get("minutes"), 0))
    qualified_apps = int(_safe_float(player_meta.get("qualified_appearances")
                                     or player_meta.get("appearances_as_starter"), 0))
    total_apps = int(_safe_float(player_meta.get("appearances_total")
                                 or player_meta.get("total_appearances"), 0))
    recent_trend = player_meta.get("recent_role_trend") or "unknown"

    if total_minutes < pw["min_total_minutes"]:
        return None
    if qualified_apps < pw["min_qualified_apps_limited"]:
        return None

    # Extra gate: if player has been benched recently, downgrade hard
    if recent_trend == "benched" and role not in ("nailed_on", "starter"):
        return None  # skip players trending to bench — unreliable

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

    # --- Home/away venue adjustment ---
    venue_factor = pw["home_boost"] if is_home else pw["away_discount"]

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

    # --- Apply venue + opponent + position relevance ---
    final_projection = blended * opp_multiplier * venue_factor * pos_relevance
    final_projection = max(0.01, final_projection)

    # --- Collect recent raw stat values for display ---
    recent_form_values = _collect_recent_values(recent_fixtures, stat, limit=6)

    # --- Poisson probabilities ---
    variance = _player_stat_variance(recent_fixtures, stat) if recent_fixtures else None
    prob_over_05 = over_prob(final_projection, 0.5, variance)
    prob_over_15 = over_prob(final_projection, 1.5, variance)
    prob_over_25 = (over_prob(final_projection, 2.5, variance)
                    if stat in ("goals", "sot") else None)

    # --- Confidence ---
    confidence = _player_confidence(
        prob_over_05, data_quality, role, qualified_apps, recent_trend, pos_relevance,
    )

    return PlayerProjection(
        player_name=player_meta.get("player_name", "Unknown"),
        player_id=int(_safe_float(player_meta.get("player_id"), 0)),
        team=player_meta.get("team", "Unknown"),
        stat=stat,
        position=position,
        fixture=fixture_label,
        projection=round(final_projection, 3),
        minutes_projection=round(expected_minutes, 1),
        minutes_risk=role,
        start_probability=round(start_rate, 2),
        recent_role_trend=recent_trend,
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
            "venue_factor": round(venue_factor, 3),
            "position_relevance": round(pos_relevance, 2),
            "qualified_apps": qualified_apps,
            "total_apps": total_apps,
        },
        opponent_adjustment=round(opp_multiplier, 3),
        data_quality=data_quality,
        recent_form_values=recent_form_values,
        is_home=is_home,
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

    Leaky defense (high goals_against_pm) -> multiplier > 1.0.
    Stingy defense -> multiplier < 1.0.
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

    High-scoring team -> boost attackers. Low-scoring -> penalize.
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


def _collect_recent_values(
    recent_fixtures: Optional[List[Dict]],
    stat: str,
    limit: int = 6,
) -> List[int]:
    """Collect raw stat values from recent fixtures for display."""
    if not recent_fixtures:
        return []

    stat_field = _stat_to_raw_field(stat)
    values = []
    for f in recent_fixtures[:limit]:
        meta = f if isinstance(f, dict) and "minutes" in f else f.get("meta", f)
        mins = int(_safe_float(meta.get("minutes"), 0))
        if mins >= 30:
            values.append(int(_safe_float(meta.get(stat_field), 0)))
    return values


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
    recent_trend: str = "unknown",
    pos_relevance: float = 1.0,
) -> str:
    """Map projections to confidence levels.

    Model-only (no bookmaker comparison), so confidence is based on:
    model probability strength, data quality, minutes risk, position fit,
    and recent role trend.
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

    # Downgrade for mixed/benched trend
    if recent_trend == "mixed" and conf == "high":
        conf = "medium"

    # Downgrade for weak position fit
    if pos_relevance < 0.50 and conf == "high":
        conf = "medium"

    return conf


# ---------------------------------------------------------------------------
# Rendering — fixture-grouped, Discord-parseable format
# ---------------------------------------------------------------------------

def render_player_prop_answer(
    league: str,
    recommendations: List[PlayerPropRecommendation],
) -> str:
    """Render player prop recommendations grouped by fixture.

    Output format is designed to be parseable by the Discord embed system
    using the === fixture separator pattern.
    """
    if not recommendations:
        return (
            f"Player Prop Predictions by fixture:\n\n"
            f"No player prop recommendations meet confidence thresholds.\n"
            f"This can happen when player data is sparse or all candidates "
            f"are rotation/bench players.\n"
            f"Method: deterministic player projection model"
        )

    # Group recommendations by fixture
    fixture_groups: Dict[str, List[PlayerPropRecommendation]] = defaultdict(list)
    for rec in recommendations:
        fixture_groups[rec.projection.fixture].append(rec)

    lines = [f"Player Prop Predictions by fixture:", ""]

    for fixture, recs in fixture_groups.items():
        lines.append(f"=== {fixture} ===")
        lines.append("")

        # Sort within fixture: high conf first, then model_prob
        recs.sort(key=lambda r: (
            {"high": 3, "medium": 2, "low": 1}.get(r.projection.confidence, 0),
            r.model_prob,
        ), reverse=True)

        for rec in recs:
            p = rec.projection
            stat_emoji = STAT_EMOJIS.get(p.stat, "")
            stat_label = STAT_LABELS.get(p.stat, p.stat.upper())
            pos_label = {"F": "FWD", "M": "MID", "D": "DEF", "G": "GK"}.get(p.position, p.position)
            venue = "HOME" if p.is_home else "AWAY"

            # Player header
            lines.append(f"{stat_emoji} {p.player_name} ({p.team}) [{pos_label}] ({venue})")

            # Recommendation line — mirrors the >>> pattern other workflows use
            lines.append(
                f"  >>> Recommended: {rec.recommended_side.upper()} {rec.recommended_line} "
                f"{stat_label} ({rec.model_prob:.0%} confidence)"
            )

            # Role and minutes context
            role_display = p.minutes_risk.replace("_", " ").title()
            trend_display = p.recent_role_trend.replace("_", " ").title() if p.recent_role_trend != "unknown" else ""
            role_line = f"  Role: {role_display} | Starts: {p.start_probability:.0%} | ~{p.minutes_projection:.0f} min"
            if trend_display:
                role_line += f" | Trend: {trend_display}"
            lines.append(role_line)

            # Projection with probabilities
            prob_parts = [f"P(1+): {p.prob_over_0_5:.0%}"]
            if p.prob_over_1_5 and p.prob_over_1_5 >= 0.05:
                prob_parts.append(f"P(2+): {p.prob_over_1_5:.0%}")
            if p.prob_over_2_5 and p.prob_over_2_5 >= 0.02:
                prob_parts.append(f"P(3+): {p.prob_over_2_5:.0%}")
            lines.append(f"  Projection: {p.projection:.2f} | {' | '.join(prob_parts)}")

            # Recent form streak
            if p.recent_form_values:
                form_str = " ".join(str(v) for v in p.recent_form_values)
                lines.append(f"  Recent: [{form_str}] (last {len(p.recent_form_values)} starts)")

            # Opponent adjustment (only show when meaningful)
            opp_pct = (p.opponent_adjustment - 1) * 100
            if abs(opp_pct) >= 2:
                opp_sign = "+" if opp_pct > 0 else ""
                opp_label = "Leaky defense" if opp_pct > 0 else "Stingy defense"
                lines.append(f"  Opponent: {opp_sign}{opp_pct:.0f}% ({opp_label})")

            # Confidence
            conf_label = p.confidence.upper()
            lines.append(f"  Confidence: {conf_label}")

            # Data quality warning
            if p.data_quality != "full":
                lines.append(
                    f"  [!] Limited data ({p.evidence.get('qualified_apps', '?')} "
                    f"qualified starts)"
                )

            lines.append("")

    lines.append("Method: deterministic player projection "
                 "(starter rates + minutes risk + opponent defense + position + venue)")
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
    rat = []
    rat.append(f"Season rate: {ev['base_rate_per_90']:.2f} {projection.stat}/90 "
               f"(from {ev.get('qualified_apps', '?')} starts)")
    if ev.get("recent_rate_per_90") is not None:
        rat.append(f"Recent form: {ev['recent_rate_per_90']:.2f} "
                   f"{projection.stat}/90")
    rat.append(f"Team {ev['team_share_factor']:.2f}x | "
               f"Opponent {ev['opponent_multiplier']:.2f}x | "
               f"Venue {ev.get('venue_factor', 1.0):.2f}x | "
               f"Position {ev.get('position_relevance', 1.0):.0%}")
    rat.append(f"Minutes: {projection.minutes_risk} "
               f"({projection.start_probability:.0%} start rate, "
               f"~{projection.minutes_projection:.0f} min)")
    return rat


def _safe_float(val, default=None):
    try:
        v = float(val)
        return v if v == v else default  # NaN check
    except (TypeError, ValueError):
        return default
