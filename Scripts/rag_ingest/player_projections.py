"""Player prop projection engine.

Deterministic per-player projections for goals, SoT, assists, cards.
Combines player rates with team context, opponent strength, position
relevance, home/away splits, recent form streaks, minutes risk,
matchup-specific signals (referee, shot volume chain, card induction),
and text-parsed player profile fields (fouls/90, aggression, shots/90).
Uses Poisson/NegBin from prob_models.py for probability distributions.

No external dependencies beyond existing RAG infrastructure.
Graceful degradation: returns None when data insufficient.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from prob_models import over_prob
except ImportError:
    from Scripts.rag_ingest.prob_models import over_prob

# Referee integration — graceful degradation when unavailable
try:
    from referee_data import get_referee_modifier, RefereeModifier
    _REFEREE_AVAILABLE = True
except ImportError:
    try:
        from Scripts.rag_ingest.referee_data import get_referee_modifier, RefereeModifier
        _REFEREE_AVAILABLE = True
    except ImportError:
        _REFEREE_AVAILABLE = False

        @dataclass
        class RefereeModifier:
            multiplier: float = 1.0
            referee_name: Optional[str] = None
            sample_size: int = 0
            confidence: float = 0.0
            strictness_ratio: float = 1.0
            avg_cards_per_match: float = 0.0
            avg_fouls_per_match: float = 0.0
            cards_per_foul: float = 0.0
            source: str = "unavailable"

        def get_referee_modifier(*args, **kwargs) -> RefereeModifier:
            return RefereeModifier()


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
    odds: Optional[float] = None      # bookmaker odds if available
    bookmaker: Optional[str] = None   # bookmaker name
    implied_prob: Optional[float] = None  # bookmaker implied probability
    value_edge: Optional[float] = None    # model_prob - implied_prob


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

    # Stricter gating for player props
    "min_expected_minutes_for_rec": 45,       # skip recommendations if projected < 45 min
    "min_expected_minutes_for_1_5": 70,       # need ~70 min for 1.5+ lines
    "bench_skip_entirely": True,              # skip bench players completely
    "rotation_max_line": 0.5,                 # rotation players only get 0.5 lines
    "low_start_rate_threshold": 0.30,         # below this, treat as bench

    # --- Enhanced matchup weights ---
    # Cards matchup model
    "cards_foul_rate_weight": 0.35,          # foul-rate pathway share
    "cards_base_rate_weight": 0.65,          # historical card rate pathway share
    "cards_referee_blend": 0.25,             # how much referee modifier shifts projection
    "cards_opp_induction_weight": 0.20,      # opponent card-induction adjustment
    "cards_aggression_nudge": 0.10,          # player aggression index nudge
    "cards_foul_card_estimate_weight": 0.30, # fouls/90 * ref cards_per_foul blend

    # Goals shot-chain model
    "goals_shot_chain_weight": 0.30,         # how much shot-chain estimate influences
    "goals_conversion_default": 0.10,        # default conversion rate (goals/shot)
    "goals_xg_team_weight": 0.15,            # team xG context boost

    # SoT shot-volume model
    "sot_shot_volume_weight": 0.30,          # how much raw shot volume influences
    "sot_conversion_default": 0.35,          # default shot-to-SoT conversion
    "sot_possession_weight": 0.10,           # possession context

    # Assists team-attack model
    "assists_team_goals_weight": 0.25,       # team goal output context
    "assists_opp_defense_weight": 0.20,      # opponent defensive weakness

    # Style matchup interaction
    "matchup_weight": 0.20,                  # overall matchup modifier weight

    # Role heuristics (small bounded nudges only)
    "role_penalty_taker_goal_boost": 0.08,
    "role_set_piece_assist_boost": 0.10,
    "role_set_piece_sot_boost": 0.03,
    "role_high_usage_goal_boost": 0.05,
    "role_high_usage_sot_boost": 0.06,
    "role_high_usage_assist_boost": 0.05,
    "role_boost_cap": 1.18,
}

# Position relevance multipliers per stat — prevents absurd recs like
# "Defender to score 2+ goals" while boosting natural fits.
POSITION_RELEVANCE = {
    "goals":   {"F": 1.0, "M": 0.85, "D": 0.40, "G": 0.0},
    "sot":     {"F": 1.0, "M": 0.90, "D": 0.45, "G": 0.0},
    "assists": {"F": 0.85, "M": 1.0, "D": 0.55, "G": 0.0},
    "cards":   {"F": 0.80, "M": 1.0, "D": 1.0, "G": 0.15},
    "fouls":   {"F": 0.70, "M": 1.0, "D": 1.0, "G": 0.05},
}

STAT_LINES = {
    "goals": [0.5, 1.5, 2.5],
    "sot": [0.5, 1.5, 2.5, 3.5],
    "assists": [0.5, 1.5],
    "cards": [0.5, 1.5],
    "fouls": [0.5, 1.5, 2.5, 3.5],
}

LEAGUE_DEFAULTS = {
    "goals": 1.30,
    "sot": 3.80,
    "cards": 1.80,
    "assists": 0.85,
    "fouls": 11.0,      # typical fouls per team per match
    "shots": 12.0,      # typical shots per team per match
    "possession": 50.0,
}

# Stat display labels
STAT_LABELS = {
    "goals": "Anytime Goalscorer",
    "sot": "Shots on Target",
    "assists": "Assists",
    "cards": "To Be Booked",
    "fouls": "To Commit a Foul",
}

STAT_EMOJIS = {
    "goals": "\u26bd",
    "sot": "\U0001f3af",
    "assists": "\U0001f3c3",
    "cards": "\U0001f7e8",
    "fouls": "\U0001f6d1",
}


# ---------------------------------------------------------------------------
# Role heuristics (penalty taker, set-piece creator, high-usage)
# ---------------------------------------------------------------------------

def infer_player_roles(player_meta: Dict, player_text: str = "") -> Dict[str, bool]:
    """Infer player roles from existing metadata.

    Returns dict with: probable_penalty_taker, probable_set_piece, high_usage_creator.
    These are heuristic — based on statistical patterns, not confirmed data.
    """
    roles = {
        "probable_penalty_taker": False,
        "probable_set_piece": False,
        "high_usage_creator": False,
    }

    goals_90 = _safe_float(player_meta.get("starter_goals_per_90") or player_meta.get("goals_per_90"), 0.0)
    shots_90 = _safe_float(player_meta.get("shots_per_90"), 0.0)
    assists_90 = _safe_float(player_meta.get("starter_assists_per_90") or player_meta.get("assists_per_90"), 0.0)
    position = (player_meta.get("position") or "").upper()

    # Penalty taker heuristic: high goals/shot ratio (>0.25) suggests penalty involvement
    # Normal conversion ~0.10-0.15; penalty takers inflate this
    if shots_90 > 0 and goals_90 / shots_90 > 0.25 and goals_90 >= 0.30:
        roles["probable_penalty_taker"] = True

    # Set-piece creator: midfielder/attacker with high assist rate
    if assists_90 >= 0.25 and position in ("M", "F", "MIDFIELDER", "ATTACKER"):
        roles["probable_set_piece"] = True

    # High-usage creator: high combined g+a rate
    if (goals_90 + assists_90) >= 0.60:
        roles["high_usage_creator"] = True

    # Text-based detection: look for "penalty" or "free kick" mentions
    text_lower = player_text.lower() if player_text else ""
    if "penalty" in text_lower or "pen taker" in text_lower:
        roles["probable_penalty_taker"] = True
    if "free kick" in text_lower or "set piece" in text_lower or "set-piece" in text_lower:
        roles["probable_set_piece"] = True

    return roles


# ---------------------------------------------------------------------------
# Player profile text parser
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Player prop odds extraction from event bookmaker data
# ---------------------------------------------------------------------------

# Map our stat types to API-Football market keys
_STAT_TO_MARKET_KEYS = {
    "goals": ["anytime_goalscorer"],
    "sot": ["player_sot"],
    "assists": ["player_assists"],
    "cards": ["player_booked"],
    "fouls": ["player_fouls", "player_fouls_home", "player_fouls_away"],
}


def _normalize_player_name(name: str) -> str:
    """Normalize player name for fuzzy matching across sources."""
    n = name.strip().lower()
    # Strip trailing " -" from API-Football format ("Amara Nallo -")
    n = re.sub(r"\s*-\s*$", "", n)
    # Strip common prefixes/suffixes
    for strip in (".", "'", "-"):
        n = n.replace(strip, "")
    return n.strip()


def extract_player_odds(
    event: Dict,
    player_name: str,
    stat: str,
    line: float = 0.5,
) -> Optional[Dict]:
    """Find bookmaker odds for a specific player prop.

    Searches event bookmaker data for the player's name in the relevant market.
    Returns {odds, bookmaker, implied_prob} or None if not found.

    For anytime goalscorer/assists: matches player name directly (no line needed).
    For SoT/fouls: matches player name + specific over line.
    """
    market_keys = _STAT_TO_MARKET_KEYS.get(stat, [])
    if not market_keys:
        return None

    player_norm = _normalize_player_name(player_name)
    if not player_norm or len(player_norm) < 3:
        return None

    best_match: Optional[Dict] = None

    for bm in event.get("bookmakers", []) or []:
        bm_name = str(bm.get("title") or "")
        for mk in bm.get("markets", []) or []:
            if mk.get("key") not in market_keys:
                continue

            for oc in mk.get("outcomes", []) or []:
                oc_name = _normalize_player_name(str(oc.get("name") or ""))
                price = oc.get("price")
                oc_point = oc.get("point")

                if not oc_name or price is None or price <= 1.0:
                    continue

                # Match player name (substring in either direction)
                if not (player_norm in oc_name or oc_name in player_norm):
                    # Try last-name match (e.g., "fernandes" in "bruno fernandes")
                    player_parts = player_norm.split()
                    oc_parts = oc_name.split()
                    if not any(p in oc_name for p in player_parts if len(p) >= 4):
                        continue

                # For anytime markets (goalscorer, assists, cards/booked): no line needed
                if stat in ("goals", "assists", "cards"):
                    # Anytime = Over 0.5 implied
                    implied = 1.0 / float(price)
                    candidate = {
                        "odds": float(price),
                        "bookmaker": bm_name,
                        "implied_prob": round(implied, 4),
                    }
                    if best_match is None or float(price) > best_match["odds"]:
                        best_match = candidate
                else:
                    # Line-based markets (SoT, fouls): match the specific line
                    if oc_point is not None and abs(float(oc_point) - line) < 0.01:
                        implied = 1.0 / float(price)
                        candidate = {
                            "odds": float(price),
                            "bookmaker": bm_name,
                            "implied_prob": round(implied, 4),
                        }
                        if best_match is None or float(price) > best_match["odds"]:
                            best_match = candidate

    return best_match


def _parse_player_profile_text(text: str) -> Dict[str, float]:
    """Extract stat fields embedded in player profile document text.

    Player profile documents include text-only fields like:
        Fouls/90:1.23  Cards/90:0.34  Shots/90:2.10  SoT/90:1.05
        FormIdx:0.72  AggIdx:0.65  CtrlIdx:0.58
        CardRiskExp:0.41  FoulPressureExp:0.33  DuelWin%:52.3
        Tags: aggressive, rotation-risk

    These are NOT in Chroma metadata, so we parse them from the raw text.
    Returns a dict of parsed float values; missing fields are omitted.
    """
    if not text:
        return {}

    parsed: Dict[str, float] = {}

    # Map of regex pattern -> output key
    patterns = {
        r"Fouls/90:([\d.]+)":            "fouls_per_90",
        r"Cards/90:([\d.]+)":            "cards_per_90_text",
        r"Shots/90:([\d.]+)":            "shots_per_90",
        r"SoT/90:([\d.]+)":             "sot_per_90_text",
        r"FormIdx:([\d.]+)":             "form_index",
        r"AggIdx:([\d.]+)":              "aggression_index",
        r"CtrlIdx:([\d.]+)":             "control_index",
        r"CardRiskExp:([\d.]+)":          "expected_card_risk",
        r"FoulPressureExp:([\d.]+)":      "expected_foul_pressure",
        r"DuelWin%:([\d.]+)":             "duel_win_pct",
    }

    for pattern, key in patterns.items():
        m = re.search(pattern, text)
        if m:
            v = _safe_float(m.group(1))
            if v is not None:
                parsed[key] = v

    # Tags (string, not float)
    m = re.search(r"Tags:\s*(.+?)(?:\n|$)", text)
    if m:
        tags_raw = m.group(1).strip()
        parsed["_tags"] = tags_raw  # type: ignore[assignment]

    return parsed


def _has_tag(parsed_text: Dict, tag: str) -> bool:
    """Check if a player has a specific style tag from parsed text."""
    tags = str(parsed_text.get("_tags", "")).lower()
    return tag.lower() in tags


# ---------------------------------------------------------------------------
# Enhanced stat-specific projection models
# ---------------------------------------------------------------------------

def _cards_matchup_projection(
    base_rate: float,
    minutes_factor: float,
    player_meta: Dict,
    team_meta: Dict,
    opponent_meta: Dict,
    parsed_text: Dict,
    ref_mod: Optional[Any] = None,
) -> Tuple[float, Dict]:
    """Full matchup model for card projection.

    Three pathways blended:
    1. Historical card rate (base_rate) -- the traditional pathway
    2. Foul-rate pathway: fouls_per_90 * referee cards_per_foul
    3. Aggression + opponent card-induction adjustment

    Returns (projection, extra_evidence).
    """
    pw = PLAYER_PROP_WEIGHTS
    extra_ev: Dict[str, Any] = {}

    # --- Pathway 1: historical card rate (already scaled to minutes) ---
    hist_projection = base_rate * minutes_factor

    # --- Parse text-based fields ---
    fouls_per_90 = parsed_text.get("fouls_per_90")
    player_aggression = parsed_text.get("aggression_index")
    expected_card_risk = parsed_text.get("expected_card_risk")

    # Fallback fouls from metadata if text parse fails
    if fouls_per_90 is None:
        fouls_per_90 = _safe_float(player_meta.get("fouls_per_90"))

    # --- Pathway 2: foul-rate * referee cards_per_foul ---
    foul_card_estimate = None
    if fouls_per_90 is not None and fouls_per_90 > 0:
        if ref_mod is not None and ref_mod.cards_per_foul > 0 and ref_mod.source == "profile":
            # Player fouls/90 * referee's card-per-foul rate, scaled to expected minutes
            foul_card_estimate = fouls_per_90 * ref_mod.cards_per_foul * minutes_factor
            extra_ev["foul_card_estimate"] = round(foul_card_estimate, 3)
            extra_ev["player_fouls_per_90"] = round(fouls_per_90, 2)
            extra_ev["ref_cards_per_foul"] = round(ref_mod.cards_per_foul, 3)

    # --- Pathway 3: opponent card-induction ---
    opp_induction = _safe_float(opponent_meta.get("opp_cards_induced_pm"))
    league_avg_cards = LEAGUE_DEFAULTS.get("cards", 1.80)
    opp_induction_multiplier = 1.0
    if opp_induction is not None and opp_induction > 0 and league_avg_cards > 0:
        raw_ratio = opp_induction / league_avg_cards
        # Clamp to [0.75, 1.35] range
        clamped = max(0.75, min(1.35, raw_ratio))
        opp_induction_multiplier = 1.0 + pw["cards_opp_induction_weight"] * (clamped - 1.0)
        extra_ev["opp_card_induction_pm"] = round(opp_induction, 2)
        extra_ev["opp_induction_multiplier"] = round(opp_induction_multiplier, 3)

    # --- Player aggression nudge ---
    aggression_nudge = 0.0
    if player_aggression is not None:
        # aggression_index is typically 0-1; above 0.5 = aggressive player
        aggression_nudge = pw["cards_aggression_nudge"] * (player_aggression - 0.5)
        extra_ev["player_aggression_index"] = round(player_aggression, 2)

    # Expected card risk from profile text (pre-computed risk estimate)
    if expected_card_risk is not None:
        extra_ev["expected_card_risk"] = round(expected_card_risk, 3)

    # --- Blend pathways ---
    if foul_card_estimate is not None:
        # Blend: foul-rate pathway + historical pathway
        blended = (pw["cards_foul_card_estimate_weight"] * foul_card_estimate
                   + (1.0 - pw["cards_foul_card_estimate_weight"]) * hist_projection)
    else:
        blended = hist_projection

    # Apply opponent induction
    blended *= opp_induction_multiplier

    # Apply aggression nudge additively (small push)
    blended += aggression_nudge

    # --- Referee strictness modifier ---
    # Applied multiplicatively, same as team-level cards in rag_cli_v2.py
    if ref_mod is not None and ref_mod.multiplier != 1.0:
        ref_blend_w = pw["cards_referee_blend"] * ref_mod.confidence
        blended = blended * (1.0 + ref_blend_w * (ref_mod.multiplier - 1.0))
        extra_ev["referee_name"] = ref_mod.referee_name or "Unknown"
        extra_ev["referee_strictness"] = round(ref_mod.strictness_ratio, 2)
        extra_ev["referee_modifier"] = round(ref_mod.multiplier, 3)
        extra_ev["referee_confidence"] = round(ref_mod.confidence, 2)

    # Floor at a small positive value
    blended = max(0.01, blended)
    extra_ev["matchup_model"] = "cards_full"

    return blended, extra_ev


def _goals_shot_chain_projection(
    base_rate: float,
    minutes_factor: float,
    player_meta: Dict,
    team_meta: Dict,
    opponent_meta: Dict,
    parsed_text: Dict,
) -> Tuple[float, Dict]:
    """Shot volume chain model for goals projection.

    Chain: shots_per_90 -> conversion_rate -> expected goals
    Adjusted by opponent SoT concession and team xG context.

    Returns (projection, extra_evidence).
    """
    pw = PLAYER_PROP_WEIGHTS
    extra_ev: Dict[str, Any] = {}

    hist_projection = base_rate * minutes_factor

    # --- Shot volume pathway ---
    shots_per_90 = parsed_text.get("shots_per_90")
    if shots_per_90 is None:
        shots_per_90 = _safe_float(player_meta.get("shots_per_90"))

    shot_chain_estimate = None
    if shots_per_90 is not None and shots_per_90 > 0:
        # Conversion rate: goals / shots. Derive from player's own rates if possible.
        goals_per_90 = _safe_float(player_meta.get("goals_per_90")) or base_rate
        if goals_per_90 > 0 and shots_per_90 > 0:
            conversion = goals_per_90 / shots_per_90
        else:
            conversion = pw["goals_conversion_default"]
        # Clamp conversion to reasonable range [0.02, 0.35]
        conversion = max(0.02, min(0.35, conversion))

        # Opponent SoT concession: if opponent lets through many SoT,
        # the player's shots are more likely to reach the keeper / go in
        opp_sot_against = _safe_float(opponent_meta.get("sot_against_pm"))
        league_avg_sot = LEAGUE_DEFAULTS.get("sot", 3.80)
        shot_volume_adjust = 1.0
        if opp_sot_against is not None and league_avg_sot > 0:
            # Opponent concedes more SoT than average -> player gets more quality chances
            shot_volume_adjust = 0.5 + 0.5 * (opp_sot_against / league_avg_sot)
            shot_volume_adjust = max(0.75, min(1.30, shot_volume_adjust))

        shot_chain_estimate = (shots_per_90 * shot_volume_adjust
                               * conversion * minutes_factor)
        extra_ev["shots_per_90"] = round(shots_per_90, 2)
        extra_ev["conversion_rate"] = round(conversion, 3)
        extra_ev["opp_sot_against_pm"] = round(opp_sot_against, 2) if opp_sot_against else None
        extra_ev["shot_volume_adjust"] = round(shot_volume_adjust, 3)
        extra_ev["shot_chain_estimate"] = round(shot_chain_estimate, 3)

    # --- Team xG context ---
    # High-xG team = more quality chances for its forwards
    xg_boost = 1.0
    xg_home = _safe_float(team_meta.get("xg_home_pm"))
    xg_away = _safe_float(team_meta.get("xg_away_pm"))
    team_xg = None
    if xg_home is not None and xg_away is not None:
        team_xg = (xg_home + xg_away) / 2.0
    if team_xg is not None and team_xg > 0:
        league_avg_goals = LEAGUE_DEFAULTS.get("goals", 1.30)
        xg_ratio = team_xg / league_avg_goals
        xg_boost = 1.0 + pw["goals_xg_team_weight"] * (xg_ratio - 1.0)
        xg_boost = max(0.85, min(1.20, xg_boost))
        extra_ev["team_xg_pm"] = round(team_xg, 2)
        extra_ev["xg_boost"] = round(xg_boost, 3)

    # --- Blend shot-chain with historical ---
    if shot_chain_estimate is not None:
        blended = ((1.0 - pw["goals_shot_chain_weight"]) * hist_projection
                   + pw["goals_shot_chain_weight"] * shot_chain_estimate)
    else:
        blended = hist_projection

    # Apply xG context boost
    blended *= xg_boost

    blended = max(0.01, blended)
    extra_ev["matchup_model"] = "goals_shot_chain"

    return blended, extra_ev


def _sot_shot_volume_projection(
    base_rate: float,
    minutes_factor: float,
    player_meta: Dict,
    team_meta: Dict,
    opponent_meta: Dict,
    parsed_text: Dict,
) -> Tuple[float, Dict]:
    """Shot volume model for SoT projection.

    Chain: shots_per_90 -> SoT conversion -> expected SoT
    Adjusted by opponent shot concession and team possession context.

    Returns (projection, extra_evidence).
    """
    pw = PLAYER_PROP_WEIGHTS
    extra_ev: Dict[str, Any] = {}

    hist_projection = base_rate * minutes_factor

    # --- Shot volume -> SoT conversion ---
    shots_per_90 = parsed_text.get("shots_per_90")
    if shots_per_90 is None:
        shots_per_90 = _safe_float(player_meta.get("shots_per_90"))

    shot_volume_estimate = None
    if shots_per_90 is not None and shots_per_90 > 0:
        # SoT conversion: SoT / total shots
        sot_per_90 = _safe_float(player_meta.get("sot_per_90")) or base_rate
        if sot_per_90 > 0 and shots_per_90 > 0:
            sot_conversion = sot_per_90 / shots_per_90
        else:
            sot_conversion = pw["sot_conversion_default"]
        sot_conversion = max(0.15, min(0.70, sot_conversion))

        # Opponent shot concession: how many SoT they allow
        opp_sot_against = _safe_float(opponent_meta.get("sot_against_pm"))
        league_avg_sot = LEAGUE_DEFAULTS.get("sot", 3.80)
        opp_sot_factor = 1.0
        if opp_sot_against is not None and league_avg_sot > 0:
            opp_sot_factor = 0.5 + 0.5 * (opp_sot_against / league_avg_sot)
            opp_sot_factor = max(0.75, min(1.30, opp_sot_factor))

        shot_volume_estimate = (shots_per_90 * opp_sot_factor
                                * sot_conversion * minutes_factor)
        extra_ev["shots_per_90"] = round(shots_per_90, 2)
        extra_ev["sot_conversion"] = round(sot_conversion, 3)
        extra_ev["opp_sot_against_pm"] = round(opp_sot_against, 2) if opp_sot_against else None
        extra_ev["opp_sot_factor"] = round(opp_sot_factor, 3)
        extra_ev["shot_volume_estimate"] = round(shot_volume_estimate, 3)

    # --- Possession context ---
    # More possession = more time in attacking positions = more shots
    possession_boost = 1.0
    team_possession = _safe_float(team_meta.get("possession"))
    if team_possession is not None and team_possession > 0:
        possession_ratio = team_possession / LEAGUE_DEFAULTS.get("possession", 50.0)
        possession_boost = 1.0 + pw["sot_possession_weight"] * (possession_ratio - 1.0)
        possession_boost = max(0.90, min(1.12, possession_boost))
        extra_ev["team_possession"] = round(team_possession, 1)
        extra_ev["possession_boost"] = round(possession_boost, 3)

    # --- Blend ---
    if shot_volume_estimate is not None:
        blended = ((1.0 - pw["sot_shot_volume_weight"]) * hist_projection
                   + pw["sot_shot_volume_weight"] * shot_volume_estimate)
    else:
        blended = hist_projection

    blended *= possession_boost
    blended = max(0.01, blended)
    extra_ev["matchup_model"] = "sot_shot_volume"

    return blended, extra_ev


def _assists_team_attack_projection(
    base_rate: float,
    minutes_factor: float,
    player_meta: Dict,
    team_meta: Dict,
    opponent_meta: Dict,
    parsed_text: Dict,
) -> Tuple[float, Dict]:
    """Team attack model for assists projection.

    Assists only happen when goals are scored, so team attack output
    and opponent defensive weakness are primary context signals.

    Returns (projection, extra_evidence).
    """
    pw = PLAYER_PROP_WEIGHTS
    extra_ev: Dict[str, Any] = {}

    hist_projection = base_rate * minutes_factor

    # --- Team goal output context ---
    # More goals from teammates = more assist opportunities
    team_goals_for = _safe_float(team_meta.get("goals_for_pm"))
    league_avg_goals = LEAGUE_DEFAULTS.get("goals", 1.30)
    team_attack_factor = 1.0
    if team_goals_for is not None and team_goals_for > 0 and league_avg_goals > 0:
        team_attack_ratio = team_goals_for / league_avg_goals
        team_attack_factor = 1.0 + pw["assists_team_goals_weight"] * (team_attack_ratio - 1.0)
        team_attack_factor = max(0.80, min(1.25, team_attack_factor))
        extra_ev["team_goals_for_pm"] = round(team_goals_for, 2)
        extra_ev["team_attack_factor"] = round(team_attack_factor, 3)

    # --- Opponent defensive weakness ---
    # Opponent concedes more goals = more assist chances
    opp_goals_against = _safe_float(opponent_meta.get("goals_against_pm"))
    opp_defense_factor = 1.0
    if opp_goals_against is not None and opp_goals_against > 0 and league_avg_goals > 0:
        opp_defense_ratio = opp_goals_against / league_avg_goals
        opp_defense_factor = 1.0 + pw["assists_opp_defense_weight"] * (opp_defense_ratio - 1.0)
        opp_defense_factor = max(0.80, min(1.25, opp_defense_factor))
        extra_ev["opp_goals_against_pm"] = round(opp_goals_against, 2)
        extra_ev["opp_defense_factor"] = round(opp_defense_factor, 3)

    # --- Team xG as secondary signal ---
    xg_home = _safe_float(team_meta.get("xg_home_pm"))
    xg_away = _safe_float(team_meta.get("xg_away_pm"))
    if xg_home is not None and xg_away is not None:
        team_xg = (xg_home + xg_away) / 2.0
        extra_ev["team_xg_pm"] = round(team_xg, 2)

    # --- Control index as chance creation proxy ---
    control_index = parsed_text.get("control_index")
    if control_index is None:
        control_index = _safe_float(player_meta.get("control_index"))
    control_boost = 1.0
    if control_index is not None:
        # Higher control = better chance creation = more assists
        control_boost = 1.0 + 0.08 * (control_index - 0.5)
        control_boost = max(0.95, min(1.08, control_boost))
        extra_ev["player_control_index"] = round(control_index, 2)

    # --- Combine ---
    blended = hist_projection * team_attack_factor * opp_defense_factor * control_boost
    blended = max(0.01, blended)
    extra_ev["matchup_model"] = "assists_team_attack"

    return blended, extra_ev


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
    player_text: str = "",
    ref_mod: Optional[Any] = None,
) -> Optional[PlayerProjection]:
    """Compute expected value for a player stat in a specific fixture.

    Pipeline:
    1. Position gate -- filter goalkeepers, apply position relevance
    2. Player's own per-90 rate (starter appearances preferred)
    3. Adjust for expected minutes (starter/rotation/bench)
    4. Home/away venue adjustment
    5. Parse player profile text for extra matchup fields
    6. Stat-specific enhanced matchup model:
       - cards: foul-rate * ref cards_per_foul + aggression + opp induction + referee
       - goals: shot volume * conversion * opp SoT concession + team xG
       - sot: shot volume * SoT conversion * opp shot concession + possession
       - assists: team attack output * opp defensive weakness + control
    7. Blend season rate with recent form (70/30)
    8. Apply position relevance scaling
    9. Feed to Poisson/NegBin for probability distribution

    New parameters:
        player_text: raw text from the player profile Chroma document
        ref_mod: RefereeModifier from referee_data.py (for card projections)
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
        return None  # skip players trending to bench -- unreliable

    # Stricter gating: skip bench players entirely if configured
    if pw.get("bench_skip_entirely", True) and role in ("bench",):
        return None

    # Stricter gating: low start_rate treated as bench
    if start_rate < pw.get("low_start_rate_threshold", 0.30) and role not in ("nailed_on", "starter"):
        return None

    data_quality = "full" if qualified_apps >= pw["min_qualified_apps"] else "limited"

    # --- Player's own per-90 rate (prefer starter-only rates) ---
    starter_rate_field = _stat_to_starter_rate_field(stat)
    base_rate = _safe_float(player_meta.get(starter_rate_field))

    # Fallback to overall rate if starter rate unavailable
    if base_rate is None or base_rate < 0:
        rate_field = _stat_to_rate_field(stat)
        base_rate = _safe_float(player_meta.get(rate_field))

    # Fallback to text-parsed rate (for stats not in Chroma metadata like fouls)
    if base_rate is None or base_rate < 0:
        parsed = _parse_player_profile_text(player_text) if player_text else {}
        text_rate_map = {
            "fouls": "fouls_per_90",
            "cards": "cards_per_90",
            "sot": "sot_per_90",
            "goals": "goals_per_90",
        }
        text_field = text_rate_map.get(stat)
        if text_field:
            base_rate = _safe_float(parsed.get(text_field))

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

    # --- Minutes feasibility check ---
    min_minutes_for_rec = pw.get("min_expected_minutes_for_rec", 45)
    if expected_minutes < min_minutes_for_rec:
        return None  # not enough expected playing time for a meaningful prop

    # --- Scale rate to expected minutes ---
    minutes_factor = expected_minutes / 90.0

    # --- Home/away venue adjustment ---
    venue_factor = pw["home_boost"] if is_home else pw["away_discount"]

    # --- Parse player profile text for extra matchup fields ---
    parsed_text = _parse_player_profile_text(player_text)
    inferred_roles = infer_player_roles(player_meta, player_text)

    # --- Recent form blend (compute recent rate first, used by all pathways) ---
    recent_rate = _compute_recent_rate(recent_fixtures, stat, min_minutes=30)

    # --- Stat-specific enhanced matchup projection ---
    extra_evidence: Dict[str, Any] = {}

    if stat == "cards":
        # Enhanced cards: foul-rate pathway + referee + opponent induction
        season_proj, extra_evidence = _cards_matchup_projection(
            base_rate, minutes_factor, player_meta, team_meta,
            opponent_meta, parsed_text, ref_mod,
        )
        if recent_rate is not None:
            recent_proj, _ = _cards_matchup_projection(
                recent_rate, minutes_factor, player_meta, team_meta,
                opponent_meta, parsed_text, ref_mod,
            )
            blended = pw["season_blend"] * season_proj + pw["recent_blend"] * recent_proj
        else:
            blended = season_proj

    elif stat == "goals":
        # Enhanced goals: shot volume chain + xG context
        season_proj, extra_evidence = _goals_shot_chain_projection(
            base_rate, minutes_factor, player_meta, team_meta,
            opponent_meta, parsed_text,
        )
        if recent_rate is not None:
            recent_proj, _ = _goals_shot_chain_projection(
                recent_rate, minutes_factor, player_meta, team_meta,
                opponent_meta, parsed_text,
            )
            blended = pw["season_blend"] * season_proj + pw["recent_blend"] * recent_proj
        else:
            blended = season_proj

    elif stat == "sot":
        # Enhanced SoT: shot volume + possession context
        season_proj, extra_evidence = _sot_shot_volume_projection(
            base_rate, minutes_factor, player_meta, team_meta,
            opponent_meta, parsed_text,
        )
        if recent_rate is not None:
            recent_proj, _ = _sot_shot_volume_projection(
                recent_rate, minutes_factor, player_meta, team_meta,
                opponent_meta, parsed_text,
            )
            blended = pw["season_blend"] * season_proj + pw["recent_blend"] * recent_proj
        else:
            blended = season_proj

    elif stat == "assists":
        # Enhanced assists: team attack model
        season_proj, extra_evidence = _assists_team_attack_projection(
            base_rate, minutes_factor, player_meta, team_meta,
            opponent_meta, parsed_text,
        )
        if recent_rate is not None:
            recent_proj, _ = _assists_team_attack_projection(
                recent_rate, minutes_factor, player_meta, team_meta,
                opponent_meta, parsed_text,
            )
            blended = pw["season_blend"] * season_proj + pw["recent_blend"] * recent_proj
        else:
            blended = season_proj

    else:
        # Fallback: legacy simple pipeline for unknown stats
        season_projection = base_rate * minutes_factor
        if recent_rate is not None:
            recent_projection = recent_rate * minutes_factor
            blended = (pw["season_blend"] * season_projection
                       + pw["recent_blend"] * recent_projection)
        else:
            blended = season_projection
        # Legacy team/opponent adjustments for unknown stats
        team_share = _team_share_adjustment(team_meta, stat)
        opp_multiplier = _opponent_adjustment(opponent_meta, stat)
        blended *= team_share * opp_multiplier

    # --- Apply venue + position relevance ---
    # NOTE: For enhanced models, opponent adjustment is built into the
    # stat-specific model. For the legacy path, it is applied above.
    if stat in ("cards", "goals", "sot", "assists"):
        final_projection = blended * venue_factor * pos_relevance
    else:
        final_projection = blended * venue_factor * pos_relevance

    final_projection = max(0.01, final_projection)

    # --- Style matchup interaction ---
    # Cross-reference player style with opponent team profile
    style_mod = _style_matchup_modifier(parsed_text, opponent_meta, stat)
    if abs(style_mod - 1.0) >= 0.01:
        final_projection *= style_mod
        extra_evidence["style_matchup_modifier"] = round(style_mod, 3)

    # --- Role heuristics (bounded, league-only, no hard overrides) ---
    role_multiplier = 1.0
    if stat == "goals":
        if inferred_roles.get("probable_penalty_taker"):
            role_multiplier += pw.get("role_penalty_taker_goal_boost", 0.0)
        if inferred_roles.get("high_usage_creator"):
            role_multiplier += pw.get("role_high_usage_goal_boost", 0.0)
    elif stat == "assists":
        if inferred_roles.get("probable_set_piece"):
            role_multiplier += pw.get("role_set_piece_assist_boost", 0.0)
        if inferred_roles.get("high_usage_creator"):
            role_multiplier += pw.get("role_high_usage_assist_boost", 0.0)
    elif stat == "sot":
        if inferred_roles.get("probable_set_piece"):
            role_multiplier += pw.get("role_set_piece_sot_boost", 0.0)
        if inferred_roles.get("high_usage_creator"):
            role_multiplier += pw.get("role_high_usage_sot_boost", 0.0)
    role_multiplier = min(role_multiplier, pw.get("role_boost_cap", 1.18))
    if role_multiplier != 1.0:
        final_projection *= role_multiplier
        extra_evidence["role_multiplier"] = round(role_multiplier, 3)

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

    # --- Build evidence dict ---
    # Compute opponent_multiplier for display (even though enhanced models
    # internalize it, we report the equivalent for transparency)
    opp_multiplier = _opponent_adjustment(opponent_meta, stat)
    team_share = _team_share_adjustment(team_meta, stat)

    evidence = {
        "base_rate_per_90": round(base_rate, 3),
        "expected_minutes": round(expected_minutes, 1),
        "season_projection": round(base_rate * minutes_factor, 3),
        "recent_rate_per_90": round(recent_rate, 3) if recent_rate is not None else None,
        "team_share_factor": round(team_share, 3),
        "opponent_multiplier": round(opp_multiplier, 3),
        "venue_factor": round(venue_factor, 3),
        "position_relevance": round(pos_relevance, 2),
        "qualified_apps": qualified_apps,
        "total_apps": total_apps,
        "probable_penalty_taker": inferred_roles.get("probable_penalty_taker", False),
        "probable_set_piece": inferred_roles.get("probable_set_piece", False),
        "high_usage_creator": inferred_roles.get("high_usage_creator", False),
    }
    # Merge extra evidence from enhanced models
    evidence.update(extra_evidence)

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
        evidence=evidence,
        opponent_adjustment=round(opp_multiplier, 3),
        data_quality=data_quality,
        recent_form_values=recent_form_values,
        is_home=is_home,
    )


def recommend_player_prop(
    projection: PlayerProjection,
    event: Optional[Dict] = None,
) -> Optional[PlayerPropRecommendation]:
    """Recommend a player prop — always OVER-oriented.

    Football player props are almost exclusively "anytime" markets:
    - Anytime Goalscorer = Over 0.5 goals
    - To Be Booked = Over 0.5 cards
    - 1+ Shot on Target = Over 0.5 SoT
    - To Get an Assist = Over 0.5 assists

    For SoT where rates are higher, also consider Over 1.5.
    Returns None if the player doesn't meet minimum start rate or
    probability is too low to be a useful recommendation.
    """
    if projection is None:
        return None

    pw = PLAYER_PROP_WEIGHTS
    if projection.start_probability < pw["min_start_rate_for_rec"]:
        return None

    lam = projection.projection
    p_over_05 = projection.prob_over_0_5 or over_prob(lam, 0.5)

    # Minimum probability to even recommend — below this the player
    # has essentially no chance and showing them is noise.
    MIN_PROB = {
        "goals": 0.08,   # ~8% = roughly 1 in 12 chance
        "cards": 0.10,   # ~10%
        "sot": 0.15,     # shots on target more common
        "assists": 0.06, # assists are rare
        "fouls": 0.30,   # fouls are common — need higher bar
    }
    min_p = MIN_PROB.get(projection.stat, 0.08)
    if p_over_05 < min_p:
        return None

    # Default: Over 0.5 (anytime)
    best_line = 0.5
    best_side = "over"
    best_prob = p_over_05

    # Line feasibility: rotation players capped at 0.5 lines
    rotation_max = pw.get("rotation_max_line", 0.5)
    is_rotation = projection.minutes_risk in ("rotation",)

    # Minutes-based line cap: need enough expected minutes for higher lines
    min_min_1_5 = pw.get("min_expected_minutes_for_1_5", 70)
    can_rec_1_5 = projection.minutes_projection >= min_min_1_5 and not is_rotation

    # For SoT: if P(2+) is strong enough, recommend Over 1.5 instead
    if projection.stat == "sot" and projection.prob_over_1_5 and can_rec_1_5:
        p_15 = projection.prob_over_1_5
        if p_15 >= 0.40:
            best_line = 1.5
            best_prob = p_15

    # For fouls: players foul often, so recommend higher lines
    # Pick the highest line where probability is still >= 50%
    if projection.stat == "fouls":
        for line in [3.5, 2.5, 1.5, 0.5]:
            p = over_prob(lam, line)
            if p >= 0.50:
                best_line = line
                best_prob = p
                break

    rationale = _build_rationale(projection, best_line, best_side, best_prob)

    # Look up bookmaker odds for this player prop
    prop_odds = None
    prop_bookmaker = None
    prop_implied = None
    prop_edge = None

    if event is not None:
        odds_match = extract_player_odds(
            event, projection.player_name, projection.stat, best_line,
        )
        if odds_match:
            prop_odds = odds_match["odds"]
            prop_bookmaker = odds_match["bookmaker"]
            prop_implied = odds_match["implied_prob"]
            prop_edge = round(best_prob - prop_implied, 4)

    return PlayerPropRecommendation(
        projection=projection,
        recommended_line=best_line,
        recommended_side=best_side,
        model_prob=round(best_prob, 4),
        rationale=rationale,
        odds=prop_odds,
        bookmaker=prop_bookmaker,
        implied_prob=prop_implied,
        value_edge=prop_edge,
    )


# ---------------------------------------------------------------------------
# Opponent & team adjustments (legacy, still used for display and fallback)
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
# Style matchup interaction
# ---------------------------------------------------------------------------

def _style_matchup_modifier(
    player_parsed: Dict,
    opponent_meta: Dict,
    stat: str,
) -> float:
    """Compute a matchup modifier based on how a player's style interacts
    with the opponent's playing profile.

    Returns a multiplier (1.0 = neutral, >1 = favorable matchup, <1 = unfavorable).

    Key interactions:
    - Fouling player vs transition/counter-attacking team → more fouls
    - Aggressive player vs aggressive opponent → more cards
    - High-shot player vs team that concedes shots → more SoT
    - Striker vs low-possession opponent (they'll sit deep, less space) → fewer goals
    - Creative player vs high-press opponent → more chance creation (assists)
    """
    if not player_parsed or not opponent_meta:
        return 1.0

    modifier = 1.0
    pw = PLAYER_PROP_WEIGHTS
    weight = pw.get("matchup_weight", 0.20)

    # Player attributes
    p_aggression = _safe_float(player_parsed.get("aggression_index"))
    p_fouls = _safe_float(player_parsed.get("fouls_per_90"))
    p_control = _safe_float(player_parsed.get("control_index"))
    p_duel_win = _safe_float(player_parsed.get("duel_win_pct"))

    # Opponent attributes
    opp_possession = _safe_float(opponent_meta.get("possession"))
    opp_aggression = _safe_float(opponent_meta.get("aggression_index_norm"))
    opp_control = _safe_float(opponent_meta.get("control_index"))
    opp_dominance = _safe_float(opponent_meta.get("dominance_index"))
    opp_fouls = _safe_float(opponent_meta.get("fouls_per_90_team"))
    opp_cards_induced = _safe_float(opponent_meta.get("opp_cards_induced_pm"))

    if stat == "fouls":
        # Fouling players foul MORE against teams that play in transition
        # Transition teams have lower possession but higher counter-attack threat
        if opp_possession is not None and p_fouls is not None:
            # Low-possession opponent = transition style = more tactical fouling needed
            if opp_possession < 0.47:
                modifier += weight * 0.30  # transition opponent → more fouls
            elif opp_possession > 0.55:
                modifier -= weight * 0.15  # possession-heavy opponent → fewer fouls needed

        # Aggressive opponent draws more fouls from defenders trying to stop them
        if opp_aggression is not None and opp_aggression > 0.55:
            modifier += weight * 0.20

        # High duel involvement → more foul opportunities
        if opp_dominance is not None and opp_dominance > 0.50:
            modifier += weight * 0.15  # dominant opponent forces more defensive actions

    elif stat == "cards":
        # Aggressive player vs card-inducing opponent → boosted
        if p_aggression is not None and opp_cards_induced is not None:
            if p_aggression > 0.55 and opp_cards_induced > 2.0:
                modifier += weight * 0.35  # aggressive player + card-heavy opponent

        # Low control player vs high-tempo opponent → more rash challenges
        if p_control is not None and opp_control is not None:
            if p_control < 0.60 and opp_control > 0.65:
                modifier += weight * 0.20  # outclassed player fouls more

        # Opponent plays with high fouls → physical game → more cards for everyone
        if opp_fouls is not None and opp_fouls > 12.0:
            modifier += weight * 0.15

    elif stat == "goals":
        # Striker vs low-block team (high possession opponent = they have the ball)
        if opp_possession is not None:
            if opp_possession > 0.55:
                modifier -= weight * 0.20  # less time on the ball → fewer goals
            elif opp_possession < 0.45:
                modifier += weight * 0.15  # opponent sits deep, counter opportunities

        # Weak defensive opponent (low control) → easier to score
        if opp_control is not None and opp_control < 0.55:
            modifier += weight * 0.20

    elif stat == "sot":
        # High-shot player vs team that concedes shots
        opp_sot_against = _safe_float(opponent_meta.get("sot_against_pm"))
        if opp_sot_against is not None and opp_sot_against > 4.5:
            modifier += weight * 0.25  # leaky defense → more shots get through

        # Dominant player vs weaker opponent → more shooting opportunities
        if opp_dominance is not None and opp_dominance < 0.40:
            modifier += weight * 0.15

    elif stat == "assists":
        # Creative player vs high-press opponent → spaces open behind the press
        if p_control is not None and opp_aggression is not None:
            if p_control > 0.65 and opp_aggression > 0.55:
                modifier += weight * 0.25  # composed passer vs aggressive presser

        # Opponent concedes a lot → more assists possible
        opp_goals_against = _safe_float(opponent_meta.get("goals_against_pm"))
        if opp_goals_against is not None and opp_goals_against > 1.5:
            modifier += weight * 0.20

    # Clamp to reasonable range
    return max(0.80, min(1.25, modifier))


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
# Rendering -- fixture-grouped, Discord-parseable format
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

        # Sort within fixture: highest probability first (best picks on top)
        recs.sort(key=lambda r: (
            r.model_prob,
            r.projection.projection,
        ), reverse=True)

        for rec in recs:
            p = rec.projection
            stat_emoji = STAT_EMOJIS.get(p.stat, "")
            stat_label = STAT_LABELS.get(p.stat, p.stat.upper())
            pos_label = {"F": "FWD", "M": "MID", "D": "DEF", "G": "GK"}.get(p.position, p.position)
            venue = "HOME" if p.is_home else "AWAY"

            # Player header
            lines.append(f"{stat_emoji} {p.player_name} ({p.team}) [{pos_label}] ({venue})")

            # Recommendation line — anytime-style display with odds when available
            if rec.recommended_line == 0.5:
                pick_text = f"  >>> {stat_label} — {rec.model_prob:.0%} chance"
            else:
                pick_text = f"  >>> Over {rec.recommended_line} {stat_label} — {rec.model_prob:.0%} chance"

            if rec.odds:
                pick_text += f" @ {rec.odds:.2f} ({rec.bookmaker})"
            lines.append(pick_text)

            # Odds comparison line when available
            if rec.odds and rec.implied_prob and rec.value_edge is not None:
                edge_sign = "+" if rec.value_edge > 0 else ""
                lines.append(
                    f"  Model: {rec.model_prob:.0%} vs Books: {rec.implied_prob:.0%} "
                    f"| Edge: {edge_sign}{rec.value_edge:.0%}"
                )

            # Confidence badge
            lines.append(f"  Confidence: {p.confidence.upper()}")

            # Role and minutes context
            role_display = p.minutes_risk.replace("_", " ").title()
            trend_display = p.recent_role_trend.replace("_", " ").title() if p.recent_role_trend != "unknown" else ""
            role_line = f"  Role: {role_display} | Starts: {p.start_probability:.0%} | ~{p.minutes_projection:.0f} min"
            if trend_display:
                role_line += f" | Trend: {trend_display}"
            lines.append(role_line)

            # Probability breakdown
            prob_parts = [f"P(1+): {p.prob_over_0_5:.0%}"]
            if p.prob_over_1_5 and p.prob_over_1_5 >= 0.05:
                prob_parts.append(f"P(2+): {p.prob_over_1_5:.0%}")
            if p.prob_over_2_5 and p.prob_over_2_5 >= 0.02:
                prob_parts.append(f"P(3+): {p.prob_over_2_5:.0%}")
            lines.append(f"  {' | '.join(prob_parts)}")

            # Recent form streak
            if p.recent_form_values:
                form_str = " ".join(str(v) for v in p.recent_form_values)
                lines.append(f"  Recent: [{form_str}] (last {len(p.recent_form_values)} starts)")

            # Enhanced matchup evidence
            ev = p.evidence
            matchup_model = ev.get("matchup_model", "")

            if matchup_model == "cards_full":
                _render_cards_evidence(lines, ev)
            elif matchup_model == "goals_shot_chain":
                _render_goals_evidence(lines, ev)
            elif matchup_model == "sot_shot_volume":
                _render_sot_evidence(lines, ev)
            elif matchup_model == "assists_team_attack":
                _render_assists_evidence(lines, ev)
            else:
                # Legacy opponent adjustment display
                opp_pct = (p.opponent_adjustment - 1) * 100
                if abs(opp_pct) >= 2:
                    opp_sign = "+" if opp_pct > 0 else ""
                    opp_label = "Leaky defense" if opp_pct > 0 else "Stingy defense"
                    lines.append(f"  Opponent: {opp_sign}{opp_pct:.0f}% ({opp_label})")

            # Style matchup modifier evidence
            style_mod = ev.get("style_matchup_modifier")
            if style_mod is not None and abs(style_mod - 1.0) >= 0.02:
                pct = (style_mod - 1.0) * 100
                if pct > 0:
                    lines.append(f"  Matchup: +{pct:.0f}% (favorable style interaction)")
                else:
                    lines.append(f"  Matchup: {pct:.0f}% (unfavorable style interaction)")

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
                 "(matchup model + starter rates + minutes risk + opponent + position + venue)")
    return "\n".join(lines)


def _render_cards_evidence(lines: List[str], ev: Dict) -> None:
    """Render enhanced cards matchup evidence."""
    parts = []

    # Foul-rate pathway
    fouls_90 = ev.get("player_fouls_per_90")
    ref_cpf = ev.get("ref_cards_per_foul")
    if fouls_90 is not None and ref_cpf is not None:
        foul_est = ev.get("foul_card_estimate", 0)
        parts.append(f"Fouls/90: {fouls_90:.1f} x Ref CPF: {ref_cpf:.2f} = {foul_est:.2f}")

    # Opponent induction
    opp_ind = ev.get("opp_card_induction_pm")
    if opp_ind is not None:
        opp_mult = ev.get("opp_induction_multiplier", 1.0)
        opp_label = "high-induction" if opp_mult > 1.02 else ("low-induction" if opp_mult < 0.98 else "neutral")
        parts.append(f"Opp cards induced: {opp_ind:.1f}/m ({opp_label})")

    # Player aggression
    agg = ev.get("player_aggression_index")
    if agg is not None:
        agg_label = "aggressive" if agg > 0.60 else ("calm" if agg < 0.40 else "moderate")
        parts.append(f"Aggression: {agg:.2f} ({agg_label})")

    # Referee info
    ref_name = ev.get("referee_name")
    ref_strict = ev.get("referee_strictness")
    if ref_name and ref_strict:
        ref_label = "strict" if ref_strict > 1.05 else ("lenient" if ref_strict < 0.95 else "average")
        ref_conf = ev.get("referee_confidence", 0)
        parts.append(f"Ref: {ref_name} ({ref_label}, {ref_strict:.2f}x, conf {ref_conf:.0%})")

    if parts:
        lines.append(f"  Matchup: {' | '.join(parts)}")


def _render_goals_evidence(lines: List[str], ev: Dict) -> None:
    """Render enhanced goals shot-chain evidence."""
    parts = []

    shots = ev.get("shots_per_90")
    conv = ev.get("conversion_rate")
    if shots is not None and conv is not None:
        parts.append(f"Shots/90: {shots:.1f} x Conv: {conv:.0%}")

    chain_est = ev.get("shot_chain_estimate")
    if chain_est is not None:
        parts.append(f"Shot-chain est: {chain_est:.2f}")

    opp_sot = ev.get("opp_sot_against_pm")
    if opp_sot is not None:
        adj = ev.get("shot_volume_adjust", 1.0)
        adj_label = "leaky" if adj > 1.02 else ("tight" if adj < 0.98 else "average")
        parts.append(f"Opp SoT against: {opp_sot:.1f} ({adj_label})")

    xg = ev.get("team_xg_pm")
    xg_boost = ev.get("xg_boost")
    if xg is not None and xg_boost is not None:
        parts.append(f"Team xG: {xg:.2f} ({xg_boost:.2f}x)")

    if parts:
        lines.append(f"  Matchup: {' | '.join(parts)}")


def _render_sot_evidence(lines: List[str], ev: Dict) -> None:
    """Render enhanced SoT shot-volume evidence."""
    parts = []

    shots = ev.get("shots_per_90")
    conv = ev.get("sot_conversion")
    if shots is not None and conv is not None:
        parts.append(f"Shots/90: {shots:.1f} x SoT conv: {conv:.0%}")

    vol_est = ev.get("shot_volume_estimate")
    if vol_est is not None:
        parts.append(f"Vol est: {vol_est:.2f}")

    opp_sot = ev.get("opp_sot_against_pm")
    if opp_sot is not None:
        factor = ev.get("opp_sot_factor", 1.0)
        label = "leaky" if factor > 1.02 else ("tight" if factor < 0.98 else "average")
        parts.append(f"Opp SoT against: {opp_sot:.1f} ({label})")

    poss = ev.get("team_possession")
    if poss is not None:
        parts.append(f"Possession: {poss:.0f}%")

    if parts:
        lines.append(f"  Matchup: {' | '.join(parts)}")


def _render_assists_evidence(lines: List[str], ev: Dict) -> None:
    """Render enhanced assists team-attack evidence."""
    parts = []

    team_gf = ev.get("team_goals_for_pm")
    if team_gf is not None:
        factor = ev.get("team_attack_factor", 1.0)
        label = "prolific" if factor > 1.05 else ("limited" if factor < 0.95 else "average")
        parts.append(f"Team goals: {team_gf:.2f}/m ({label})")

    opp_ga = ev.get("opp_goals_against_pm")
    if opp_ga is not None:
        factor = ev.get("opp_defense_factor", 1.0)
        label = "leaky" if factor > 1.05 else ("solid" if factor < 0.95 else "average")
        parts.append(f"Opp concedes: {opp_ga:.2f}/m ({label})")

    ctrl = ev.get("player_control_index")
    if ctrl is not None:
        parts.append(f"Control: {ctrl:.2f}")

    xg = ev.get("team_xg_pm")
    if xg is not None:
        parts.append(f"Team xG: {xg:.2f}")

    if parts:
        lines.append(f"  Matchup: {' | '.join(parts)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stat_to_rate_field(stat: str) -> str:
    return {"goals": "goals_per_90", "sot": "sot_per_90",
            "assists": "assists_per_90", "cards": "cards_per_90",
            "fouls": "fouls_per_90"}.get(stat, f"{stat}_per_90")


def _stat_to_starter_rate_field(stat: str) -> str:
    return {"goals": "starter_goals_per_90", "sot": "starter_sot_per_90",
            "assists": "starter_assists_per_90", "cards": "starter_cards_per_90",
            "fouls": "starter_fouls_per_90"
            }.get(stat, f"starter_{stat}_per_90")


def _stat_to_raw_field(stat: str) -> str:
    return {"goals": "goals", "sot": "shots_on", "assists": "assists",
            "cards": "cards_total", "fouls": "fouls_committed"}.get(stat, stat)


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

    # Enhanced matchup rationale
    matchup_model = ev.get("matchup_model", "")
    if matchup_model == "cards_full":
        fouls = ev.get("player_fouls_per_90")
        ref_name = ev.get("referee_name")
        ref_strict = ev.get("referee_strictness")
        if fouls is not None:
            rat.append(f"Fouls/90: {fouls:.1f}")
        if ref_name and ref_strict:
            rat.append(f"Referee: {ref_name} (strictness {ref_strict:.2f}x)")
        agg = ev.get("player_aggression_index")
        if agg is not None:
            rat.append(f"Player aggression: {agg:.2f}")
    elif matchup_model == "goals_shot_chain":
        shots = ev.get("shots_per_90")
        conv = ev.get("conversion_rate")
        if shots is not None and conv is not None:
            rat.append(f"Shot chain: {shots:.1f} shots/90 x {conv:.0%} conversion")
        xg = ev.get("team_xg_pm")
        if xg is not None:
            rat.append(f"Team xG: {xg:.2f}/match")
    elif matchup_model == "sot_shot_volume":
        shots = ev.get("shots_per_90")
        conv = ev.get("sot_conversion")
        if shots is not None and conv is not None:
            rat.append(f"Shot volume: {shots:.1f}/90 x {conv:.0%} on target")
        poss = ev.get("team_possession")
        if poss is not None:
            rat.append(f"Team possession: {poss:.0f}%")
    elif matchup_model == "assists_team_attack":
        team_gf = ev.get("team_goals_for_pm")
        if team_gf is not None:
            rat.append(f"Team scores: {team_gf:.2f}/match")
        opp_ga = ev.get("opp_goals_against_pm")
        if opp_ga is not None:
            rat.append(f"Opp concedes: {opp_ga:.2f}/match")
    else:
        rat.append(f"Team {ev['team_share_factor']:.2f}x | "
                   f"Opponent {ev['opponent_multiplier']:.2f}x | "
                   f"Venue {ev.get('venue_factor', 1.0):.2f}x | "
                   f"Position {ev.get('position_relevance', 1.0):.0%}")

    role_tags = []
    if ev.get("probable_penalty_taker"):
        role_tags.append("penalty duty")
    if ev.get("probable_set_piece"):
        role_tags.append("set-piece role")
    if ev.get("high_usage_creator"):
        role_tags.append("high-usage role")
    if role_tags:
        rat.append("Role signal: " + ", ".join(role_tags))

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
