#!/usr/bin/env python3
"""
Settings
League

EPL

Templates
Match day

Today — Feb 11th
Workflow

Schedule & Availability
Prompt template

Schedule Check

All workflow examples

Session


RAG CLI v2 (from scratch, odds-first, lenient intent handling)

Goals:
- Capability-first: detect what the Odds API actually provides today.
- User-friendly: avoid hard failures when constraints are impossible.
- Constraint-aware: apply hard constraints when feasible, then gracefully fallback.
- Lightweight: focused parlay assistant with concise, evidence-backed output.
"""

from __future__ import annotations

import argparse
import copy
import difflib
from collections import Counter
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from itertools import combinations
from math import comb as _comb, log
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ensure local modules (chroma_backend, prob_models, etc.) are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
from dotenv import load_dotenv
from openai import OpenAI
from zoneinfo import ZoneInfo

try:
    from heuristics import top_markets
except ImportError:
    from Scripts.rag_ingest.heuristics import top_markets

try:
    from prob_models import (
        over_prob, under_prob, interval_prob, implied_prob, remove_vig_two_way,
        value_edge, expected_value, kelly_fraction,
        poisson_pmf, negbin_pmf, negbin_from_mean_var,
    )
except ImportError:
    from Scripts.rag_ingest.prob_models import (
        over_prob, under_prob, interval_prob, implied_prob, remove_vig_two_way,
        value_edge, expected_value, kelly_fraction,
        poisson_pmf, negbin_pmf, negbin_from_mean_var,
    )

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

try:
    from knockout_context import get_knockout_context, knockout_evidence_line, KnockoutContext
    _HAS_KNOCKOUT = True
except ImportError:
    try:
        from Scripts.rag_ingest.knockout_context import get_knockout_context, knockout_evidence_line, KnockoutContext
        _HAS_KNOCKOUT = True
    except ImportError:
        _HAS_KNOCKOUT = False
        from dataclasses import dataclass as _dc_ko

        @_dc_ko
        class KnockoutContext:
            is_knockout: bool = False
            goals_modifier: float = 1.0
            corners_modifier: float = 1.0
            cards_modifier: float = 1.0
            sot_modifier: float = 1.0
            source: str = "none"

        def get_knockout_context(*a, **kw) -> "KnockoutContext":
            return KnockoutContext()

        def knockout_evidence_line(ctx) -> None:
            return None

try:
    from stake_context import get_stake_context, stake_evidence_line, StakeContext
    _HAS_STAKES = True
except ImportError:
    try:
        from Scripts.rag_ingest.stake_context import get_stake_context, stake_evidence_line, StakeContext
        _HAS_STAKES = True
    except ImportError:
        _HAS_STAKES = False
        from dataclasses import dataclass as _dc_st

        @_dc_st
        class StakeContext:
            goals_modifier: float = 1.0
            corners_modifier: float = 1.0
            cards_modifier: float = 1.0
            sot_modifier: float = 1.0
            source: str = "none"

        def get_stake_context(*a, **kw) -> "StakeContext":
            return StakeContext()

        def stake_evidence_line(ctx) -> None:
            return None

try:
    from lineup_context import get_lineup_context, lineup_evidence_line, LineupContext
    _HAS_LINEUP = True
except ImportError:
    try:
        from Scripts.rag_ingest.lineup_context import get_lineup_context, lineup_evidence_line, LineupContext
        _HAS_LINEUP = True
    except ImportError:
        _HAS_LINEUP = False
        from dataclasses import dataclass as _dc_lu

        @_dc_lu
        class LineupContext:
            home_starters: list = None
            away_starters: list = None
            home_adjustment: dict = None
            away_adjustment: dict = None
            home_missing_key: list = None
            away_missing_key: list = None
            source: str = "unavailable"
            is_available: bool = False

        def get_lineup_context(*a, **kw) -> "LineupContext":
            return LineupContext()

        def lineup_evidence_line(ctx) -> None:
            return None

try:
    from chroma_backend import backend_description, env_bool, env_first, get_chroma_client
except ImportError:
    from Scripts.rag_ingest.chroma_backend import backend_description, env_bool, env_first, get_chroma_client

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

try:
    from player_projections import (
        projected_player_stat, recommend_player_prop,
        render_player_prop_answer, PlayerPropRecommendation,
    )
    _HAS_PLAYER_PROJ = True
except ImportError:
    try:
        from Scripts.rag_ingest.player_projections import (
            projected_player_stat, recommend_player_prop,
            render_player_prop_answer, PlayerPropRecommendation,
        )
        _HAS_PLAYER_PROJ = True
    except ImportError:
        _HAS_PLAYER_PROJ = False

try:
    from prediction_tracker import log_from_render_output as _log_predictions
    _HAS_TRACKER = True
except ImportError:
    try:
        from Scripts.rag_ingest.prediction_tracker import log_from_render_output as _log_predictions
        _HAS_TRACKER = True
    except ImportError:
        _HAS_TRACKER = False
        def _log_predictions(*a, **kw): return 0


load_dotenv()

# ---- Config ----
ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = str(ROOT / "Index" / "chroma")
COLLECTION = env_first("CHROMA_COLLECTION", default="football_top5")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_KEY = env_first("ODDS-API", "ODDS_API_KEY")
ODDS_API_REGIONS = env_first("ODDS_API_REGIONS", default="uk,eu,us")
OPENAI_API_KEY = env_first("OPENAI_API_KEY")
CHAT_MODEL = env_first("RAG_CHAT_MODEL", default="gpt-5.4")

LEAGUE_TO_ODDS_SPORT = {
    "EPL": "soccer_epl",
    "LaLiga": "soccer_spain_la_liga",
    "SerieA": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue1": "soccer_france_ligue_one",
    "UCL": "soccer_uefa_champs_league",
    "UEL": "soccer_uefa_europa_league",
    "UECL": "soccer_uefa_europa_conference_league",
}
ALL_LEAGUES = list(LEAGUE_TO_ODDS_SPORT.keys())
DOMESTIC_LEAGUES = {"EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1"}
EUROPEAN_COMPETITIONS = {"UCL", "UEL", "UECL"}

# Provider-safe defaults. Some subscriptions do not expose extra market families.
DEFAULT_MARKETS = ["h2h", "totals", "spreads"]
KNOWN_GROUPS = {"corners", "cards", "moneyline", "totals", "spreads", "sot", "btts"}
AROUND_TARGET_TOLERANCE = 0.20
try:
    MIN_LEG_ODDS = max(1.01, float(env_first("MIN_LEG_ODDS", default="1.10")))
except Exception:
    MIN_LEG_ODDS = 1.10
# Higher floor for totals-style markets — removes trivially safe lines like o6.5 corners @1.12
MIN_TOTALS_LEG_ODDS = 1.25

# ---- Scoring Weights (centralized for regression testing + tuning) ----
SCORING_WEIGHTS = {
    "combo": {
        "around_target_log_penalty": 12.0,
        "neutral_anchor_base": 1.38,
        "neutral_anchor_deviation": 4.8,
        "target_abs_deviation": 0.6,
        "target_over_limit_penalty": 3.5,
        "around_over_tolerance_penalty": 15.0,
        "desired_leg_floor_ratio": 0.82,
        "desired_leg_under_penalty": 9.5,
        "soft_prefer_miss": 0.35,
        "conservative_threshold": 1.25,
        "conservative_penalty": 24.0,
        "kb_quality_weight": 1.2,
        "ml_spread_stack_penalty": 6.0,
        "hard_exclude_penalty": 25.0,
        "forbid_spread_penalty": 30.0,
        "require_total_corner_penalty": 35.0,
        "hard_include_missing": 18.0,
        "composition_missing": 22.0,
    },
    "kb_quality": {
        "heuristic_group_bonus": 1.5,
        "form_edge": 1.0,
        "control_edge": 0.8,
        "dominance_edge": 0.6,
        "sot_edge": 0.08,
        "goals_over_offset": 2.5,
        "goals_under_offset": 2.5,
        "goals_line_fit": 0.55,
        "corners_line_fit": 0.50,
        "corners_side_edge": 0.55,
        "cards_line_fit": 0.7,
        "cards_foul_momentum": 0.15,
        "cards_side_edge": 0.6,
        "cards_opp_induced_w": 0.12,  # halved: projected_cards() already includes induction
        "cards_matchup_adjust_w": 0.15,
        "corners_dom_asymmetry_w": 0.20,
        "corners_style_clash_w": 0.35,
        "sot_over_offset": 8.0,
        "sot_under_offset": 10.5,
        "sot_line_fit": 0.40,
        "long_odds_threshold": 3.0,
        "long_odds_offset": 2.5,
        "long_odds_penalty": 0.7,
        "btts_quality_weight": 1.5,
        "spreads_edge_weight": 1.2,
    },
    "totals_line": {
        "min_odds_hard": 1.20,
        "min_odds_preferred": 1.35,
        "ideal_odds_low": 1.55,
        "ideal_odds_high": 2.20,
        "proximity_penalty": 0.55,
        "edge_weight": 1.15,
        "short_price_penalty": 2.0,
        "ideal_range_bonus": 0.18,
        "outside_ideal_bonus": 0.10,
        "far_threshold": 3.5,
        "far_penalty": 1.3,
    },
    "projection": {
        "corners_own": 0.6,
        "corners_opp": 0.4,
        "goals_own": 0.6,
        "goals_opp": 0.4,
        "cards_base": 0.85,
        "cards_agg": 0.15,
        "cards_agg_scale": 3.0,
        "blend_season": 0.75,
        "blend_recent": 0.25,
        # Stat-specific blend overrides (backtested across 632 fixtures, 5 leagues)
        "blend_goals_season": 0.85,
        "blend_goals_recent": 0.15,
        "blend_corners_season": 0.80,
        "blend_corners_recent": 0.20,
        "blend_cards_season": 0.85,
        "blend_cards_recent": 0.15,
        "blend_sot_season": 0.80,
        "blend_sot_recent": 0.20,
        "corners_venue_blend": 0.50,
        "goals_venue_blend": 0.50,
        "sot_venue_blend": 0.50,
        "xg_blend": 0.50,  # 50% xG + 50% actual goals; best hit rate at 2.5 line
        # Home advantage: 0.0 — venue splits already capture this fully
        "home_advantage": 0.0,
    },
    "comparison_cards": {
        "cards": 0.55,
        "fouls": 0.08,
        "agg": 0.35,
        "agg_scale": 2.0,
        "recent_cards": 0.25,
        "recent_fouls": 0.03,
        "opp_control": 0.25,
        "opp_induced": 0.20,
    },
    "comparison_corners": {
        "proj": 0.50,
        "season_for": 0.25,
        "opp_allow": 0.15,
        "recent_for": 0.15,
        "control": 0.25,
        "edge": 0.12,
        "shots": 0.08,
    },
    "comparison_sot": {
        "proj": 0.45,
        "season_for": 0.25,
        "opp_allow": 0.15,
        "recent_for": 0.25,
        "sot_ratio": 0.20,
    },
    "projection_sot": {
        "own": 0.6,
        "opp": 0.4,
    },
    "projection_cards": {
        "own": 0.70,
        "opp": 0.30,
        "venue_blend": 0.50,
        "agg_nudge": 0.05,
        "agg_scale": 2.0,
        "foul_card_blend": 0.15,
    },
    "team_lines": {
        "corners_season": 0.70,
        "corners_recent": 0.30,
        "corners_recent_own": 0.50,
        "cards_season": 0.65,
        "cards_recent": 0.35,
        "cards_recent_own": 0.50,
        "corners_model": {
            "season": 0.43,
            "recent": 0.34,
            "style": 0.23,
            "trend_nudge": 0.06,
            "trend_cap": 0.25,
        },
        "corners_style": {
            "shots": 0.01,
            "sot": 0.05,
            "control": 1.50,
            "dominance": 0.85,
            "possession": 1.60,
            "opp_sot_against": 0.23,
        },
        "cards_direct": {
            "season": 0.65,
            "recent": 0.35,
        },
        "cards_output": {
            "share_weight": 0.50,
            "stabilizer_weight": 0.50,
        },
        "cards_share": {
            "season_blend": 0.68,
            "recent_blend": 0.32,
            "shrink": 0.06,
            "floor": 0.26,
            "ceiling": 0.71,
            "season_own": 0.34,
            "season_opp_induced": 0.43,
            "season_fouls": 0.06,
            "season_aggression": 0.02,
            "season_cards_per_foul": 0.05,
            "season_opp_control": 0.97,
            "recent_own": 0.41,
            "recent_opp_induced": 0.59,
            "recent_fouls": 0.10,
            "recent_aggression": 0.46,
            "recent_low_control": 1.47,
            "recent_opp_control": 0.84,
        },
    },
    "projection_spreads": {
        "form_w": 0.10,
        "dominance_w": 0.10,
        "blend_season": 0.75,
        "blend_recent": 0.25,
    },
    "spreads_line": {
        "edge_weight": 1.2,
        "proximity_penalty": 0.40,
        "min_odds_hard": 1.20,
        "min_odds_preferred": 1.40,
        "ideal_odds_low": 1.60,
        "ideal_odds_high": 2.30,
        "short_price_penalty": 1.5,
        "ideal_range_bonus": 0.20,
        "outside_ideal_bonus": 0.12,
    },
    "confidence": {
        "edge_high": 0.75,
        "edge_medium": 0.35,
        "corners_gap_high": 0.60,
        "corners_gap_medium": 0.25,
        "cards_gap_high": 0.45,
        "cards_gap_medium": 0.20,
        "sot_gap_high": 0.50,
        "sot_gap_medium": 0.20,
    },
    "referee": {
        "modifier_weight": 0.35,
        "anchor_weight": 0.15,
        "min_sample_size": 8,
        "full_confidence_sample": 20,
        "evidence_threshold": 0.02,
    },
    "prob": {
        "value_edge_weight": 2.0,
        "negative_ev_penalty": 3.0,
        "min_value_edge": 0.02,
        "min_model_prob": 0.55,
        "margin_std_default": 1.25,
        "prob_quality_weight": 1.5,
        "ev_combo_weight": 0.5,
        "model_prob_weight": 4.0,
    },
    "pool": {
        "default_anchor": 1.55,
        "quality_bonus": 0.12,
        "min_kb_quality": 0.0,  # legs with KB quality <= this are rejected from the pool
        "cap": 220,
        "beam_width": 60,
        "brute_force_threshold": 50000,
    },
    "group_preference": {
        "corners": 0.72,
        "cards": 1.0,
        "totals": 1.05,
        "sot": 0.95,
        "moneyline": 1.0,
        "spreads": 1.0,
        "btts": 1.0,
    },
    "diversity": {
        "concentration_threshold": 0.5,
        "concentration_penalty": 3.5,
        "unique_group_bonus": 0.6,
        "max_same_group": 2,
        "excess_same_penalty": 4.0,
        "unique_league_bonus": 0.3,
        "league_concentration_penalty": 2.0,
    },
    "ml": {
        "blend_weight": 0.0,  # static fallback (used when dynamic_blend=False). Set >0 to force blend.
        "dynamic_blend": True,  # R2-based per-stat blend: cards=12.5%, sot=20%, corners=0% (R2<0.20).
        "skip_stats": {"goals"},  # Goals heuristic already beats ML — skip to avoid adding bias.
        "elo_signal_weight": 0.15,  # Elo signal still active (doesn't suffer from the same skew)
        "elo_scale": 400.0,
        "min_elo_diff": 20,
    },
    "recency": {
        "alpha": 0.85,  # exponential decay per match (most recent=1.0, next=0.85, then 0.72…)
        "trend_weight": 0.0,  # disabled: slope signal too noisy with 6-game windows
    },
    "european": {
        "domestic_weight": 0.80,       # weight for domestic league profile in blended projections
        "euro_weight": 0.20,           # weight for European competition profile
        "min_euro_fixtures": 3,        # minimum European fixtures before blending kicks in
        "non_top5_confidence_penalty": 0.15,  # confidence penalty for teams without domestic profiles
    },
    "market_blend": {
        "model_weight": 0.70,          # weight for our model projection
        "market_weight": 0.30,         # weight for market-implied total
    },
}

client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

memory = {
    "last_query": None,
    "last_answer": None,
    "last_selected_event_ids": [],
    "last_selected_fixtures": [],
    "last_leg_count": None,
    "last_intent": None,
    "last_league": None,
    "last_constraint_spec": None,
    "last_selected_legs": [],
    "last_market_groups_used": [],
    "history": [],
}

_team_meta_cache: Dict[Tuple[str, str], Dict] = {}
_team_profile_doc_cache: Dict[Tuple[str, str], Dict] = {}
_team_recent_stats_cache: Dict[Tuple, Dict] = {}
_team_fixture_meta_cache: Dict[Tuple[str, str, str, str, str], Optional[Dict]] = {}
_chroma_client = None
_chroma_collection = None
_odds_response_cache: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], object] = {}
_event_market_keys_cache: Dict[Tuple[str, str], Set[str]] = {}
_event_odds_cache: Dict[Tuple[str, str, Tuple[str, ...]], Dict] = {}


HALF_MARKET_PATTERN = re.compile(r"(?:^|_)(h1|h2|1st_half|2nd_half|first_half|second_half)(?:_|$)")


@dataclass(frozen=True)
class CandidateLeg:
    event_id: str
    fixture: str
    home_team: str
    away_team: str
    market_key: str
    outcome: str
    odds: float
    point: Optional[float]
    bookmaker: Optional[str]
    league: str = ""
    event: Optional[Dict] = field(default=None, repr=False, hash=False, compare=False)


@dataclass
class ConstraintSpec:
    requested_markets: Set[str]
    hard_include_groups: Set[str]
    hard_exclude_groups: Set[str]
    soft_prefer_groups: Set[str]
    required_group_counts: Dict[str, int]
    forbid_spread_keys: bool
    require_total_corner_keys: bool
    target_multiplier: Optional[float]
    target_mode: str  # none | around | max | min
    leg_count: Optional[int]
    per_match_mode: bool
    require_unique_events: bool
    time_window: str  # today | tomorrow | upcoming
    league: str
    leagues: List[str] = None       # cross-league mode: list of leagues to fetch
    target_min: Optional[float] = None   # odds range lower bound
    target_max: Optional[float] = None   # odds range upper bound


def _leg_league(leg: CandidateLeg, fallback: str) -> str:
    """Return the league for a leg, falling back to constraint league."""
    return leg.league if leg.league else fallback


TEAM_ALIAS_MAP = {
    # Arsenal
    "arsenal": "Arsenal",
    "arsenal fc": "Arsenal",
    "the gunners": "Arsenal",
    # Aston Villa
    "aston villa": "Aston Villa",
    "aston villa fc": "Aston Villa",
    "villa": "Aston Villa",
    # Bournemouth
    "bournemouth": "Bournemouth",
    "afc bournemouth": "Bournemouth",
    "bournemouth afc": "Bournemouth",
    "the cherries": "Bournemouth",
    # Brentford
    "brentford": "Brentford",
    "brentford fc": "Brentford",
    "the bees": "Brentford",
    # Brighton
    "brighton": "Brighton",
    "brighton and hove albion": "Brighton",
    "brighton and hove albion fc": "Brighton",
    "the seagulls": "Brighton",
    # Burnley
    "burnley": "Burnley",
    "burnley fc": "Burnley",
    "the clarets": "Burnley",
    # Chelsea
    "chelsea": "Chelsea",
    "chelsea fc": "Chelsea",
    "the blues": "Chelsea",
    # Crystal Palace
    "crystal palace": "Crystal Palace",
    "crystal palace fc": "Crystal Palace",
    "palace": "Crystal Palace",
    "cpfc": "Crystal Palace",
    # Everton
    "everton": "Everton",
    "everton fc": "Everton",
    "the toffees": "Everton",
    # Fulham
    "fulham": "Fulham",
    "fulham fc": "Fulham",
    # Leeds
    "leeds": "Leeds",
    "leeds united": "Leeds",
    "leeds united fc": "Leeds",
    "lufc": "Leeds",
    # Liverpool
    "liverpool": "Liverpool",
    "liverpool fc": "Liverpool",
    "the reds": "Liverpool",
    # Manchester City
    "manchester city": "Manchester City",
    "manchester city fc": "Manchester City",
    "man city": "Manchester City",
    "city": "Manchester City",
    "mcfc": "Manchester City",
    # Manchester United
    "manchester united": "Manchester United",
    "manchester united fc": "Manchester United",
    "man united": "Manchester United",
    "man utd": "Manchester United",
    "mufc": "Manchester United",
    # Newcastle
    "newcastle": "Newcastle",
    "newcastle united": "Newcastle",
    "newcastle united fc": "Newcastle",
    "nufc": "Newcastle",
    # Nottingham Forest
    "nottingham forest": "Nottingham Forest",
    "nottingham forest fc": "Nottingham Forest",
    "nottm forest": "Nottingham Forest",
    "forest": "Nottingham Forest",
    # Sunderland
    "sunderland": "Sunderland",
    "sunderland afc": "Sunderland",
    "sunderland fc": "Sunderland",
    "safc": "Sunderland",
    # Tottenham
    "tottenham": "Tottenham",
    "tottenham hotspur": "Tottenham",
    "tottenham hotspur fc": "Tottenham",
    "spurs": "Tottenham",
    # West Ham
    "west ham": "West Ham",
    "west ham united": "West Ham",
    "west ham united fc": "West Ham",
    "whufc": "West Ham",
    "hammers": "West Ham",
    # Wolves
    "wolves": "Wolves",
    "wolverhampton": "Wolves",
    "wolverhampton wanderers": "Wolves",
    "wolverhampton wanderers fc": "Wolves",
    "wwfc": "Wolves",
    # --- LaLiga ---
    # Alaves
    "alaves": "Alaves",
    "deportivo alaves": "Alaves",
    "alavés": "Alaves",
    # Athletic Club
    "athletic club": "Athletic Club",
    "athletic bilbao": "Athletic Club",
    "bilbao": "Athletic Club",
    # Atletico Madrid
    "atletico madrid": "Atletico Madrid",
    "atlético madrid": "Atletico Madrid",
    "atletico": "Atletico Madrid",
    "atleti": "Atletico Madrid",
    # Barcelona
    "barcelona": "Barcelona",
    "fc barcelona": "Barcelona",
    "barca": "Barcelona",
    "barça": "Barcelona",
    # Celta Vigo
    "celta vigo": "Celta Vigo",
    "celta": "Celta Vigo",
    "rc celta": "Celta Vigo",
    # Elche
    "elche": "Elche",
    "elche cf": "Elche",
    # Espanyol
    "espanyol": "Espanyol",
    "rcd espanyol": "Espanyol",
    # Getafe
    "getafe": "Getafe",
    "getafe cf": "Getafe",
    # Girona
    "girona": "Girona",
    "girona fc": "Girona",
    # Levante
    "levante": "Levante",
    "levante ud": "Levante",
    # Mallorca
    "mallorca": "Mallorca",
    "rcd mallorca": "Mallorca",
    # Osasuna
    "osasuna": "Osasuna",
    "ca osasuna": "Osasuna",
    # Oviedo
    "oviedo": "Oviedo",
    "real oviedo": "Oviedo",
    # Rayo Vallecano
    "rayo vallecano": "Rayo Vallecano",
    "rayo": "Rayo Vallecano",
    # Real Betis
    "real betis": "Real Betis",
    "betis": "Real Betis",
    # Real Madrid
    "real madrid": "Real Madrid",
    "real madrid cf": "Real Madrid",
    "madrid": "Real Madrid",
    # Real Sociedad
    "real sociedad": "Real Sociedad",
    "sociedad": "Real Sociedad",
    "la real": "Real Sociedad",
    # Sevilla
    "sevilla": "Sevilla",
    "sevilla fc": "Sevilla",
    # Valencia
    "valencia": "Valencia",
    "valencia cf": "Valencia",
    # Villarreal
    "villarreal": "Villarreal",
    "villarreal cf": "Villarreal",
    # --- Bundesliga ---
    # 1. FC Heidenheim
    "1. fc heidenheim": "1. FC Heidenheim",
    "heidenheim": "1. FC Heidenheim",
    # 1. FC Köln
    "1. fc köln": "1. FC Köln",
    "1. fc koln": "1. FC Köln",
    "koln": "1. FC Köln",
    "köln": "1. FC Köln",
    "fc koln": "1. FC Köln",
    # 1899 Hoffenheim
    "1899 hoffenheim": "1899 Hoffenheim",
    "hoffenheim": "1899 Hoffenheim",
    "tsg hoffenheim": "1899 Hoffenheim",
    # Bayer Leverkusen
    "bayer leverkusen": "Bayer Leverkusen",
    "leverkusen": "Bayer Leverkusen",
    "bayer 04": "Bayer Leverkusen",
    # Bayern München
    "bayern münchen": "Bayern München",
    "bayern munchen": "Bayern München",
    "bayern munich": "Bayern München",
    "bayern": "Bayern München",
    "fcb": "Bayern München",
    # Borussia Dortmund
    "borussia dortmund": "Borussia Dortmund",
    "dortmund": "Borussia Dortmund",
    "bvb": "Borussia Dortmund",
    # Borussia Mönchengladbach
    "borussia mönchengladbach": "Borussia Mönchengladbach",
    "borussia monchengladbach": "Borussia Mönchengladbach",
    "monchengladbach": "Borussia Mönchengladbach",
    "mönchengladbach": "Borussia Mönchengladbach",
    "gladbach": "Borussia Mönchengladbach",
    # Eintracht Frankfurt
    "eintracht frankfurt": "Eintracht Frankfurt",
    "frankfurt": "Eintracht Frankfurt",
    "sge": "Eintracht Frankfurt",
    # FC Augsburg
    "fc augsburg": "FC Augsburg",
    "augsburg": "FC Augsburg",
    # FC St. Pauli
    "fc st. pauli": "FC St. Pauli",
    "fc st pauli": "FC St. Pauli",
    "st. pauli": "FC St. Pauli",
    "st pauli": "FC St. Pauli",
    # FSV Mainz 05
    "fsv mainz 05": "FSV Mainz 05",
    "mainz": "FSV Mainz 05",
    "mainz 05": "FSV Mainz 05",
    # Hamburger SV
    "hamburger sv": "Hamburger SV",
    "hamburg": "Hamburger SV",
    "hsv": "Hamburger SV",
    # RB Leipzig
    "rb leipzig": "RB Leipzig",
    "leipzig": "RB Leipzig",
    "rasenballsport leipzig": "RB Leipzig",
    # SC Freiburg
    "sc freiburg": "SC Freiburg",
    "freiburg": "SC Freiburg",
    # Union Berlin
    "union berlin": "Union Berlin",
    "1. fc union berlin": "Union Berlin",
    # VfB Stuttgart
    "vfb stuttgart": "VfB Stuttgart",
    "stuttgart": "VfB Stuttgart",
    # VfL Wolfsburg
    "vfl wolfsburg": "VfL Wolfsburg",
    "wolfsburg": "VfL Wolfsburg",
    # Werder Bremen
    "werder bremen": "Werder Bremen",
    "bremen": "Werder Bremen",
    "sv werder bremen": "Werder Bremen",
    # --- Ligue 1 ---
    # Angers
    "angers": "Angers",
    "angers sco": "Angers",
    # Auxerre
    "auxerre": "Auxerre",
    "aj auxerre": "Auxerre",
    # Le Havre
    "le havre": "Le Havre",
    "le havre ac": "Le Havre",
    # Lens
    "lens": "Lens",
    "rc lens": "Lens",
    # Lille
    "lille": "Lille",
    "losc lille": "Lille",
    "losc": "Lille",
    # Lorient
    "lorient": "Lorient",
    "fc lorient": "Lorient",
    # Lyon
    "lyon": "Lyon",
    "olympique lyonnais": "Lyon",
    "ol": "Lyon",
    # Marseille
    "marseille": "Marseille",
    "olympique de marseille": "Marseille",
    "olympique marseille": "Marseille",
    "om": "Marseille",
    # Metz
    "metz": "Metz",
    "fc metz": "Metz",
    # Monaco
    "monaco": "Monaco",
    "as monaco": "Monaco",
    # Nantes
    "nantes": "Nantes",
    "fc nantes": "Nantes",
    # Nice
    "nice": "Nice",
    "ogc nice": "Nice",
    # Paris FC
    "paris fc": "Paris FC",
    # Paris Saint Germain
    "paris saint germain": "Paris Saint Germain",
    "paris saint-germain": "Paris Saint Germain",
    "paris sg": "Paris Saint Germain",
    "psg": "Paris Saint Germain",
    # Rennes
    "rennes": "Rennes",
    "stade rennais": "Rennes",
    # Stade Brestois 29
    "stade brestois 29": "Stade Brestois 29",
    "stade brestois": "Stade Brestois 29",
    "brest": "Stade Brestois 29",
    "brestois": "Stade Brestois 29",
    # Strasbourg
    "strasbourg": "Strasbourg",
    "rc strasbourg": "Strasbourg",
    "rc strasbourg alsace": "Strasbourg",
    # Toulouse
    "toulouse": "Toulouse",
    "toulouse fc": "Toulouse",
}


LEAGUE_HINTS = {
    "EPL": ["epl", "premier league", "england"],
    "LaLiga": ["laliga", "la liga", "spain"],
    "SerieA": ["serie a", "italy"],
    "Bundesliga": ["bundesliga", "germany"],
    "Ligue1": ["ligue 1", "france"],
    "UCL": ["ucl", "champions league", "uefa champions"],
    "UEL": ["uel", "europa league", "uefa europa league"],
    "UECL": ["uecl", "conference league", "europa conference"],
}

_CROSS_LEAGUE_PATTERNS = [
    r"all\s+(?:5\s+|8\s+)?leagues?",
    r"top\s+5\s+leagues?",
    r"cross[- ]league",
    r"any\s+league",
    r"mix\s+leagues",
    r"from\s+all\s+leagues",
    r"across\s+(?:all\s+)?leagues",
    r"multiple\s+leagues",
    r"every\s+league",
    r"all\s+(?:european\s+)?leagues",
    r"all\s+(?:8\s+)?competitions?",
    r"european\s+(?:and\s+domestic|competitions?)",
]


def resolve_league(user_q: str, default: str = "EPL") -> str:
    q = user_q.lower()
    for lg, keys in LEAGUE_HINTS.items():
        if any(k in q for k in keys):
            return lg
    return default


def _detect_cross_league(user_q: str) -> List[str]:
    """Detect cross-league intent. Returns list of leagues (empty = single league)."""
    q = user_q.lower()
    is_cross = any(re.search(p, q) for p in _CROSS_LEAGUE_PATTERNS)
    # Also detect explicit multi-league mentions (e.g. "EPL and LaLiga")
    mentioned = [lg for lg, keys in LEAGUE_HINTS.items() if any(k in q for k in keys)]
    if len(mentioned) >= 2:
        is_cross = True
    if not is_cross:
        return []
    # If a broad cross-league pattern matched (e.g. "all top 5 leagues"),
    # return ALL leagues even if specific leagues are also mentioned.
    broad_pattern_matched = any(re.search(p, q) for p in _CROSS_LEAGUE_PATTERNS)
    if broad_pattern_matched:
        return list(ALL_LEAGUES)
    return mentioned if len(mentioned) >= 2 else list(ALL_LEAGUES)


def parse_time_window(user_q: str) -> str:
    q = user_q.lower()
    if "today" in q or "tonight" in q:
        return "today"
    if "tomorrow" in q:
        return "tomorrow"
    return "upcoming"


def parse_explicit_date(user_q: str, tz_name: str = "Europe/London") -> Optional[date]:
    """
    Parse explicit date mentions like:
    - 10th of February
    - February 10
    - Feb 10
    - 10/02 or 10-02
    """
    q = user_q.lower()
    month_map = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    tz = ZoneInfo(tz_name)
    today = datetime.now(tz).date()

    day = None
    month = None
    year = None

    # "10th of february [2026]"
    m = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:of\s+)?"
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
        r"aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"(?:\s*,?\s*(20\d{2}))?\b",
        q,
    )
    if m:
        day = int(m.group(1))
        month = month_map.get(m.group(2), None)
        if m.group(3):
            year = int(m.group(3))

    # "february 10 [2026]"
    if day is None or month is None:
        m = re.search(
            r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
            r"aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+"
            r"(\d{1,2})(?:st|nd|rd|th)?(?:\s*,?\s*(20\d{2}))?\b",
            q,
        )
        if m:
            month = month_map.get(m.group(1), None)
            day = int(m.group(2))
            if m.group(3):
                year = int(m.group(3))

    # "10/02[/2026]" or "10-02[-2026]" (assume DD/MM)
    if day is None or month is None:
        m = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](20\d{2}|\d{2}))?\b", q)
        if m:
            day = int(m.group(1))
            month = int(m.group(2))
            if m.group(3):
                y = m.group(3)
                year = int(y) if len(y) == 4 else (2000 + int(y))

    if day is None or month is None:
        return None

    if year is None:
        year = today.year
        # If the date already passed this season, roll to next year.
        try:
            tentative = date(year, month, day)
        except ValueError:
            return None
        if tentative < today:
            year += 1

    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_target(user_q: str) -> Tuple[Optional[float], str]:
    q = user_q.lower()
    patterns = [
        (r"(?:no more than|under|below|at most|max(?:imum)?)\s*(\d+(?:\.\d+)?)\s*x?", "max"),
        (r"(?:at least|over|above|min(?:imum)?)\s*(\d+(?:\.\d+)?)\s*x?", "min"),
        (r"(?:around|about|approx(?:imately)?|target(?: of)?)\s*(\d+(?:\.\d+)?)\s*x?", "around"),
        (r"(\d+(?:\.\d+)?)\s*x", "around"),
    ]
    for pat, mode in patterns:
        m = re.search(pat, q)
        if not m:
            continue
        try:
            return float(m.group(1)), mode
        except ValueError:
            break
    return None, "none"


def parse_target_range(user_q: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse an odds range like 'between 3x and 8x', 'odds 5-10', '3x to 8x'."""
    q = user_q.lower()
    patterns = [
        r"between\s+(\d+(?:\.\d+)?)\s*x?\s+and\s+(\d+(?:\.\d+)?)\s*x?",
        r"(?:odds?\s+)?range\s+(\d+(?:\.\d+)?)\s*x?\s*[-\u2013]\s*(\d+(?:\.\d+)?)\s*x?",
        r"from\s+(\d+(?:\.\d+)?)\s*x?\s+to\s+(\d+(?:\.\d+)?)\s*x?",
        r"(\d+(?:\.\d+)?)\s*x?\s+to\s+(\d+(?:\.\d+)?)\s*x?",
        r"(\d+(?:\.\d+)?)\s*x?\s*[-\u2013]\s*(\d+(?:\.\d+)?)\s*x?\s*(?:odds|combined|parlay)",
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            try:
                lo, hi = float(m.group(1)), float(m.group(2))
                if lo > hi:
                    lo, hi = hi, lo
                return lo, hi
            except ValueError:
                pass
    return None, None


def parse_leg_count(user_q: str) -> Optional[int]:
    q = user_q.lower()
    m = re.search(r"(\d+)\s*[- ]?leg", q)
    if not m:
        # Also match "N game parlay", "N match parlay", "N fixture parlay"
        m = re.search(r"(\d+)\s*[- ]?(?:game|match|fixture)\s*(?:parlay|bet|acca)?", q)
    if not m:
        return None
    try:
        return max(1, min(int(m.group(1)), 20))
    except ValueError:
        return None


def market_group_from_key(market_key: str) -> str:
    k = (market_key or "").lower()
    if "corner" in k:
        return "corners"
    if "card" in k or "booking" in k:
        return "cards"
    # SoT must be checked BEFORE generic "total" catch-all
    if ("shot" in k and "target" in k) or k == "sot":
        return "sot"
    if "btts" in k or "both_teams" in k:
        return "btts"
    if "h2h" in k or "moneyline" in k or "1x2" in k:
        return "moneyline"
    if "spread" in k or "handicap" in k:
        return "spreads"
    if "total" in k:
        return "totals"
    return k


def is_full_game_market(market_key: str) -> bool:
    k = (market_key or "").lower().strip()
    if not k:
        return False
    # Exchange lay markets are semantically inverted and produce duplicate/confusing exposure.
    if "lay" in k:
        return False
    if "player" in k:
        return False
    if HALF_MARKET_PATTERN.search(k):
        return False
    return True


def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", (s or "").lower()).strip()


def canonical_team_name(name: str) -> str:
    n = normalize_name(name)
    if not n:
        return name
    # Exact alias match
    if n in TEAM_ALIAS_MAP:
        return TEAM_ALIAS_MAP[n]
    # Fuzzy fallback: find closest alias key (cutoff 0.65 to catch e.g. "Brighton Hove Albion" → "Brighton and Hove Albion")
    matches = difflib.get_close_matches(n, TEAM_ALIAS_MAP.keys(), n=1, cutoff=0.65)
    if matches:
        return TEAM_ALIAS_MAP[matches[0]]
    return name


def team_name_aliases(name: str) -> Set[str]:
    n = normalize_name(name)
    aliases = {n}
    # drop common suffix tokens for matching in casual queries
    for suffix in [" united", " hotspur", " fc", " afc", " cf"]:
        if n.endswith(suffix):
            aliases.add(n[: -len(suffix)].strip())
    # add known dictionary aliases
    canon = canonical_team_name(name)
    aliases.add(normalize_name(canon))
    for k, v in TEAM_ALIAS_MAP.items():
        if normalize_name(v) == normalize_name(canon):
            aliases.add(k)
    return {x for x in aliases if x}


def add_chat_turn(user_q: str, assistant_a: str, max_turns: int = 10) -> None:
    hist = memory.get("history") or []
    hist.append({"user": user_q, "assistant": assistant_a})
    memory["history"] = hist[-max_turns:]
    memory["last_query"] = user_q
    memory["last_answer"] = assistant_a


def render_recent_chat_context(max_turns: int = 3) -> str:
    hist = memory.get("history") or []
    if not hist:
        return "None."
    lines = []
    for i, turn in enumerate(hist[-max_turns:], start=1):
        u = (turn.get("user") or "").strip()
        a = (turn.get("assistant") or "").strip()
        lines.append(f"Turn {i} User: {u}")
        lines.append(f"Turn {i} Assistant: {a}")
    return "\n".join(lines)


def is_followup_request(user_q: str) -> bool:
    q = user_q.lower()
    if not ("leg" in q or "parlay" in q or "bet" in q):
        return False
    # Direct follow-up indicators
    if re.search(r"\b(add|another|those|previous)\b", q):
        return True
    # "same" is a follow-up indicator, but NOT in "same-game parlay" (market descriptor)
    if re.search(r"\bsame\b", q) and not re.search(r"\bsame[\s-]game\b", q):
        return True
    return False


def parse_season_rank(season_value: Optional[str]) -> int:
    if not season_value:
        return -1
    s = str(season_value)
    m = re.search(r"(20\d{2})", s)
    if not m:
        return -1
    try:
        return int(m.group(1))
    except ValueError:
        return -1


def build_where(league: Optional[str], extra_filters: Optional[List[Dict]] = None) -> Dict:
    clauses: List[Dict] = []
    if league:
        clauses.append({"league": league})
    if extra_filters:
        clauses.extend(extra_filters)
    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _numeric(value) -> Optional[float]:
    try:
        return float(value)
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


def get_collection_handle(create_if_missing: bool = True):
    global _chroma_client, _chroma_collection
    if _chroma_client is None:
        _chroma_client = get_chroma_client(CHROMA_DIR)
    if _chroma_collection is not None:
        return _chroma_collection
    if create_if_missing:
        _chroma_collection = _chroma_client.get_or_create_collection(COLLECTION)
    else:
        _chroma_collection = _chroma_client.get_collection(COLLECTION)
    return _chroma_collection


def kb_healthcheck() -> Tuple[bool, str]:
    try:
        col = get_collection_handle(create_if_missing=False)
    except Exception as exc:
        return False, f"Collection '{COLLECTION}' unavailable ({exc}). Backend={backend_description(CHROMA_DIR)}"
    try:
        cnt = col.count()
    except Exception:
        cnt = None
    if cnt is None:
        return True, f"Connected to {backend_description(CHROMA_DIR)} (count unavailable)"
    if cnt <= 0:
        return False, f"Collection '{COLLECTION}' is empty on {backend_description(CHROMA_DIR)}"
    return True, f"Connected to {backend_description(CHROMA_DIR)} | vectors={cnt}"


def parse_constraints(user_q: str, default_league: str = "EPL") -> ConstraintSpec:
    q = user_q.lower()
    target_multiplier, target_mode = parse_target(user_q)
    target_min, target_max = parse_target_range(user_q)
    explicit_leg_count = parse_leg_count(user_q)
    cross_leagues = _detect_cross_league(user_q)
    per_match_mode = bool(
        re.search(
            r"(as many legs as|for each game|for every game|for each fixture|for every fixture"
            r"|one leg per game|one leg per match|one leg per fixture"
            r"|one\s+\w+\s+per\s+(?:game|match|fixture)"          # "one corner per game", "one total per match"
            r"|one\s+\w+\s+\w+\s+per\s+(?:game|match|fixture)"   # "one corner total per game"
            r"|per\s+(?:game|match|fixture)\b"                     # trailing "per game"
            r"|all\s+(?:the\s+)?(?:games|matches|fixtures)"        # "all games", "all the matches"
            r"|every\s+(?:game|match|fixture)"                     # "every game"
            r"|include\s+all\s+(?:games|matches|fixtures)"           # "include all games"
            r"|\d+\s*[- ]?(?:game|match|fixture)\s*(?:parlay|bet|acca))",  # "5 game parlay"
            q,
        )
    )

    # Hard include/exclude groups
    hard_include: Set[str] = set()
    hard_exclude: Set[str] = set()
    soft_prefer: Set[str] = set()
    required_group_counts: Dict[str, int] = {}

    only_corners = bool(re.search(r"(only|just)\s+.*corner", q) or re.search(r"corner.*(only|just)", q))
    only_cards = bool(re.search(r"(only|just)\s+.*card", q) or re.search(r"card.*(only|just)", q))
    only_ml = bool(re.search(r"(only|just)\s+.*money\s*line", q) or re.search(r"(only|just)\s+.*\bml\b", q))
    only_sot = bool(re.search(r"(only|just)\s+.*(shot|sot)", q) or re.search(r"(shot|sot).*(only|just)", q))

    if only_corners:
        hard_include.add("corners")
    if only_cards:
        hard_include.add("cards")
    if only_ml:
        hard_include.add("moneyline")
    if only_sot:
        hard_include.add("sot")

    if only_corners:
        hard_exclude |= (KNOWN_GROUPS - {"corners"})
    if only_cards:
        hard_exclude |= (KNOWN_GROUPS - {"cards"})
    if only_ml:
        hard_exclude |= (KNOWN_GROUPS - {"moneyline"})
    if only_sot:
        hard_exclude |= (KNOWN_GROUPS - {"sot"})

    if re.search(r"(?:no|without|avoid|dont|don't)\s+.*money\s*line", q) or re.search(
        r"(?:no|without|avoid|dont|don't)\s+.*\bml\b", q
    ):
        hard_exclude.add("moneyline")

    # Soft preferences from query terms (ignore explicit negations)
    no_spreads_mentioned = bool(re.search(r"(?:no|without|avoid|dont|don't)\s+.*(?:spread|handicap)", q))
    no_totals_mentioned = bool(re.search(r"(?:no|without|avoid|dont|don't)\s+.*(?:total|over|under)", q))
    no_corners_mentioned = bool(re.search(r"(?:no|without|avoid|dont|don't)\s+.*corner", q))
    no_cards_mentioned = bool(re.search(r"(?:no|without|avoid|dont|don't)\s+.*(?:card|booking)", q))
    no_sot_mentioned = bool(re.search(r"(?:no|without|avoid|dont|don't)\s+.*(?:shot|sot)", q))

    if re.search(r"\bcorner(s)?\b", q):
        if not no_corners_mentioned:
            soft_prefer.add("corners")
    if re.search(r"\bcard(s)?\b", q) or re.search(r"\bbooking(s)?\b", q):
        if not no_cards_mentioned:
            soft_prefer.add("cards")
    if re.search(r"\b(shot|sot)\b", q) or "shots on target" in q:
        if not no_sot_mentioned:
            soft_prefer.add("sot")
    if re.search(r"\bspread(s)?\b", q) or re.search(r"\bhandicap(s)?\b", q):
        if not no_spreads_mentioned:
            soft_prefer.add("spreads")
    if re.search(r"\btotal(s)?\b", q) or re.search(r"\bover\b", q) or re.search(r"\bunder\b", q):
        if not no_totals_mentioned:
            soft_prefer.add("totals")
    if re.search(r"\bmoney\s*line\b", q) or re.search(r"\bml\b", q) or re.search(r"\bh2h\b", q):
        soft_prefer.add("moneyline")

    # Composition constraints (lenient NL parse)
    if re.search(r"(one|1)\s+leg.*corner", q):
        required_group_counts["corners"] = max(required_group_counts.get("corners", 0), 1)
    if re.search(r"(one|1)\s+leg.*(booking|card)", q):
        required_group_counts["cards"] = max(required_group_counts.get("cards", 0), 1)
    if re.search(r"(one|1)\s+leg.*money\s*line", q) or re.search(r"(one|1)\s+leg.*\bml\b", q):
        required_group_counts["moneyline"] = max(required_group_counts.get("moneyline", 0), 1)

    forbid_spread_keys = bool(re.search(r"(?:no|without|avoid|dont|don't)\s+.*(?:spread|handicap)", q))
    require_total_corner_keys = bool(
        re.search(r"total\s+corner", q)
        or re.search(r"corner\s+line", q)
    )

    # Requested market keys for provider fetch.
    # Keep this as provider-safe base; specialized markets (corners/cards)
    # are discovered per-event later.
    requested_markets = set(DEFAULT_MARKETS)
    if "moneyline" in hard_include:
        requested_markets = {"h2h"}
    elif hard_exclude == {"moneyline"}:
        requested_markets = {"totals", "spreads"}

    require_unique = bool(
        per_match_mode
        or re.search(r"(different matches|across matches|one per match|one per game|one per fixture)", q)
    )

    return ConstraintSpec(
        requested_markets=requested_markets,
        hard_include_groups=hard_include,
        hard_exclude_groups=hard_exclude,
        soft_prefer_groups=soft_prefer,
        required_group_counts=required_group_counts,
        forbid_spread_keys=forbid_spread_keys,
        require_total_corner_keys=require_total_corner_keys,
        target_multiplier=target_multiplier,
        target_mode=target_mode,
        leg_count=explicit_leg_count,
        per_match_mode=per_match_mode,
        require_unique_events=require_unique,
        time_window=parse_time_window(user_q),
        league=resolve_league(user_q, default=default_league),
        leagues=cross_leagues or None,
        target_min=target_min,
        target_max=target_max,
    )


def _freeze_params(params: Dict) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in (params or {}).items()))


def odds_get(path: str, params: Dict) -> Tuple[Optional[object], Optional[str]]:
    api_key = env_first("ODDS-API", "ODDS_API_KEY", default=ODDS_API_KEY)
    if not api_key:
        return None, "Missing ODDS API key (ODDS-API)."
    url = f"{ODDS_API_BASE}{path}"
    p = dict(params)
    p["api_key"] = api_key
    p["oddsFormat"] = "decimal"
    p["dateFormat"] = "iso"
    p["regions"] = ODDS_API_REGIONS

    cache_key = (path, _freeze_params(p))
    cached = _odds_response_cache.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached), None

    try:
        r = requests.get(url, params=p, timeout=25)
        if r.status_code != 200:
            return None, f"Odds API HTTP {r.status_code}: {(r.text or '')[:600]}"
        payload = r.json()
        _odds_response_cache[cache_key] = payload
        return copy.deepcopy(payload), None
    except Exception as exc:
        return None, f"Odds API request failed: {exc}"


def fetch_events(league: str, markets: Set[str],
                  target_date: Optional[date] = None) -> Tuple[List[Dict], List[str]]:
    """Fetch events and odds. Uses API-Football as primary provider (richer markets),
    falls back to The Odds API if API-Football returns no results."""
    from datetime import date as _date

    if target_date is None:
        target_date = _date.today()

    try:
        from odds_provider import fetch_events_apifootball
        return fetch_events_apifootball(league, target_date)
    except ImportError:
        return [], ["odds_provider module not available."]
    except Exception as exc:
        log.warning("API-Football odds failed for %s: %s", league, exc)
        return [], [f"API-Football error: {exc}"]


def fetch_events_multi(leagues: List[str], markets: Set[str]) -> Tuple[List[Dict], List[str]]:
    """Fetch events from multiple leagues, tagging each event with its league."""
    all_events: List[Dict] = []
    all_notes: List[str] = []
    for league in leagues:
        events, notes = fetch_events(league, markets)
        for ev in events:
            ev["_league"] = league
        all_events.extend(events)
        all_notes.extend(notes)
    return all_events, all_notes


def extract_market_keys(payload: object) -> Set[str]:
    keys: Set[str] = set()

    def scan_block(block: object) -> None:
        if isinstance(block, dict):
            if "bookmakers" in block:
                for bm in block.get("bookmakers", []) or []:
                    for mk in bm.get("markets", []) or []:
                        k = mk.get("key")
                        if k:
                            keys.add(str(k))
            if "markets" in block:
                for mk in block.get("markets", []) or []:
                    k = mk.get("key")
                    if k:
                        keys.add(str(k))
        elif isinstance(block, list):
            for item in block:
                scan_block(item)

    scan_block(payload)
    return keys


def event_market_groups(event: Dict) -> Set[str]:
    groups: Set[str] = set()
    for bm in event.get("bookmakers", []) or []:
        for mk in bm.get("markets", []) or []:
            k = mk.get("key")
            if k:
                groups.add(market_group_from_key(str(k)))
    return groups


def discover_event_market_keys(league: str, event_id: str) -> Tuple[Set[str], Optional[str]]:
    """Discover available market keys for an event via API-Football odds re-fetch."""
    cache_key = (league, event_id)
    cached = _event_market_keys_cache.get(cache_key)
    if cached is not None:
        return set(cached), None

    try:
        from odds_provider import enrich_event_odds
    except ImportError:
        return set(), "odds_provider module not available."

    # Build a minimal event dict for enrich_event_odds
    event = {"id": event_id, "_fixture_id": int(event_id), "bookmakers": [],
             "home_team": "", "away_team": ""}
    keys, err = enrich_event_odds(event, league)
    if err:
        return set(), err
    _event_market_keys_cache[cache_key] = set(keys)
    return keys, None


def fetch_event_odds_for_market_keys(league: str, event_id: str, market_keys: Set[str]) -> Tuple[Optional[Dict], Optional[str]]:
    """Fetch odds for specific markets via API-Football. Returns event dict with bookmakers."""
    if not market_keys:
        return None, "No market keys requested."

    try:
        from odds_provider import enrich_event_odds
    except ImportError:
        return None, "odds_provider module not available."

    event = {"id": event_id, "_fixture_id": int(event_id), "bookmakers": [],
             "home_team": "", "away_team": ""}
    keys, err = enrich_event_odds(event, league)
    if err:
        return None, err
    return event, None


def merge_bookmakers(base: List[Dict], extra: List[Dict]) -> List[Dict]:
    """
    Merge bookmaker market blocks by bookmaker title + market key + outcome key.
    Keep highest price when duplicates occur.
    """
    rows: Dict[Tuple[str, str, str, str], Dict] = {}

    def ingest(bookmakers: List[Dict]) -> None:
        for bm in bookmakers or []:
            bname = str(bm.get("title") or "")
            for mk in bm.get("markets", []) or []:
                mkey = str(mk.get("key") or "")
                for out in mk.get("outcomes", []) or []:
                    oname = str(out.get("name") or "")
                    pkey = "" if out.get("point") is None else str(out.get("point"))
                    k = (bname, mkey, oname, pkey)
                    cur = rows.get(k)
                    try:
                        price = float(out.get("price"))
                    except Exception:
                        continue
                    if (cur is None) or (price > float(cur.get("price", 0.0))):
                        rows[k] = dict(out)

    ingest(base)
    ingest(extra)

    # Rebuild nested structure
    nested: Dict[Tuple[str, str], List[Dict]] = {}
    for (bname, mkey, _, _), out in rows.items():
        nested.setdefault((bname, mkey), []).append(out)

    bm_map: Dict[str, Dict] = {}
    for (bname, mkey), outcomes in nested.items():
        bm = bm_map.setdefault(bname, {"title": bname, "markets": []})
        bm["markets"].append({"key": mkey, "outcomes": outcomes})

    return list(bm_map.values())


def enrich_events_for_groups(events: List[Dict], league: str, desired_groups: Set[str]) -> Tuple[List[Dict], List[str]]:
    """
    For groups like corners/cards that are often unavailable in the initial fetch,
    re-fetch per-fixture odds from API-Football and merge new markets in.
    """
    if not desired_groups:
        return events, []

    try:
        from odds_provider import enrich_event_odds
    except ImportError:
        return events, ["odds_provider module not available for enrichment."]

    notes: List[str] = []
    for ev in events:
        existing_groups = event_market_groups(ev)
        missing = {g for g in desired_groups if g not in existing_groups}
        if not missing:
            continue

        all_keys, err = enrich_event_odds(ev, league)
        if err:
            notes.append(f"{ev.get('home_team')} vs {ev.get('away_team')}: enrichment failed ({err})")

    return events, notes


def totals_key_matches_group(market_key: str, stat_group: str) -> bool:
    k = (market_key or "").lower()
    if not is_full_game_market(k):
        return False
    if "total" not in k:
        return False
    if stat_group == "goals":
        if "corner" in k or "card" in k or "booking" in k:
            return False
        if "team_total" in k or "player" in k or "goalscorer" in k:
            return False
        return True
    if stat_group == "corners":
        return ("corner" in k)
    if stat_group == "cards":
        return ("card" in k or "booking" in k)
    return False


def enrich_events_for_totals_depth(events: List[Dict], league: str, stat_group: str = "goals") -> Tuple[List[Dict], List[str]]:
    """
    Expand totals markets even when 'totals' group already exists, by discovering
    event-level alternate totals keys and merging them into each event.
    """
    out: List[Dict] = []
    notes: List[str] = []
    for ev in events:
        event_id = str(ev.get("id") or "")
        if not event_id:
            out.append(ev)
            continue

        keys, err = discover_event_market_keys(league, event_id)
        if err:
            notes.append(f"{ev.get('home_team')} vs {ev.get('away_team')}: totals depth discovery failed ({err})")
            out.append(ev)
            continue

        wanted_keys = {k for k in keys if totals_key_matches_group(k, stat_group)}
        if not wanted_keys:
            notes.append(
                f"{ev.get('home_team')} vs {ev.get('away_team')}: no full-game {stat_group} totals keys returned by provider."
            )
            out.append(ev)
            continue

        # Keep request bounded while preserving a broad odds ladder.
        wanted_keys = set(sorted(wanted_keys)[:30])
        ev_payload, err2 = fetch_event_odds_for_market_keys(league, event_id, wanted_keys)
        if err2 or not ev_payload:
            notes.append(f"{ev.get('home_team')} vs {ev.get('away_team')}: totals depth fetch failed.")
            out.append(ev)
            continue

        merged = dict(ev)
        merged["bookmakers"] = merge_bookmakers(ev.get("bookmakers", []) or [], ev_payload.get("bookmakers", []) or [])
        out.append(merged)

    return out, notes


def filter_events_by_window(events: List[Dict], window: str, tz_name: str = "Europe/London") -> List[Dict]:
    if window == "upcoming":
        return events
    tz = ZoneInfo(tz_name)
    today = datetime.now(tz).date()
    target = today if window == "today" else today.fromordinal(today.toordinal() + 1)
    out: List[Dict] = []
    for ev in events:
        ts = ev.get("commence_time")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(tz)
        except Exception:
            continue
        if dt.date() == target:
            out.append(ev)
    return out


def filter_events_by_exact_date(events: List[Dict], target: date, tz_name: str = "Europe/London") -> List[Dict]:
    tz = ZoneInfo(tz_name)
    out: List[Dict] = []
    for ev in events:
        ts = ev.get("commence_time")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(tz)
        except Exception:
            continue
        if dt.date() == target:
            out.append(ev)
    return out


def event_fixtures_text(events: List[Dict]) -> str:
    if not events:
        return "No fixtures found."
    lines = []
    for i, ev in enumerate(events, start=1):
        lines.append(f"{i}. {ev.get('home_team')} vs {ev.get('away_team')} | {ev.get('commence_time')}")
    return "\n".join(lines)


def build_candidates(events: List[Dict]) -> List[CandidateLeg]:
    best_rows: Dict[Tuple[str, str, str, str], CandidateLeg] = {}
    for ev in events:
        event_id = str(ev.get("id") or "")
        home = str(ev.get("home_team") or "Home")
        away = str(ev.get("away_team") or "Away")
        fixture = f"{home} vs {away}"
        ev_league = ev.get("_league", "")
        for bm in ev.get("bookmakers", []) or []:
            bm_title = bm.get("title")
            for mk in bm.get("markets", []) or []:
                market_key = str(mk.get("key") or "")
                if not is_full_game_market(market_key):
                    continue
                # Exclude per-team totals from parlay pool — they have
                # trivially high probabilities (Over 2.5 home corners = 97%)
                # and add noise. Parlays should use match-level lines.
                if "team_total" in market_key.lower():
                    continue
                for out in mk.get("outcomes", []) or []:
                    name = str(out.get("name") or "")
                    if not name:
                        continue
                    point_raw = out.get("point")
                    try:
                        point = float(point_raw) if point_raw is not None else None
                    except Exception:
                        point = None
                    try:
                        odds = float(out.get("price"))
                    except Exception:
                        continue
                    if odds < MIN_LEG_ODDS:
                        continue
                    # Higher floor for totals-style markets to avoid trivially safe lines
                    grp = market_group_from_key(market_key)
                    if grp in {"totals", "corners", "cards", "sot"} and odds < MIN_TOTALS_LEG_ODDS:
                        continue
                    row = CandidateLeg(
                        event_id=event_id,
                        fixture=fixture,
                        home_team=home,
                        away_team=away,
                        market_key=market_key,
                        outcome=name,
                        odds=odds,
                        point=point,
                        bookmaker=bm_title,
                        league=ev_league,
                        event=ev,
                    )
                    k = (event_id, market_key, name, "" if point is None else str(point))
                    cur = best_rows.get(k)
                    if cur is None or row.odds > cur.odds:
                        best_rows[k] = row
    return list(best_rows.values())


def available_market_groups(candidates: List[CandidateLeg]) -> Set[str]:
    return {market_group_from_key(c.market_key) for c in candidates}


def contradictory(a: CandidateLeg, b: CandidateLeg) -> bool:
    if a.event_id != b.event_id:
        return False
    ga = market_group_from_key(a.market_key)
    gb = market_group_from_key(b.market_key)

    # One-leg-per-market-family per fixture.
    # This blocks duplicated exposure like:
    # - Over/Under corners in the same match
    # - multiple cards lines in the same match
    # - multiple spreads / multiple moneylines in the same match
    if ga == gb and ga in {"moneyline", "spreads", "totals", "corners", "cards", "sot", "btts"}:
        return True

    if (
        ga == gb
        and ga in {"totals", "spreads"}
        and a.point == b.point
        and a.outcome.lower() != b.outcome.lower()
    ):
        return True
    # If same market key + same point but opposite outcomes, treat as contradictory.
    if a.market_key == b.market_key and a.point == b.point and a.outcome.lower() != b.outcome.lower():
        return True

    def side_of_outcome(leg: CandidateLeg) -> Optional[str]:
        out = (leg.outcome or "").strip().lower()
        h = (leg.home_team or "").strip().lower()
        aw = (leg.away_team or "").strip().lower()
        if out == h:
            return "home"
        if out == aw:
            return "away"
        if out in {"draw", "tie", "x"}:
            return "draw"
        # Try prefix match for provider naming variants.
        if h and out.startswith(h):
            return "home"
        if aw and out.startswith(aw):
            return "away"
        return None

    side_a = side_of_outcome(a)
    side_b = side_of_outcome(b)

    # Equivalent result exposure:
    # ML(home) ~= spread(home -0.5), ML(away) ~= spread(away +0.5)
    if {ga, gb} == {"moneyline", "spreads"} and side_a in {"home", "away"} and side_b in {"home", "away"}:
        ml = a if ga == "moneyline" else b
        sp = b if ga == "moneyline" else a
        ml_side = side_of_outcome(ml)
        sp_side = side_of_outcome(sp)
        sp_point = sp.point
        if sp_point is not None and ml_side == sp_side:
            # Same-side spread near equivalent to ML:
            # -0.5 or 0.0 or +0.25 on selected side generally overlaps heavily with ML exposure.
            if abs(float(sp_point)) <= 0.5:
                return True
            # Also catch away +0.5 paired with away ML and home -0.5 paired with home ML.
            if (sp_side == "home" and sp_point <= -0.5) or (sp_side == "away" and sp_point >= 0.5):
                return True
    return False


def combo_valid(combo: List[CandidateLeg], require_unique_events: bool) -> bool:
    # Global product rule:
    # never allow ultra-short legs below configured minimum odds.
    if any(x.odds < MIN_LEG_ODDS for x in combo):
        return False

    if require_unique_events:
        ids = [x.event_id for x in combo]
        if len(ids) != len(set(ids)):
            return False

    # Global product rule:
    # Never mix moneyline and spread legs in the same parlay.
    groups = {market_group_from_key(x.market_key) for x in combo}
    if "moneyline" in groups and "spreads" in groups:
        return False

    for i in range(len(combo)):
        for j in range(i + 1, len(combo)):
            if contradictory(combo[i], combo[j]):
                return False
    return True


def heuristic_groups_for_event(home_meta: Dict, away_meta: Dict) -> Set[str]:
    if not home_meta or not away_meta:
        return set()
    groups: Set[str] = set()
    try:
        picks = top_markets(home_meta, away_meta, n=5)
    except Exception:
        picks = []
    for p in picks:
        label = str(p.get("market") or "").lower()
        if "corner" in label:
            groups.add("corners")
        elif "card" in label or "booking" in label:
            groups.add("cards")
        elif "shot" in label or "sot" in label:
            groups.add("sot")
        elif "btts" in label:
            groups.add("btts")
        elif "goal" in label or "under" in label or "over" in label:
            groups.add("totals")
    return groups


def _market_blended_proj(proj_total: float, event: Optional[Dict], stat_group: str) -> float:
    """Blend model projection with market-implied total when available.

    Uses SCORING_WEIGHTS["market_blend"] weights (default 70/30 model/market).
    Gracefully returns proj_total unchanged when event is None or market total
    is unavailable.
    """
    if event is None:
        return proj_total
    try:
        market_total = extract_market_implied_total(event, stat_group)
    except Exception:
        return proj_total
    if market_total is None:
        return proj_total
    mbw = SCORING_WEIGHTS["market_blend"]
    return mbw["model_weight"] * proj_total + mbw["market_weight"] * market_total


def kb_leg_quality(leg: CandidateLeg, league: str) -> float:
    """
    Positive score indicates stronger multi-signal support from local KB.
    """
    hm = _profile_meta(leg.home_team, league)
    am = _profile_meta(leg.away_team, league)
    h_recent = _recent_stats(leg.home_team, league, last_n=6)
    a_recent = _recent_stats(leg.away_team, league, last_n=6)
    heur_groups = heuristic_groups_for_event(hm, am)

    g = market_group_from_key(leg.market_key)
    score = 0.0

    h_ctrl = _numeric(hm.get("control_index"))
    a_ctrl = _numeric(am.get("control_index"))
    h_form = _numeric(hm.get("form_index_team"))
    a_form = _numeric(am.get("form_index_team"))
    h_dom = _numeric(hm.get("dominance_index"))
    a_dom = _numeric(am.get("dominance_index"))

    w = SCORING_WEIGHTS["kb_quality"]

    if g in heur_groups:
        score += w["heuristic_group_bonus"]

    if g == "moneyline":
        edge = 0.0
        home_sign = 1.0 if leg.outcome.lower().startswith(leg.home_team.lower()) else -1.0
        if h_form is not None and a_form is not None:
            edge += (h_form - a_form) * w["form_edge"] * home_sign
        if h_ctrl is not None and a_ctrl is not None:
            edge += (h_ctrl - a_ctrl) * w["control_edge"] * home_sign
        if h_dom is not None and a_dom is not None:
            edge += (h_dom - a_dom) * w["dominance_edge"] * home_sign
        h_sot = h_recent.get("sot_for_avg")
        a_sot = a_recent.get("sot_for_avg")
        if h_sot is not None and a_sot is not None:
            edge += (h_sot - a_sot) * w["sot_edge"] * home_sign
        score += edge
        # Probability-based moneyline scoring (consistent with count markets)
        try:
            ml_probs = projected_moneyline_probs(leg.home_team, leg.away_team, league)
            if ml_probs is not None:
                p_home, p_draw, p_away = ml_probs[:3]
                if p_home is not None and p_away is not None:
                    is_home = leg.outcome.lower().startswith(leg.home_team.lower())
                    model_p = p_home if is_home else p_away
                    prob_bonus = (model_p - 0.50) * SCORING_WEIGHTS["prob"]["prob_quality_weight"]
                    score += prob_bonus
        except Exception:
            pass

    if g == "spreads":
        proj_diff, _, _ = projected_goal_difference(leg.home_team, leg.away_team, league)
        if proj_diff is not None:
            is_home = leg.outcome.lower().startswith(leg.home_team.lower())
            sign = 1.0 if is_home else -1.0
            point = leg.point if leg.point is not None else 0.0
            edge = sign * proj_diff + point
            score += edge * w["spreads_edge_weight"]

    if g == "totals":
        goals_proj, _, _ = projected_total_goals(leg.home_team, leg.away_team, league)
        if goals_proj is not None:
            goals_proj = _market_blended_proj(goals_proj, leg.event, "totals")
            direction = leg.outcome.lower()
            if "over" in direction:
                score += goals_proj - w["goals_over_offset"]
            elif "under" in direction:
                score += w["goals_under_offset"] - goals_proj
            if leg.point is not None:
                if "over" in direction:
                    score += (goals_proj - leg.point) * w["goals_line_fit"]
                elif "under" in direction:
                    score += (leg.point - goals_proj) * w["goals_line_fit"]

    if g == "corners":
        corners_proj, _, _ = projected_total_corners(leg.home_team, leg.away_team, league)
        if corners_proj is not None:
            corners_proj = _market_blended_proj(corners_proj, leg.event, "corners")
            direction = leg.outcome.lower()
            if leg.point is not None and "total" in leg.market_key.lower():
                if "over" in direction:
                    score += (corners_proj - leg.point) * w["corners_line_fit"]
                elif "under" in direction:
                    score += (leg.point - corners_proj) * w["corners_line_fit"]
            # Side edge for corner-winner markets
            if leg.point is None or "total" not in leg.market_key.lower():
                h_proj_c, a_proj_c = projected_corners(hm, am)
                if h_proj_c is not None and a_proj_c is not None:
                    side_edge = h_proj_c - a_proj_c
                    h_recent_for = h_recent.get("corners_for_avg")
                    a_recent_for = a_recent.get("corners_for_avg")
                    if h_recent_for is not None and a_recent_for is not None:
                        recent_side = h_recent_for - a_recent_for
                        side_edge = 0.65 * side_edge + 0.35 * recent_side
                    if leg.outcome.lower().startswith(leg.home_team.lower()):
                        score += side_edge * w["corners_side_edge"]
                    elif leg.outcome.lower().startswith(leg.away_team.lower()):
                        score -= side_edge * w["corners_side_edge"]

    if g == "cards":
        cards_result = projected_total_cards(leg.home_team, leg.away_team, league)
        cards_proj = cards_result[0] if cards_result else None

        if cards_proj is not None:
            cards_proj = _market_blended_proj(cards_proj, leg.event, "cards")
            direction = leg.outcome.lower()
            if leg.point is not None and "total" in leg.market_key.lower():
                if "over" in direction:
                    score += (cards_proj - leg.point) * w["cards_line_fit"]
                elif "under" in direction:
                    score += (leg.point - cards_proj) * w["cards_line_fit"]

                # Foul momentum: recent fouls trending up → cards more likely
                h_fouls = _numeric(hm.get("fouls_per_90_team"))
                a_fouls = _numeric(am.get("fouls_per_90_team"))
                h_recent_fouls = h_recent.get("fouls_avg")
                a_recent_fouls = a_recent.get("fouls_avg")
                foul_momentum = 0.0
                n_mom = 0
                if h_fouls is not None and h_recent_fouls is not None:
                    foul_momentum += h_recent_fouls - h_fouls; n_mom += 1
                if a_fouls is not None and a_recent_fouls is not None:
                    foul_momentum += a_recent_fouls - a_fouls; n_mom += 1
                if n_mom > 0:
                    avg_mom = foul_momentum / n_mom
                    if "over" in direction:
                        score += avg_mom * w["cards_foul_momentum"]
                    elif "under" in direction:
                        score -= avg_mom * w["cards_foul_momentum"]

            # Side edge for card-winner markets
            if leg.point is None or "total" not in leg.market_key.lower():
                h_cards_proj, a_cards_proj = projected_cards(hm, am)
                if h_cards_proj is not None and a_cards_proj is not None:
                    side_edge = h_cards_proj - a_cards_proj
                    if leg.outcome.lower().startswith(leg.home_team.lower()):
                        score += side_edge * w["cards_side_edge"]
                    elif leg.outcome.lower().startswith(leg.away_team.lower()):
                        score -= side_edge * w["cards_side_edge"]
                # Card-induction side edge
                h_opp_induced = _numeric(hm.get("opp_cards_induced_pm"))
                a_opp_induced = _numeric(am.get("opp_cards_induced_pm"))
                if h_opp_induced is not None and a_opp_induced is not None:
                    induction_edge = a_opp_induced - h_opp_induced
                    if leg.outcome.lower().startswith(leg.home_team.lower()):
                        score += induction_edge * w["cards_opp_induced_w"]
                    elif leg.outcome.lower().startswith(leg.away_team.lower()):
                        score -= induction_edge * w["cards_opp_induced_w"]

    if g == "sot":
        sot_total, sot_season, sot_recent = projected_total_sot(leg.home_team, leg.away_team, league)
        if sot_total is not None:
            sot_total = _market_blended_proj(sot_total, leg.event, "sot")
            direction = leg.outcome.lower()
            if "over" in direction:
                score += sot_total - w["sot_over_offset"]
            elif "under" in direction:
                score += w["sot_under_offset"] - sot_total
            if leg.point is not None:
                if "over" in direction:
                    score += (sot_total - leg.point) * w["sot_line_fit"]
                elif "under" in direction:
                    score += (leg.point - sot_total) * w["sot_line_fit"]

    if g == "btts":
        result = projected_btts_prob(leg.home_team, leg.away_team, league)
        p_btts = result[0] if result else None
        if p_btts is not None:
            direction = leg.outcome.lower().strip()
            if direction == "yes":
                score += (p_btts - 0.50) * w.get("btts_quality_weight", 1.5)
            elif direction == "no":
                score += (0.50 - p_btts) * w.get("btts_quality_weight", 1.5)

    if leg.odds >= w["long_odds_threshold"]:
        score -= (leg.odds - w["long_odds_offset"]) * w["long_odds_penalty"]

    # --- Group preference multiplier ---
    group_pref = SCORING_WEIGHTS["group_preference"].get(g, 1.0)
    score *= group_pref

    # --- Elo signal for count-based markets ---
    if g in ("corners", "cards", "sot", "totals"):
        stat_map = {"corners": "corners", "cards": "cards", "sot": "sot", "totals": "goals"}
        mw = SCORING_WEIGHTS["ml"]
        elo_diff = get_elo_edge(leg.home_team, leg.away_team, stat_map[g])
        if elo_diff is not None and abs(elo_diff) >= mw["min_elo_diff"]:
            elo_norm = elo_diff / mw["elo_scale"]
            direction = leg.outcome.lower()
            home_outcome = direction.startswith(leg.home_team.lower()) if hasattr(leg, 'home_team') else False
            if "over" in direction:
                # Positive elo_norm means home stronger → higher total expected → supports over
                score += elo_norm * mw["elo_signal_weight"]
            elif "under" in direction:
                # Negative elo_norm means away stronger → lower total expected → supports under
                score -= elo_norm * mw["elo_signal_weight"]
            elif home_outcome:
                score += elo_norm * mw["elo_signal_weight"]
            else:
                score -= elo_norm * mw["elo_signal_weight"]

    # --- Probability-aware quality bonus for count-based markets ---
    if leg.point is not None and g in ("corners", "cards", "sot", "totals"):
        mk_lower = leg.market_key.lower()
        is_team_total = "team_total" in mk_lower

        if is_team_total:
            # Per-team line — use team-specific projection, not match total
            # Determine which team this line is for from the market key
            is_home_team = "home" in mk_lower
            if g == "corners":
                team_proj, _, _ = projected_team_corners(
                    leg.home_team if is_home_team else leg.away_team,
                    leg.away_team if is_home_team else leg.home_team,
                    league, is_home=is_home_team,
                )
                proj_total = team_proj
            elif g == "cards":
                team_proj, _, _ = projected_team_cards(
                    leg.home_team if is_home_team else leg.away_team,
                    leg.away_team if is_home_team else leg.home_team,
                    league, is_home=is_home_team,
                )
                proj_total = team_proj
            elif g == "totals":
                # team_totals_home_goals / team_totals_away_goals
                h_proj, a_proj = projected_goals(hm, am)
                proj_total = h_proj if is_home_team else a_proj
            else:
                proj_total = None
        else:
            # Match-level total — use match total projection
            proj_funcs = {
                "corners": lambda: projected_total_corners(leg.home_team, leg.away_team, league),
                "cards": lambda: projected_total_cards(leg.home_team, leg.away_team, league),
                "sot": lambda: projected_total_sot(leg.home_team, leg.away_team, league),
                "totals": lambda: projected_total_goals(leg.home_team, leg.away_team, league),
            }
            proj_result = proj_funcs[g]()
            proj_total = proj_result[0] if proj_result else None
            # Apply market blend for match-level projections
            if proj_total is not None:
                proj_total = _market_blended_proj(proj_total, leg.event, g if g != "totals" else "totals")

        if proj_total is not None:
            direction = leg.outcome.lower()
            is_over = "over" in direction
            # Compute combined variance for Poisson/NegBin distribution selection
            h_var = get_team_recent_variance(leg.home_team, league)
            a_var = get_team_recent_variance(leg.away_team, league)
            _stat_var_key = {"corners": "corners_for_var", "cards": "cards_var",
                             "sot": "sot_for_var", "totals": "goals_var"}[g]
            h_v = h_var.get(_stat_var_key)
            a_v = a_var.get(_stat_var_key)
            combined_var = (h_v + a_v) if (h_v is not None and a_v is not None) else None
            model_p = (over_prob(proj_total, leg.point, combined_var) if is_over
                       else under_prob(proj_total, leg.point, combined_var))
            prob_bonus = (model_p - 0.50) * SCORING_WEIGHTS["prob"]["prob_quality_weight"]
            score += prob_bonus

    return score


def score_combo(combo: List[CandidateLeg], c: ConstraintSpec, leg_quality: Dict[CandidateLeg, float]) -> float:
    w = SCORING_WEIGHTS["combo"]
    combo_odds = 1.0
    for leg in combo:
        combo_odds *= leg.odds

    # Base scoring
    if c.target_multiplier and c.target_mode == "around":
        score = abs(log(combo_odds) - log(c.target_multiplier)) * w["around_target_log_penalty"]
    else:
        neutral_anchor = w["neutral_anchor_base"] ** max(1, len(combo))
        score = abs(log(combo_odds) - log(neutral_anchor)) * w["neutral_anchor_deviation"]

    if c.target_multiplier and c.target_multiplier > 0:
        tgt = c.target_multiplier
        if c.target_mode == "max":
            score += abs(combo_odds - tgt) * w["target_abs_deviation"]
            if combo_odds > tgt:
                score += (combo_odds - tgt) * w["target_over_limit_penalty"]
        elif c.target_mode == "min":
            score += abs(combo_odds - tgt) * w["target_abs_deviation"]
            if combo_odds < tgt:
                score += (tgt - combo_odds) * w["target_over_limit_penalty"]
        else:
            rel_dev = abs(combo_odds - tgt) / max(tgt, 1e-9)
            if rel_dev > AROUND_TARGET_TOLERANCE:
                score += (rel_dev - AROUND_TARGET_TOLERANCE) * w["around_over_tolerance_penalty"]

        desired_leg = tgt ** (1.0 / max(1, len(combo)))
        for leg in combo:
            if c.target_mode in {"around", "min"} and leg.odds < desired_leg * w["desired_leg_floor_ratio"]:
                score += (desired_leg * w["desired_leg_floor_ratio"] - leg.odds) * w["desired_leg_under_penalty"]

    # --- Odds range constraint ---
    if c.target_min is not None and c.target_max is not None:
        if not c.target_multiplier:
            # Use range midpoint as soft anchor (half-strength penalty within range)
            range_mid = (c.target_min + c.target_max) / 2.0
            score = abs(log(combo_odds) - log(range_mid)) * w["around_target_log_penalty"] * 0.5
        if combo_odds < c.target_min:
            score += (c.target_min - combo_odds) * w["target_over_limit_penalty"]
        elif combo_odds > c.target_max:
            score += (combo_odds - c.target_max) * w["target_over_limit_penalty"]

    # Soft preferences
    if c.soft_prefer_groups:
        for leg in combo:
            g = market_group_from_key(leg.market_key)
            if g not in c.soft_prefer_groups:
                score += w["soft_prefer_miss"]

    # Global anti-conservative guardrail.
    for leg in combo:
        if leg.odds < w["conservative_threshold"]:
            score += (w["conservative_threshold"] - leg.odds) * w["conservative_penalty"]

    # Prefer higher multi-signal support from local KB.
    # Lower score = better: subtract quality so high-quality combos rank higher.
    quality = sum(leg_quality.get(leg, 0.0) for leg in combo)
    score -= quality * w["kb_quality_weight"]

    # --- EV-aware scoring: reward positive-EV combos ---
    pw = SCORING_WEIGHTS["prob"]
    joint_prob = 1.0
    any_prob = False
    for leg in combo:
        g = market_group_from_key(leg.market_key)
        if leg.point is not None and g in ("corners", "cards", "sot", "totals"):
            leg_lg = _leg_league(leg, c.league)
            mk_lower = leg.market_key.lower()
            is_team_total = "team_total" in mk_lower

            if is_team_total:
                # Per-team line — use team projection
                is_home_team = "home" in mk_lower
                _hm = _profile_meta(leg.home_team, leg_lg)
                _am = _profile_meta(leg.away_team, leg_lg)
                if g == "corners":
                    tp, _, _ = projected_team_corners(
                        leg.home_team if is_home_team else leg.away_team,
                        leg.away_team if is_home_team else leg.home_team,
                        leg_lg, is_home=is_home_team)
                    proj_total = tp
                elif g == "cards":
                    tp, _, _ = projected_team_cards(
                        leg.home_team if is_home_team else leg.away_team,
                        leg.away_team if is_home_team else leg.home_team,
                        leg_lg, is_home=is_home_team)
                    proj_total = tp
                elif g == "totals":
                    h_proj, a_proj = projected_goals(_hm, _am)
                    proj_total = h_proj if is_home_team else a_proj
                else:
                    proj_total = None
            else:
                # Match total
                proj_funcs = {
                    "corners": lambda l=leg, lg=leg_lg: projected_total_corners(l.home_team, l.away_team, lg),
                    "cards": lambda l=leg, lg=leg_lg: projected_total_cards(l.home_team, l.away_team, lg),
                    "sot": lambda l=leg, lg=leg_lg: projected_total_sot(l.home_team, l.away_team, lg),
                    "totals": lambda l=leg, lg=leg_lg: projected_total_goals(l.home_team, l.away_team, lg),
                }
                proj_result = proj_funcs[g]()
                proj_total = proj_result[0] if proj_result else None
                # Apply market blend for match-level projections
                if proj_total is not None:
                    proj_total = _market_blended_proj(proj_total, leg.event, g if g != "totals" else "totals")

            if proj_total is not None:
                is_over = "over" in leg.outcome.lower()
                # Use variance for Poisson/NegBin distribution selection (consistent
                # with kb_leg_quality probability scoring)
                _stat_var_key = {"corners": "corners_for_var", "cards": "cards_var",
                                 "sot": "sot_for_var", "totals": "goals_var"}[g]
                h_var = get_team_recent_variance(leg.home_team, leg_lg)
                a_var = get_team_recent_variance(leg.away_team, leg_lg)
                h_v = h_var.get(_stat_var_key)
                a_v = a_var.get(_stat_var_key)
                combined_var = (h_v + a_v) if (h_v is not None and a_v is not None) else None
                model_p = (over_prob(proj_total, leg.point, combined_var) if is_over
                           else under_prob(proj_total, leg.point, combined_var))
                joint_prob *= model_p
                any_prob = True
            else:
                joint_prob *= 0.5
        else:
            joint_prob *= 0.5
    if any_prob:
        combo_ev = (joint_prob * combo_odds) - 1.0
        score -= combo_ev * pw["ev_combo_weight"]

    # Diversity penalty for same-event, same-thesis stacking beyond contradictions.
    for i in range(len(combo)):
        for j in range(i + 1, len(combo)):
            a = combo[i]
            b = combo[j]
            if a.event_id != b.event_id:
                continue
            ga = market_group_from_key(a.market_key)
            gb = market_group_from_key(b.market_key)
            if {ga, gb} == {"moneyline", "spreads"}:
                score += w["ml_spread_stack_penalty"]

    # --- Market group diversity ---
    dw = SCORING_WEIGHTS["diversity"]
    group_counts = Counter(market_group_from_key(leg.market_key) for leg in combo)
    n_legs = len(combo)
    max_count = max(group_counts.values())
    concentration = max_count / n_legs
    if concentration > dw["concentration_threshold"]:
        score += (concentration - dw["concentration_threshold"]) * dw["concentration_penalty"] * n_legs
    for _grp, cnt in group_counts.items():
        if cnt > dw["max_same_group"]:
            score += (cnt - dw["max_same_group"]) * dw["excess_same_penalty"]
    unique_groups = len(group_counts)
    score -= unique_groups * dw["unique_group_bonus"]

    # --- League diversity (cross-league mode) ---
    if c.leagues:
        league_counts = Counter(_leg_league(leg, c.league) for leg in combo)
        unique_leagues = len(league_counts)
        score -= unique_leagues * dw.get("unique_league_bonus", 0.3)
        max_league_count = max(league_counts.values())
        if max_league_count > n_legs * 0.6:
            score += (max_league_count - n_legs * 0.6) * dw.get("league_concentration_penalty", 2.0)

    # Hard excludes should be strongly avoided.
    for leg in combo:
        g = market_group_from_key(leg.market_key)
        if g in c.hard_exclude_groups:
            score += w["hard_exclude_penalty"]
        if c.forbid_spread_keys and (("spread" in leg.market_key.lower()) or ("handicap" in leg.market_key.lower())):
            score += w["forbid_spread_penalty"]
        if c.require_total_corner_keys:
            k = leg.market_key.lower()
            if ("corner" not in k) or ("total" not in k):
                score += w["require_total_corner_penalty"]

    # Hard includes should appear when available.
    for req in c.hard_include_groups:
        if not any(market_group_from_key(x.market_key) == req for x in combo):
            score += w["hard_include_missing"]

    # Composition constraints (e.g., 1 corner leg + 1 booking leg)
    for grp, need in c.required_group_counts.items():
        have = sum(1 for x in combo if market_group_from_key(x.market_key) == grp)
        if have < need:
            score += (need - have) * w["composition_missing"]

    return score


def check_feasibility(candidates: List[CandidateLeg], leg_count: int, c: ConstraintSpec) -> Tuple[bool, List[str]]:
    """
    Pre-check whether constraints are satisfiable before the expensive solver.
    Returns (feasible, warnings). Non-feasible means the solver would waste time.
    """
    warnings: List[str] = []
    if leg_count <= 0:
        return False, ["Invalid leg count."]
    if not candidates:
        return False, ["No odds candidates available."]

    avail_groups = {market_group_from_key(c_leg.market_key) for c_leg in candidates}
    unique_events = len({c_leg.event_id for c_leg in candidates})

    if c.require_unique_events and unique_events < leg_count:
        warnings.append(
            f"Requested {leg_count} legs across unique fixtures, but only {unique_events} fixture(s) have odds. "
            f"Reducing leg count to {unique_events}."
        )

    for grp in c.hard_include_groups:
        if grp not in avail_groups:
            warnings.append(
                f"Requested '{grp}' markets, but none are available from the current odds feed. "
                f"Available groups: {', '.join(sorted(avail_groups)) or 'none'}."
            )

    for grp, need in c.required_group_counts.items():
        grp_legs = [c_leg for c_leg in candidates if market_group_from_key(c_leg.market_key) == grp]
        if len(grp_legs) < need:
            warnings.append(
                f"Requested {need} '{grp}' leg(s), but only {len(grp_legs)} candidate(s) available."
            )

    if c.require_total_corner_keys:
        corner_total_legs = [
            c_leg for c_leg in candidates
            if "corner" in c_leg.market_key.lower() and "total" in c_leg.market_key.lower()
        ]
        if not corner_total_legs:
            warnings.append(
                "Total-corners-only requested, but no full-game total corner market keys found in feed."
            )

    feasible = not any(w.startswith("Requested") and "none are available" in w for w in warnings)
    return feasible, warnings


def _brute_force_combos(
    pool: List[CandidateLeg], leg_count: int, c: ConstraintSpec, leg_quality: Dict[CandidateLeg, float]
) -> List[Tuple[List[CandidateLeg], float, float]]:
    scored: List[Tuple[List[CandidateLeg], float, float]] = []
    for combo in combinations(pool, leg_count):
        combo_list = list(combo)
        if not combo_valid(combo_list, c.require_unique_events):
            continue
        s = score_combo(combo_list, c, leg_quality)
        scored.append((combo_list, s, combined_odds(combo_list)))
    return scored


def _beam_search_combos(
    pool: List[CandidateLeg], leg_count: int, c: ConstraintSpec, leg_quality: Dict[CandidateLeg, float]
) -> List[Tuple[List[CandidateLeg], float, float]]:
    """Greedy beam search: O(beam_width * pool_size * leg_count)."""
    pw = SCORING_WEIGHTS["pool"]
    cw = SCORING_WEIGHTS["combo"]
    beam_width = int(pw["beam_width"])

    # Seed beams with individual legs
    seeds: List[Tuple[List[CandidateLeg], float]] = []
    for leg in pool:
        if leg.odds < MIN_LEG_ODDS:
            continue
        seeds.append(([leg], -leg_quality.get(leg, 0.0)))
    seeds.sort(key=lambda x: x[1])
    beam = seeds[:beam_width]

    for step in range(1, leg_count):
        next_beam: List[Tuple[List[CandidateLeg], float]] = []
        seen_keys: Set[frozenset] = set()
        for combo, _ in beam:
            combo_set = set(id(l) for l in combo)
            for leg in pool:
                if id(leg) in combo_set:
                    continue
                extended = combo + [leg]
                if not combo_valid(extended, c.require_unique_events):
                    continue
                # Deduplicate by leg identity
                key = frozenset((l.event_id, l.market_key, l.outcome, l.point) for l in extended)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                # Heuristic for partial combos: proximity to target + KB quality
                partial_quality = sum(leg_quality.get(l, 0.0) for l in extended)
                combo_odds = 1.0
                for l in extended:
                    combo_odds *= l.odds
                if c.target_multiplier and c.target_multiplier > 0:
                    target_for_step = c.target_multiplier ** ((step + 1) / leg_count)
                    proximity = abs(log(max(1.01, combo_odds)) - log(max(1.01, target_for_step)))
                else:
                    neutral = cw["neutral_anchor_base"] ** (step + 1)
                    proximity = abs(log(max(1.01, combo_odds)) - log(neutral))
                h_score = proximity - partial_quality * 0.3
                next_beam.append((extended, h_score))
        if not next_beam:
            break
        next_beam.sort(key=lambda x: x[1])
        beam = next_beam[:beam_width]

    # Final scoring of complete combos
    scored: List[Tuple[List[CandidateLeg], float, float]] = []
    for combo, _ in beam:
        if len(combo) != leg_count:
            continue
        if not combo_valid(combo, c.require_unique_events):
            continue
        s = score_combo(combo, c, leg_quality)
        scored.append((combo, s, combined_odds(combo)))
    return scored


def select_parlay(candidates: List[CandidateLeg], leg_count: int, c: ConstraintSpec) -> Tuple[List[CandidateLeg], List[str]]:
    notes: List[str] = []
    if leg_count <= 0:
        return [], ["Invalid leg count."]
    if not candidates:
        return [], ["No odds candidates available."]

    # ---- Hard pre-filter: remove legs that violate non-negotiable constraints ----
    # This prevents the solver from ever considering invalid legs.
    pre_filter_count = len(candidates)

    # Minimum odds floor for parlay legs
    _PARLAY_MIN_ODDS = 1.30
    candidates = [leg for leg in candidates if leg.odds >= _PARLAY_MIN_ODDS]

    # require_total_corner_keys: only keep legs whose market key contains both "corner" and "total"
    if c.require_total_corner_keys:
        candidates = [
            leg for leg in candidates
            if "corner" in leg.market_key.lower() and "total" in leg.market_key.lower()
        ]

    # hard_exclude_groups: remove legs in excluded market groups
    if c.hard_exclude_groups:
        candidates = [
            leg for leg in candidates
            if market_group_from_key(leg.market_key) not in c.hard_exclude_groups
        ]

    # hard_include_groups: if set, only keep legs in included groups
    if c.hard_include_groups:
        candidates = [
            leg for leg in candidates
            if market_group_from_key(leg.market_key) in c.hard_include_groups
        ]

    # forbid_spread_keys: remove spread/handicap legs
    if c.forbid_spread_keys:
        candidates = [
            leg for leg in candidates
            if "spread" not in leg.market_key.lower() and "handicap" not in leg.market_key.lower()
        ]

    filtered_out = pre_filter_count - len(candidates)
    if filtered_out > 0:
        notes.append(f"Hard-filtered {filtered_out} legs that violate market constraints ({len(candidates)} remaining).")

    if not candidates:
        return [], ["No candidates remain after applying market constraints (hard filter)."]

    # Quality gate: compute KB quality for all candidates and reject legs
    # where the model disagrees with the bet (negative quality).
    _pre_quality_count = len(candidates)
    _quality_map: Dict[CandidateLeg, float] = {}
    for leg in candidates:
        _quality_map[leg] = kb_leg_quality(leg, _leg_league(leg, c.league))
    min_q = SCORING_WEIGHTS["pool"].get("min_kb_quality", 0.0)
    candidates = [leg for leg in candidates if _quality_map[leg] > min_q]
    _quality_dropped = _pre_quality_count - len(candidates)
    if _quality_dropped > 0:
        notes.append(f"Quality-filtered {_quality_dropped} negative-signal legs ({len(candidates)} remaining).")

    if not candidates:
        return [], ["No candidates with positive model signal remain after quality filter."]

    # Reduce search size while keeping odds diversity.
    by_bucket: Dict[Tuple[str, str], List[CandidateLeg]] = {}
    for row in sorted(candidates, key=lambda x: x.odds):
        k = (row.event_id, row.market_key)
        by_bucket.setdefault(k, [])
        by_bucket[k].append(row)

    target_leg_anchor = None
    if c.target_multiplier and c.target_multiplier > 0:
        target_leg_anchor = c.target_multiplier ** (1.0 / max(1, leg_count))
    if target_leg_anchor is None:
        target_leg_anchor = SCORING_WEIGHTS["pool"]["default_anchor"]

    pool: List[CandidateLeg] = []
    for rows in by_bucket.values():
        n = len(rows)
        if n <= 8:
            pool.extend(rows)
            continue
        # Pick representative low/mid/high plus lines near target leg odds.
        picks: List[CandidateLeg] = []
        picks.extend(rows[:2])               # low odds
        picks.extend(rows[-2:])              # high odds
        rows_by_anchor = sorted(rows, key=lambda x: abs(log(max(1.01, x.odds)) - log(max(1.01, target_leg_anchor))))
        picks.extend(rows_by_anchor[:4])     # around target odds
        seen = set()
        dedup: List[CandidateLeg] = []
        for p in picks:
            key = (p.event_id, p.market_key, p.outcome, p.point, p.bookmaker, p.odds)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(p)
        pool.extend(dedup)

    # Final cap by proximity-to-target odds + KB quality, not raw low odds.
    # Reuse quality scores from the pre-filter where available.
    leg_quality: Dict[CandidateLeg, float] = {}
    for leg in pool:
        leg_quality[leg] = _quality_map.get(leg) if leg in _quality_map else kb_leg_quality(leg, _leg_league(leg, c.league))

    pw = SCORING_WEIGHTS["pool"]

    def leg_rank(leg: CandidateLeg) -> float:
        proximity = abs(log(max(1.01, leg.odds)) - log(max(1.01, target_leg_anchor)))
        quality_bonus = pw["quality_bonus"] * leg_quality.get(leg, 0.0)
        return proximity - quality_bonus

    cap = int(pw["cap"])
    if c.leagues and len(c.leagues) > 1:
        cap = min(cap * 2, 400)
    pool = sorted(pool, key=leg_rank)[:cap]

    if len(pool) < leg_count:
        return [], [f"Only {len(pool)} candidate legs available for {leg_count}-leg request."]

    total_combos = _comb(len(pool), leg_count)
    threshold = SCORING_WEIGHTS["pool"]["brute_force_threshold"]
    if total_combos <= threshold:
        scored = _brute_force_combos(pool, leg_count, c, leg_quality)
    else:
        scored = _beam_search_combos(pool, leg_count, c, leg_quality)
        notes.append(f"Used beam search ({len(pool)} candidates, {leg_count} legs, ~{total_combos:,} combos avoided).")

    if not scored:
        return [], ["No valid parlay combination found."]

    if c.target_multiplier and c.target_mode in {"around", "max", "min"}:
        tgt = c.target_multiplier
        if c.target_mode == "around":
            band = AROUND_TARGET_TOLERANCE * tgt
            in_band = [row for row in scored if abs(row[2] - tgt) <= band]
            if in_band:
                best = min(in_band, key=lambda x: x[1])
                return best[0], notes
            best = min(scored, key=lambda x: abs(log(x[2]) - log(tgt)))
            notes.append(
                f"No combo within ±{int(AROUND_TARGET_TOLERANCE*100)}% of target {tgt:.2f}x; returning closest at {best[2]:.2f}x."
            )
            return best[0], notes

        if c.target_mode == "max":
            feasible = [row for row in scored if row[2] <= tgt]
            if feasible:
                best = min(feasible, key=lambda x: x[1])
                return best[0], notes
            best = min(scored, key=lambda x: abs(x[2] - tgt))
            notes.append(f"No combo <= {tgt:.2f}x; returning closest at {best[2]:.2f}x.")
            return best[0], notes

        if c.target_mode == "min":
            feasible = [row for row in scored if row[2] >= tgt]
            if feasible:
                best = min(feasible, key=lambda x: x[1])
                return best[0], notes
            best = min(scored, key=lambda x: abs(x[2] - tgt))
            notes.append(f"No combo >= {tgt:.2f}x; returning closest at {best[2]:.2f}x.")
            return best[0], notes

    # --- Odds range mode ---
    if c.target_min is not None and c.target_max is not None:
        in_range = [row for row in scored if c.target_min <= row[2] <= c.target_max]
        if in_range:
            best = min(in_range, key=lambda x: x[1])
            return best[0], notes
        best = min(scored, key=lambda x: min(abs(x[2] - c.target_min), abs(x[2] - c.target_max)))
        notes.append(
            f"No combo within odds range [{c.target_min:.1f}x, {c.target_max:.1f}x]; "
            f"returning closest at {best[2]:.2f}x."
        )
        return best[0], notes

    # default: best score
    best = min(scored, key=lambda x: x[1])
    return best[0], notes


def _kb_team_variants(team_name: str) -> List[str]:
    """Generate team name variants for KB lookup: exact → alias → fuzzy."""
    canonical = canonical_team_name(team_name)
    variants = list({team_name, canonical, team_name.lower(), canonical.lower(), team_name.title(), canonical.title()})
    # Add token-stripped variants (e.g., "AFC Bournemouth" → "Bournemouth")
    for suffix in [" united", " hotspur", " fc", " afc", " cf", " city"]:
        for base in [team_name.lower(), canonical.lower()]:
            if base.endswith(suffix):
                stripped = base[: -len(suffix)].strip().title()
                if stripped and stripped not in variants:
                    variants.append(stripped)
    for prefix in ["afc ", "fc "]:
        for base in [team_name.lower(), canonical.lower()]:
            if base.startswith(prefix):
                stripped = base[len(prefix):].strip().title()
                if stripped and stripped not in variants:
                    variants.append(stripped)
    return variants


def get_team_profile_doc(team_name: str, league: str) -> Dict:
    cache_key = (league, canonical_team_name(team_name).lower())
    cached = _team_profile_doc_cache.get(cache_key)
    if cached is not None:
        return cached

    col = get_collection_handle(create_if_missing=True)
    variants = _kb_team_variants(team_name)
    rows: List[Dict] = []
    for v in variants:
        where = build_where(league, extra_filters=[{"doc_type": "team_profile"}, {"team": v}])
        res = col.get(where=where, include=["documents", "metadatas"], limit=6)
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        for d, m in zip(docs, metas):
            if not m:
                continue
            rows.append({"text": d or "", "meta": m})
    if not rows:
        _team_profile_doc_cache[cache_key] = {}
        return {}

    rows.sort(key=lambda x: parse_season_rank((x.get("meta") or {}).get("season")), reverse=True)
    best = rows[0]
    _team_profile_doc_cache[cache_key] = best
    return best


def get_team_profile_meta(team_name: str, league: str) -> Dict:
    cache_key = (league, canonical_team_name(team_name).lower())
    cached = _team_meta_cache.get(cache_key)
    if cached is not None:
        return cached
    doc = get_team_profile_doc(team_name, league)
    meta = (doc.get("meta") or {}) if doc else {}
    _team_meta_cache[cache_key] = meta
    return meta


# ---------------------------------------------------------------------------
# European competition: domestic-anchored projection helpers
# ---------------------------------------------------------------------------
_domestic_league_cache: Dict[str, Optional[str]] = {}


def resolve_domestic_league(team_name: str) -> Optional[str]:
    """Find which domestic league a team belongs to by searching Chroma profiles."""
    canon = canonical_team_name(team_name).lower()
    if canon in _domestic_league_cache:
        return _domestic_league_cache[canon]
    for lg in sorted(DOMESTIC_LEAGUES):
        meta = get_team_profile_meta(team_name, lg)
        if meta:
            _domestic_league_cache[canon] = lg
            return lg
    _domestic_league_cache[canon] = None
    return None


def get_blended_profile_meta(team_name: str, competition: str) -> Dict:
    """For European competitions, blend domestic (80%) + European (20%) profiles.
    For domestic leagues, returns the standard profile unchanged."""
    if competition not in EUROPEAN_COMPETITIONS:
        return get_team_profile_meta(team_name, competition)

    ew = SCORING_WEIGHTS["european"]
    euro_meta = get_team_profile_meta(team_name, competition)
    domestic_lg = resolve_domestic_league(team_name)

    if not domestic_lg:
        return euro_meta  # Non-top-5 team — European data only

    domestic_meta = get_team_profile_meta(team_name, domestic_lg)
    if not domestic_meta:
        return euro_meta
    if not euro_meta:
        return domestic_meta

    # Need enough European fixtures before blending is meaningful
    euro_n = _numeric(euro_meta.get("matches_played")) or 0
    if euro_n < ew["min_euro_fixtures"]:
        return domestic_meta  # Not enough European data yet — domestic only

    # Weighted blend of all numeric fields
    blended = dict(domestic_meta)  # start with domestic as base
    d_w, e_w = ew["domestic_weight"], ew["euro_weight"]
    for key, d_val in domestic_meta.items():
        e_val = euro_meta.get(key)
        if isinstance(d_val, (int, float)) and isinstance(e_val, (int, float)):
            blended[key] = d_w * d_val + e_w * e_val
    blended["_domestic_league"] = domestic_lg
    blended["_euro_competition"] = competition
    blended["_blend_mode"] = "domestic_anchored"
    return blended


def get_blended_recent_stats(team_name: str, competition: str, last_n: int = 6,
                             venue: Optional[str] = None) -> Dict:
    """For European competitions, blend domestic + European recent stats."""
    if competition not in EUROPEAN_COMPETITIONS:
        return get_team_recent_stats(team_name, competition, last_n, venue=venue)

    ew = SCORING_WEIGHTS["european"]
    euro_stats = get_team_recent_stats(team_name, competition, last_n, venue=venue)
    domestic_lg = resolve_domestic_league(team_name)

    if not domestic_lg:
        return euro_stats  # Non-top-5 team

    domestic_stats = get_team_recent_stats(team_name, domestic_lg, last_n, venue=venue)
    if not domestic_stats or domestic_stats.get("n", 0) == 0:
        return euro_stats
    if not euro_stats or euro_stats.get("n", 0) == 0:
        return domestic_stats

    # Weighted blend
    blended: Dict = {}
    d_w, e_w = ew["domestic_weight"], ew["euro_weight"]
    all_keys = set(list(domestic_stats.keys()) + list(euro_stats.keys()))
    for key in all_keys:
        d_val = domestic_stats.get(key)
        e_val = euro_stats.get(key)
        if isinstance(d_val, (int, float)) and isinstance(e_val, (int, float)):
            blended[key] = d_w * d_val + e_w * e_val
        elif d_val is not None:
            blended[key] = d_val
        else:
            blended[key] = e_val
    blended["n"] = domestic_stats.get("n", 0) + euro_stats.get("n", 0)
    return blended


def _profile_meta(team: str, league: str) -> Dict:
    """Profile metadata — auto-blends for European competitions."""
    if league in EUROPEAN_COMPETITIONS:
        return get_blended_profile_meta(team, league)
    return get_team_profile_meta(team, league)


def _recent_stats(team: str, league: str, last_n: int = 6,
                  venue: Optional[str] = None) -> Dict:
    """Recent stats — auto-blends for European competitions."""
    if league in EUROPEAN_COMPETITIONS:
        return get_blended_recent_stats(team, league, last_n, venue=venue)
    return get_team_recent_stats(team, league, last_n, venue=venue)


def get_recent_team_fixture_rows(team_name: str, league: str, limit: int = 8,
                                  season: Optional[str] = None) -> List[Dict]:
    col = get_collection_handle(create_if_missing=True)
    variants = _kb_team_variants(team_name)

    # Derive current season from team profile to avoid cross-season contamination.
    # Try profile season first, then check if fixtures actually exist for that season.
    target_season = season
    if target_season is None:
        profile = get_team_profile_doc(team_name, league)
        target_season = (profile.get("meta") or {}).get("season")

    def _fetch(season_filter: Optional[str]) -> Dict[str, Dict]:
        dedup: Dict[str, Dict] = {}
        for v in variants:
            filters: List[Dict] = [{"doc_type": "team_fixture"}, {"team": v}]
            if season_filter:
                filters.append({"season": season_filter})
            where = build_where(league, extra_filters=filters)
            res = col.get(where=where, include=["metadatas", "documents"], limit=50)
            docs = res.get("documents") or []
            metas = res.get("metadatas") or []
            for d, m in zip(docs, metas):
                if not m:
                    continue
                fixture = str(m.get("fixture") or "")
                fixture_date = str(m.get("fixture_date") or "")
                key = f"{fixture_date}|{fixture}"
                dedup[key] = {"meta": m, "text": d or ""}
        return dedup

    dedup = _fetch(target_season)

    # Fallback: if profile season yields no fixtures (season mismatch), retry without
    if not dedup and target_season:
        dedup = _fetch(None)

    rows = list(dedup.values())
    rows.sort(key=lambda x: ((x.get("meta") or {}).get("fixture_date") or ""), reverse=True)
    return rows[:limit]


def _get_team_fixture_meta(team_name: str, league: str, fixture: str,
                           fixture_date: str = "", season: Optional[str] = None) -> Optional[Dict]:
    """Fetch one team_fixture metadata row for a specific team+fixture."""
    cache_key = (
        league,
        canonical_team_name(team_name).lower(),
        str(fixture or ""),
        str(fixture_date or ""),
        str(season or ""),
    )
    if cache_key in _team_fixture_meta_cache:
        return _team_fixture_meta_cache[cache_key]

    col = get_collection_handle(create_if_missing=True)
    variants = _kb_team_variants(team_name)
    for v in variants:
        filters: List[Dict] = [{"doc_type": "team_fixture"}, {"team": v}, {"fixture": fixture}]
        if fixture_date:
            filters.append({"fixture_date": fixture_date})
        if season:
            filters.append({"season": season})
        where = build_where(league, extra_filters=filters)
        res = col.get(where=where, include=["metadatas"], limit=4)
        metas = res.get("metadatas") or []
        if metas:
            meta = metas[0] or None
            _team_fixture_meta_cache[cache_key] = meta
            return meta

    _team_fixture_meta_cache[cache_key] = None
    return None


def get_team_recent_stats(team_name: str, league: str, last_n: int = 6,
                          venue: Optional[str] = None) -> Dict:
    cache_key = (league, canonical_team_name(team_name).lower(), int(last_n),
                 venue or "all")
    cached = _team_recent_stats_cache.get(cache_key)
    if cached is not None:
        return cached

    # Fetch more rows when venue-filtering so we still get enough after filtering
    fetch_limit = max(last_n * 3, 12) if venue else max(last_n, 4)
    rows = get_recent_team_fixture_rows(team_name, league, limit=fetch_limit)
    if venue:
        rows = [r for r in rows
                if (r.get("meta") or {}).get("home_away", "").lower() == venue.lower()]
    rows = rows[:last_n]
    metas = [(r.get("meta") or {}) for r in rows]

    alpha = SCORING_WEIGHTS.get("recency", {}).get("alpha", 0.85)

    # Extract raw series for both weighted avg and trend computation
    corners_for_vals = [_numeric(m.get("corners_for")) for m in metas]
    sot_for_vals = [_numeric(m.get("sot_for")) for m in metas]
    cards_vals = [_numeric(m.get("cards_per_90_team")) for m in metas]
    goals_vals = [_numeric(m.get("xg_for")) for m in metas]
    cards_induced_vals = []
    for m in metas:
        opp = str(m.get("opponent") or "")
        fixture = str(m.get("fixture") or "")
        fixture_date = str(m.get("fixture_date") or "")
        season = str(m.get("season") or "")
        if not opp or not fixture:
            cards_induced_vals.append(None)
            continue
        opp_meta = _get_team_fixture_meta(opp, league, fixture, fixture_date, season=season)
        cards_induced_vals.append(_numeric((opp_meta or {}).get("cards_per_90_team")))

    stats = {
        "n": len(metas),
        "corners_for_avg": _weighted_avg(corners_for_vals, alpha),
        "corners_against_avg": _weighted_avg([_numeric(m.get("corners_against")) for m in metas], alpha),
        "shots_for_avg": _weighted_avg([_numeric(m.get("shots_for")) for m in metas], alpha),
        "sot_for_avg": _weighted_avg(sot_for_vals, alpha),
        "shots_against_avg": _weighted_avg([_numeric(m.get("shots_against")) for m in metas], alpha),
        "sot_against_avg": _weighted_avg([_numeric(m.get("sot_against")) for m in metas], alpha),
        "cards_avg": _weighted_avg(cards_vals, alpha),
        "cards_induced_avg": _weighted_avg(cards_induced_vals, alpha),
        "aggression_avg": _weighted_avg([_numeric(m.get("aggression_index_norm")) for m in metas], alpha),
        "control_avg": _weighted_avg([_numeric(m.get("control_index")) for m in metas], alpha),
        "form_avg": _weighted_avg([_numeric(m.get("form_index_team")) for m in metas], alpha),
        "fouls_avg": _weighted_avg([_numeric(m.get("fouls_per_90_team")) for m in metas], alpha),
        "possession_avg": _weighted_avg([_numeric(m.get("possession")) for m in metas], alpha),
        "xg_for_avg": _weighted_avg(goals_vals, alpha),
        # Trend slopes (positive = improving, units per match)
        "corners_for_slope": _linear_slope(corners_for_vals),
        "sot_for_slope": _linear_slope(sot_for_vals),
        "cards_slope": _linear_slope(cards_vals),
        "goals_slope": _linear_slope(goals_vals),
    }
    _team_recent_stats_cache[cache_key] = stats
    return stats


_team_recent_var_cache: Dict[Tuple, Dict] = {}


def get_team_recent_variance(team_name: str, league: str, last_n: int = 8) -> Dict:
    """Compute per-stat variance from recent fixtures for distribution modeling."""
    cache_key = (league, canonical_team_name(team_name).lower(), int(last_n))
    cached = _team_recent_var_cache.get(cache_key)
    if cached is not None:
        return cached

    rows = get_recent_team_fixture_rows(team_name, league, limit=last_n)
    metas = [(r.get("meta") or {}) for r in rows]

    def _var(values):
        clean = [v for v in values if v is not None]
        if len(clean) < 3:
            return None
        mean = sum(clean) / len(clean)
        return sum((x - mean) ** 2 for x in clean) / (len(clean) - 1)

    result = {
        "n": len(metas),
        "corners_for_var": _var([_numeric(m.get("corners_for")) for m in metas]),
        "sot_for_var": _var([_numeric(m.get("sot_for")) for m in metas]),
        "goals_var": _var([_numeric(m.get("xg_for")) for m in metas]),
        "cards_var": _var([_numeric(m.get("cards_per_90_team")) for m in metas]),
    }
    _team_recent_var_cache[cache_key] = result
    return result


def get_blended_variance(team_name: str, league: str, stat_key: str) -> Optional[float]:
    """Bayesian blend of season variance (prior) with recent variance (update).
    Returns blended variance for distribution selection (Poisson vs NegBin)."""
    season_var_key = {"goals_var": "goals_var", "corners_for_var": "corners_var",
                      "cards_var": "cards_var", "sot_for_var": "sot_var"}.get(stat_key)
    recent_dict = get_team_recent_variance(team_name, league)
    recent_var = recent_dict.get(stat_key)
    season_var = None
    if season_var_key:
        meta = _profile_meta(team_name, league)
        season_var = _numeric(meta.get(season_var_key)) if meta else None
    if season_var is not None and recent_var is not None:
        return 0.7 * season_var + 0.3 * recent_var
    return season_var if season_var is not None else recent_var


def get_team_profile_snippets_for_events(events: List[Dict], league: str, per_team: int = 1) -> List[Dict]:
    out: List[Dict] = []
    teams: List[str] = []
    for ev in events:
        for t in [ev.get("home_team"), ev.get("away_team")]:
            if t and t not in teams:
                teams.append(t)
    for team in teams:
        doc = get_team_profile_doc(str(team), league)
        if not doc:
            continue
        out.append({"text": doc.get("text") or "", "meta": doc.get("meta") or {}})
    return out[: max(1, per_team * max(len(teams), 1))]


def get_team_fixture_snippets_for_events(events: List[Dict], league: str, per_team: int = 3) -> List[Dict]:
    out: List[Dict] = []
    teams: List[str] = []
    for ev in events:
        for t in [ev.get("home_team"), ev.get("away_team")]:
            if t and t not in teams:
                teams.append(t)
    for team in teams:
        out.extend(get_recent_team_fixture_rows(str(team), league, limit=per_team))
    return out


def get_player_snippets_for_events(events: List[Dict], league: str, per_team_profiles: int = 4, per_team_fixtures: int = 6, min_minutes: int = 45) -> List[Dict]:
    col = get_collection_handle(create_if_missing=True)
    out_map: Dict[str, Dict] = {}

    teams: List[str] = []
    for ev in events:
        for t in [ev.get("home_team"), ev.get("away_team")]:
            if t and t not in teams:
                teams.append(str(t))

    # Determine current season to avoid cross-season player contamination
    current_season: Optional[str] = None
    if teams:
        prof = get_team_profile_doc(teams[0], league)
        current_season = (prof.get("meta") or {}).get("season")

    for team in teams:
        variants = _kb_team_variants(team)

        picked_profiles = 0
        for tv in variants:
            filters: List[Dict] = [{"doc_type": "player_profile"}, {"team": tv}]
            if current_season:
                filters.append({"season": current_season})
            where = build_where(league, extra_filters=filters)
            res = col.get(where=where, include=["documents", "metadatas"], limit=per_team_profiles)
            for i, d, m in zip(res.get("ids", []), res.get("documents", []), res.get("metadatas", [])):
                out_map[str(i)] = {"id": i, "text": d or "", "meta": m or {}}
                picked_profiles += 1
            if picked_profiles >= per_team_profiles:
                break

        # Collect player fixtures, then sort by minutes desc to prioritize starters
        team_fixtures: List[Dict] = []
        for tv in variants:
            filters = [
                {"doc_type": "player_fixture"},
                {"team": tv},
                {"minutes": {"$gte": min_minutes}},
            ]
            if current_season:
                filters.append({"season": current_season})
            where = build_where(league, extra_filters=filters)
            # Fetch more than needed so we can sort and pick the best
            res = col.get(where=where, include=["documents", "metadatas"], limit=per_team_fixtures * 3)
            for i, d, m in zip(res.get("ids", []), res.get("documents", []), res.get("metadatas", [])):
                team_fixtures.append({"id": i, "text": d or "", "meta": m or {}})

        # Sort by minutes descending — starters with full 90 min come first
        team_fixtures.sort(key=lambda d: d.get("meta", {}).get("minutes") or 0, reverse=True)
        for doc in team_fixtures[:per_team_fixtures]:
            out_map[str(doc["id"])] = doc

    return list(out_map.values())


def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


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


def _venue_blend(venue_rate: Optional[float], overall_rate: Optional[float], blend_w: float) -> Optional[float]:
    """Blend venue-specific and overall rate.  Falls back gracefully."""
    if venue_rate is not None and overall_rate is not None:
        return blend_w * venue_rate + (1.0 - blend_w) * overall_rate
    if venue_rate is not None:
        return venue_rate
    return overall_rate


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


def _stat_blend(stat: str) -> Tuple[float, float]:
    """Return (season_weight, recent_weight) for a given stat type."""
    pw = SCORING_WEIGHTS["projection"]
    s_key = f"blend_{stat}_season"
    r_key = f"blend_{stat}_recent"
    if s_key in pw and r_key in pw:
        return pw[s_key], pw[r_key]
    return pw["blend_season"], pw["blend_recent"]


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


def projected_total_sot(home: str, away: str, league: str, knockout_ctx: Optional[KnockoutContext] = None) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    pw = SCORING_WEIGHTS["projection"]
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

    if season_total is not None and recent_total is not None:
        # Dynamic blend: increase recent weight when form diverges from season avg
        base_season_w, base_recent_w = _stat_blend("sot")
        divergence = abs(recent_total - season_total) / max(season_total, 1.0)
        # When divergence > 20%, shift blend toward recent (up to 60/40)
        if divergence > 0.20:
            shift = min(0.30, (divergence - 0.20) * 1.0)  # max 30% shift
            blend_season = base_season_w - shift
            blend_recent = base_recent_w + shift
        else:
            blend_season = base_season_w
            blend_recent = base_recent_w
        blended = blend_season * season_total + blend_recent * recent_total
    elif season_total is not None:
        blended = season_total
    elif recent_total is not None:
        blended = recent_total
    else:
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


def leg_label(leg: CandidateLeg) -> str:
    g = market_group_from_key(leg.market_key)
    if leg.point is None:
        return f"{leg.outcome} ({leg.market_key}/{g})"
    if g == "spreads":
        return f"{leg.outcome} {leg.point:+g} ({leg.market_key}/{g})"
    return f"{leg.outcome} {leg.point:g} ({leg.market_key}/{g})"


def leg_evidence(leg: CandidateLeg, league: str) -> List[str]:
    hm = _profile_meta(leg.home_team, league)
    am = _profile_meta(leg.away_team, league)
    hr = _recent_stats(leg.home_team, league, last_n=6)
    ar = _recent_stats(leg.away_team, league, last_n=6)
    heur_groups = heuristic_groups_for_event(hm, am)
    lines: List[str] = []

    def v(meta: Dict, key: str) -> Optional[float]:
        x = meta.get(key)
        try:
            return float(x)
        except Exception:
            return None

    g = market_group_from_key(leg.market_key)
    if g == "corners":
        h_for, h_against = v(hm, "corners_pm"), v(hm, "corners_against_pm")
        a_for, a_against = v(am, "corners_pm"), v(am, "corners_against_pm")
        h_edge, a_edge = v(hm, "corner_edge_pm"), v(am, "corner_edge_pm")
        h_ctrl, a_ctrl = v(hm, "control_index"), v(am, "control_index")
        h_proj, a_proj = projected_corners(hm, am)
        if h_for is not None and h_against is not None:
            lines.append(f"{leg.home_team} corners for/against: {h_for:.2f}/{h_against:.2f} per match.")
        if a_for is not None and a_against is not None:
            lines.append(f"{leg.away_team} corners for/against: {a_for:.2f}/{a_against:.2f} per match.")
        if h_proj is not None and a_proj is not None:
            total_proj = h_proj + a_proj
            total_proj = _market_blended_proj(total_proj, leg.event, "corners")
            side = f"{leg.home_team} {h_proj:.2f} vs {leg.away_team} {a_proj:.2f}"
            lines.append(f"Projected corners: {side} (combined {total_proj:.2f}).")
            k = leg.market_key.lower()
            if leg.point is not None and "total" in k:
                direction = leg.outcome.lower()
                if "over" in direction:
                    lines.append(f"Line fit: projected {total_proj:.2f} vs over {leg.point:g}.")
                elif "under" in direction:
                    lines.append(f"Line fit: projected {total_proj:.2f} vs under {leg.point:g}.")
        if h_edge is not None and a_edge is not None:
            lines.append(f"Corner edge: {leg.home_team} {h_edge:+.2f}, {leg.away_team} {a_edge:+.2f} per match.")
        if h_ctrl is not None and a_ctrl is not None:
            lines.append(f"Control index: {leg.home_team} {h_ctrl:.2f}, {leg.away_team} {a_ctrl:.2f}.")
        h_recent = hr.get("corners_for_avg")
        a_recent = ar.get("corners_for_avg")
        if h_recent is not None and a_recent is not None:
            lines.append(f"Recent 6-match corners for: {leg.home_team} {h_recent:.2f}, {leg.away_team} {a_recent:.2f}.")
    elif g == "totals":
        h_gf, a_gf = v(hm, "goals_for_pm"), v(am, "goals_for_pm")
        h_ctrl, a_ctrl = v(hm, "control_index"), v(am, "control_index")
        h_dom, a_dom = v(hm, "dominance_index"), v(am, "dominance_index")
        h_edge, a_edge = v(hm, "corner_edge_pm"), v(am, "corner_edge_pm")
        if h_gf is not None and a_gf is not None:
            lines.append(f"Goal baseline: {leg.home_team} {h_gf:.2f}/match, {leg.away_team} {a_gf:.2f}/match.")
        if h_ctrl is not None and a_ctrl is not None:
            lines.append(f"Control index: {leg.home_team} {h_ctrl:.2f}, {leg.away_team} {a_ctrl:.2f}.")
        if h_dom is not None and a_dom is not None:
            lines.append(f"Dominance index: {leg.home_team} {h_dom:.2f}, {leg.away_team} {a_dom:.2f}.")
        if h_edge is not None and a_edge is not None:
            lines.append(f"Corner edge proxy: {leg.home_team} {h_edge:+.2f}, {leg.away_team} {a_edge:+.2f}.")
        h_sot, a_sot = hr.get("sot_for_avg"), ar.get("sot_for_avg")
        if h_sot is not None and a_sot is not None:
            lines.append(f"Recent 6-match SoT for: {leg.home_team} {h_sot:.2f}, {leg.away_team} {a_sot:.2f}.")
    elif g == "sot":
        h_sot_pm, a_sot_pm = v(hm, "sot_for_pm"), v(am, "sot_for_pm")
        if h_sot_pm is not None and a_sot_pm is not None:
            lines.append(f"Season SoT/match: {leg.home_team} {h_sot_pm:.2f}, {leg.away_team} {a_sot_pm:.2f}.")
        sot_total, sot_season, sot_recent = projected_total_sot(leg.home_team, leg.away_team, league)
        if sot_total is not None:
            sot_total = _market_blended_proj(sot_total, leg.event, "sot")
            lines.append(f"Projected combined SoT: {sot_total:.2f}.")
            if leg.point is not None:
                direction = leg.outcome.lower()
                if "over" in direction:
                    lines.append(f"Line fit: projected {sot_total:.2f} vs over {leg.point:g}.")
                elif "under" in direction:
                    lines.append(f"Line fit: projected {sot_total:.2f} vs under {leg.point:g}.")
        h_recent_sot, a_recent_sot = hr.get("sot_for_avg"), ar.get("sot_for_avg")
        if h_recent_sot is not None and a_recent_sot is not None:
            lines.append(f"Recent 6-match SoT: {leg.home_team} {h_recent_sot:.2f}, {leg.away_team} {a_recent_sot:.2f}.")
    elif g == "btts":
        result = projected_btts_prob(leg.home_team, leg.away_team, league)
        p_btts, h_proj, a_proj, _, _ = result if result else (None, None, None, None, None)
        if p_btts is not None:
            lines.append(f"P(BTTS Yes) = {p_btts:.1%}.")
        if h_proj is not None and a_proj is not None:
            lines.append(f"Goal projections: {leg.home_team} {h_proj:.2f}, {leg.away_team} {a_proj:.2f}.")
        h_gf = v(hm, "goals_for_pm")
        a_gf = v(am, "goals_for_pm")
        if h_gf is not None and a_gf is not None:
            lines.append(f"Season goals/match: {leg.home_team} {h_gf:.2f}, {leg.away_team} {a_gf:.2f}.")
        h_ga = v(hm, "goals_against_pm")
        a_ga = v(am, "goals_against_pm")
        if h_ga is not None and a_ga is not None:
            lines.append(f"Goals conceded/match: {leg.home_team} {h_ga:.2f}, {leg.away_team} {a_ga:.2f}.")
    elif g == "spreads":
        h_form, a_form = v(hm, "form_index_team"), v(am, "form_index_team")
        h_ctrl, a_ctrl = v(hm, "control_index"), v(am, "control_index")
        h_dom, a_dom = v(hm, "dominance_index"), v(am, "dominance_index")
        h_edge, a_edge = v(hm, "corner_edge_pm"), v(am, "corner_edge_pm")
        if h_form is not None and a_form is not None:
            lines.append(f"Form index: {leg.home_team} {h_form:.2f} vs {leg.away_team} {a_form:.2f}.")
        if h_ctrl is not None and a_ctrl is not None:
            lines.append(f"Control index: {leg.home_team} {h_ctrl:.2f} vs {leg.away_team} {a_ctrl:.2f}.")
        if h_dom is not None and a_dom is not None:
            lines.append(f"Dominance index: {leg.home_team} {h_dom:.2f} vs {leg.away_team} {a_dom:.2f}.")
        if h_edge is not None and a_edge is not None:
            lines.append(f"Corner edge: {leg.home_team} {h_edge:+.2f} vs {leg.away_team} {a_edge:+.2f}.")
        h_sot, a_sot = hr.get("sot_for_avg"), ar.get("sot_for_avg")
        if h_sot is not None and a_sot is not None:
            lines.append(f"Recent 6-match SoT edge: {leg.home_team} {h_sot:.2f} vs {leg.away_team} {a_sot:.2f}.")
        proj_diff, season_diff, _ = projected_goal_difference(leg.home_team, leg.away_team, league)
        if proj_diff is not None:
            lines.append(f"Projected goal diff (home-away): {proj_diff:+.2f}.")
        if season_diff is not None:
            lines.append(f"Season goal diff: {season_diff:+.2f}.")
    elif g == "moneyline":
        h_form, a_form = v(hm, "form_index_team"), v(am, "form_index_team")
        h_dom, a_dom = v(hm, "dominance_index"), v(am, "dominance_index")
        if h_form is not None and a_form is not None:
            lines.append(f"Form index: {leg.home_team} {h_form:.2f} vs {leg.away_team} {a_form:.2f}.")
        if h_dom is not None and a_dom is not None:
            lines.append(f"Dominance index: {leg.home_team} {h_dom:.2f} vs {leg.away_team} {a_dom:.2f}.")
        h_recent_form, a_recent_form = hr.get("form_avg"), ar.get("form_avg")
        if h_recent_form is not None and a_recent_form is not None:
            lines.append(f"Recent 6-match form index: {leg.home_team} {h_recent_form:.2f} vs {leg.away_team} {a_recent_form:.2f}.")
    elif g == "cards":
        h_cards, a_cards = v(hm, "cards_per_90_team"), v(am, "cards_per_90_team")
        h_fouls, a_fouls = v(hm, "fouls_per_90_team"), v(am, "fouls_per_90_team")
        h_opp_induced = v(hm, "opp_cards_induced_pm")
        a_opp_induced = v(am, "opp_cards_induced_pm")

        if h_cards is not None and a_cards is not None:
            lines.append(f"Cards/90: {leg.home_team} {h_cards:.2f}, {leg.away_team} {a_cards:.2f}.")
        if h_fouls is not None and a_fouls is not None:
            lines.append(f"Fouls/90: {leg.home_team} {h_fouls:.2f}, {leg.away_team} {a_fouls:.2f}.")
        if h_opp_induced is not None and a_opp_induced is not None:
            lines.append(f"Opp cards induced/match: {leg.home_team} {h_opp_induced:.2f}, {leg.away_team} {a_opp_induced:.2f}.")

        cards_total, cards_season, cards_recent, cards_ref_mod = projected_total_cards(
            leg.home_team, leg.away_team, league
        )
        if cards_total is not None:
            cards_total = _market_blended_proj(cards_total, leg.event, "cards")
            lines.append(f"Projected combined cards: {cards_total:.2f}.")
            if leg.point is not None:
                direction = leg.outcome.lower()
                if "over" in direction:
                    lines.append(f"Line fit: projected {cards_total:.2f} vs over {leg.point:g}.")
                elif "under" in direction:
                    lines.append(f"Line fit: projected {cards_total:.2f} vs under {leg.point:g}.")

        # Referee evidence for cards
        if cards_ref_mod.source == "profile" and cards_ref_mod.referee_name:
            rw = SCORING_WEIGHTS["referee"]
            if abs(cards_ref_mod.multiplier - 1.0) > rw["evidence_threshold"]:
                direction_word = "stricter" if cards_ref_mod.multiplier > 1.0 else "more lenient"
                lines.append(
                    f"Referee: {cards_ref_mod.referee_name.title()} is {direction_word} than avg "
                    f"({cards_ref_mod.strictness_ratio:.2f}x, {cards_ref_mod.sample_size} matches)."
                )
                if cards_ref_mod.cards_per_foul > 0:
                    lines.append(
                        f"Ref cards/foul: {cards_ref_mod.cards_per_foul:.3f}, "
                        f"avg fouls/match: {cards_ref_mod.avg_fouls_per_match:.1f}."
                    )

        h_recent_cards = hr.get("cards_avg")
        a_recent_cards = ar.get("cards_avg")
        if h_recent_cards is not None and a_recent_cards is not None:
            lines.append(f"Recent 6-match cards avg: {leg.home_team} {h_recent_cards:.2f}, {leg.away_team} {a_recent_cards:.2f}.")
    else:
        h_ctrl, a_ctrl = v(hm, "control_index"), v(am, "control_index")
        if h_ctrl is not None and a_ctrl is not None:
            lines.append(f"Control profile: {leg.home_team} {h_ctrl:.2f}, {leg.away_team} {a_ctrl:.2f}.")

    # --- Probability evidence for count-based markets ---
    if leg.point is not None and g in ("corners", "cards", "sot", "totals"):
        proj_map = {
            "corners": lambda: projected_total_corners(leg.home_team, leg.away_team, league),
            "cards": lambda: projected_total_cards(leg.home_team, leg.away_team, league),
            "sot": lambda: projected_total_sot(leg.home_team, leg.away_team, league),
            "totals": lambda: projected_total_goals(leg.home_team, leg.away_team, league),
        }
        proj_result = proj_map[g]()
        proj_total = proj_result[0] if proj_result else None
        if proj_total is not None:
            proj_total = _market_blended_proj(proj_total, leg.event, g if g != "totals" else "totals")
            direction = leg.outcome.lower()
            is_over = "over" in direction
            h_var = get_team_recent_variance(leg.home_team, league)
            a_var = get_team_recent_variance(leg.away_team, league)
            var_key = {"corners": "corners_for_var", "cards": "cards_var",
                       "sot": "sot_for_var", "totals": "goals_var"}[g]
            h_v, a_v = h_var.get(var_key), a_var.get(var_key)
            comb_var = (h_v + a_v) if (h_v is not None and a_v is not None) else None
            model_p = over_prob(proj_total, leg.point, comb_var) if is_over else under_prob(proj_total, leg.point, comb_var)
            implied_p = implied_prob(leg.odds)
            ve = value_edge(model_p, implied_p)
            side_label = "Over" if is_over else "Under"
            lines.append(f"P({side_label} {leg.point:g}) = {model_p:.1%} | Books: {implied_p:.1%} | Value: {ve:+.1%}")

    # --- ML evidence for count-based markets ---
    if _HAS_ML and g in ("corners", "cards", "sot", "totals"):
        stat_map = {"corners": "corners", "cards": "cards", "sot": "sot", "totals": "goals"}
        stat_label = {"corners": "corners", "cards": "cards", "sot": "SoT", "totals": "goals"}
        stat_key = stat_map[g]
        ml_proj = ml_predict_total(leg.home_team, leg.away_team, league, stat_key, home_meta=hm, away_meta=am)
        elo_diff = get_elo_edge(leg.home_team, leg.away_team, stat_key)
        if ml_proj is not None:
            lines.append(f"ML model: projected {ml_proj:.1f} {stat_label[g]}.")
        if elo_diff is not None and abs(elo_diff) >= 20:
            leader = "home" if elo_diff > 0 else "away"
            lines.append(f"Elo: {leader} +{abs(elo_diff):.0f} {stat_label[g]} rating edge.")

    if g in heur_groups:
        lines.append(f"Heuristic slate also favors {g} for this fixture.")

    if not lines:
        lines.append("Team-profile stats unavailable for this fixture in local KB.")
    # prioritize multi-signal support — allow extra lines for probability + ML
    return lines[:6]


def combined_odds(legs: List[CandidateLeg]) -> float:
    p = 1.0
    for leg in legs:
        p *= leg.odds
    return p


def summarize_constraints(c: ConstraintSpec, available_groups: Set[str], selected: List[CandidateLeg]) -> Tuple[List[str], List[str]]:
    applied: List[str] = []
    unmet: List[str] = []

    applied.append("global rule: no moneyline + spread mix")
    applied.append("global rule: one leg per market family per fixture")
    applied.append(f"global rule: min leg odds >= {MIN_LEG_ODDS:.2f}")

    if c.per_match_mode:
        applied.append("one leg per fixture")
    elif c.require_unique_events:
        applied.append("unique fixtures")

    if c.target_multiplier is not None and c.target_mode != "none":
        combo = combined_odds(selected)
        tgt = c.target_multiplier
        if c.target_mode == "max":
            if combo <= tgt:
                applied.append(f"target max {tgt:.2f}x met")
            else:
                unmet.append(f"target max {tgt:.2f}x not met (selected {combo:.2f}x)")
        elif c.target_mode == "min":
            if combo >= tgt:
                applied.append(f"target min {tgt:.2f}x met")
            else:
                unmet.append(f"target min {tgt:.2f}x not met (selected {combo:.2f}x)")
        else:
            rel_dev = abs(combo - tgt) / max(tgt, 1e-9)
            if rel_dev <= AROUND_TARGET_TOLERANCE:
                applied.append(f"target around {tgt:.2f}x met")
            else:
                unmet.append(
                    f"target around {tgt:.2f}x not met (selected {combo:.2f}x, deviation {rel_dev*100:.1f}%)"
                )

    for req in c.hard_include_groups:
        if req not in available_groups:
            unmet.append(f"requested '{req}' markets are unavailable from current odds feed")
        elif all(market_group_from_key(x.market_key) != req for x in selected):
            unmet.append(f"requested '{req}' not represented in selected legs")
        else:
            applied.append(f"{req}-only preference honored")

    for grp, need in c.required_group_counts.items():
        have = sum(1 for x in selected if market_group_from_key(x.market_key) == grp)
        if have >= need:
            applied.append(f"{grp} legs {have}/{need}")
        else:
            unmet.append(f"required {need} {grp} leg(s), got {have}")

    for ex in c.hard_exclude_groups:
        if any(market_group_from_key(x.market_key) == ex for x in selected):
            unmet.append(f"exclude '{ex}' was violated by selected legs")
        else:
            applied.append(f"excluded '{ex}'")

    if c.forbid_spread_keys:
        if any(("spread" in x.market_key.lower()) or ("handicap" in x.market_key.lower()) for x in selected):
            unmet.append("no spreads requested, but spread/handicap market key present")
        else:
            applied.append("no spread keys")

    if c.require_total_corner_keys:
        bad = [
            x for x in selected
            if (("corner" not in x.market_key.lower()) or ("total" not in x.market_key.lower()))
        ]
        if bad:
            unmet.append("total-corners-only requested, but non-total-corner key selected")
        else:
            applied.append("total-corners-only")

    return applied, unmet


def parse_llm_leg_evidence(text: str, leg_count: int) -> Tuple[Dict[int, List[str]], Optional[str]]:
    by_leg: Dict[int, List[str]] = {}
    why_lines: List[str] = []
    current_leg: Optional[int] = None
    in_why = False
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"LEG\s+(\d+)\s*:", line, flags=re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= leg_count:
                current_leg = idx
                in_why = False
                by_leg.setdefault(idx, [])
            else:
                current_leg = None
            continue
        if re.match(r"WHY\s*:", line, flags=re.IGNORECASE):
            in_why = True
            current_leg = None
            continue
        if in_why:
            why_lines.append(line.lstrip("- ").strip())
            continue
        if current_leg is not None and line.startswith("-"):
            by_leg.setdefault(current_leg, []).append(line.lstrip("- ").strip())
    why = " ".join(x for x in why_lines if x).strip() or None
    return by_leg, why


def build_reasoning_messages(
    user_q: str,
    selected: List[CandidateLeg],
    profile_docs: List[Dict],
    team_fixture_docs: List[Dict],
    player_docs: List[Dict],
    chat_context: str,
) -> List[Dict]:
    selected_lines = []
    for i, leg in enumerate(selected, start=1):
        bm = f" | bookmaker={leg.bookmaker}" if leg.bookmaker else ""
        point = "none" if leg.point is None else f"{leg.point:g}"
        selected_lines.append(
            f"LEG {i}: fixture={leg.fixture} | market_key={leg.market_key} | outcome={leg.outcome} | point={point} | odds={leg.odds:.2f}{bm}"
        )
    selected_block = "\n".join(selected_lines)
    profile_text = "\n\n".join((d.get("text") or "") for d in profile_docs[:10])
    team_fx_text = "\n\n".join((d.get("text") or "") for d in team_fixture_docs[:14])
    player_text = "\n\n".join((d.get("text") or "") for d in player_docs[:18])

    system_msg = (
        "You are a football betting analyst.\n"
        "You must explain the PROVIDED fixed legs only.\n"
        "Do not add, remove, reorder, or replace any leg.\n"
        "Use only numbers from context.\n"
        "Avoid single-match recency logic unless framed as a small supporting point.\n"
        "Use multi-signal support (form/control/dominance/corners/cards/shots/sot where available).\n"
        "If player data is sparse, still give the best available evidence from provided player/team context.\n"
        "Return EXACTLY in this format:\n"
        "LEG 1:\n"
        "- <evidence line>\n"
        "- <evidence line>\n"
        "LEG 2:\n"
        "- ...\n"
        "WHY:\n"
        "- <2-3 concise lines on why this parlay structure makes sense>\n"
    )
    user_msg = (
        f"User query:\n{user_q}\n\n"
        f"Recent chat context:\n{chat_context}\n\n"
        f"Fixed selected legs:\n{selected_block}\n\n"
        f"Team profile context:\n{profile_text or 'None'}\n\n"
        f"Recent team fixture context:\n{team_fx_text or 'None'}\n\n"
        f"Player context:\n{player_text or 'None'}\n"
    )
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]


def get_llm_reasoning(
    user_q: str,
    selected: List[CandidateLeg],
    events: List[Dict],
    league: str,
) -> Tuple[Dict[int, List[str]], Optional[str], Optional[str]]:
    if not client:
        return {}, None, "OPENAI_API_KEY not set; using deterministic evidence."
    selected_event_ids = {x.event_id for x in selected}
    selected_events = [ev for ev in events if str(ev.get("id") or "") in selected_event_ids]
    profile_docs = get_team_profile_snippets_for_events(selected_events, league, per_team=1)
    team_fixture_docs = get_team_fixture_snippets_for_events(selected_events, league, per_team=3)
    player_docs = get_player_snippets_for_events(selected_events, league, per_team_profiles=4, per_team_fixtures=6)
    messages = build_reasoning_messages(
        user_q,
        selected,
        profile_docs,
        team_fixture_docs,
        player_docs,
        chat_context=render_recent_chat_context(max_turns=3),
    )
    try:
        resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
        text = (resp.choices[0].message.content or "").strip()
        evidence_map, why = parse_llm_leg_evidence(text, leg_count=len(selected))
        if not evidence_map:
            return {}, None, "LLM explanation parse failed; using deterministic evidence."
        return evidence_map, why, None
    except Exception as exc:
        return {}, None, f"LLM explanation failed: {exc}"


def get_llm_reasoning_multi(
    user_q: str,
    selected: List[CandidateLeg],
    events: List[Dict],
) -> Tuple[Dict[int, List[str]], Optional[str], Optional[str]]:
    """LLM reasoning for cross-league parlays — retrieves KB snippets per leg's league."""
    if not client:
        return {}, None, "OPENAI_API_KEY not set; using deterministic evidence."
    profile_docs: List[str] = []
    team_fixture_docs: List[str] = []
    player_docs: List[str] = []
    for leg in selected:
        lg = leg.league or "EPL"
        leg_events = [ev for ev in events if str(ev.get("id") or "") == leg.event_id]
        profile_docs.extend(get_team_profile_snippets_for_events(leg_events, lg, per_team=1))
        team_fixture_docs.extend(get_team_fixture_snippets_for_events(leg_events, lg, per_team=2))
        player_docs.extend(get_player_snippets_for_events(leg_events, lg, per_team_profiles=3, per_team_fixtures=4))
    messages = build_reasoning_messages(
        user_q, selected, profile_docs, team_fixture_docs, player_docs,
        chat_context=render_recent_chat_context(max_turns=3),
    )
    try:
        resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
        text = (resp.choices[0].message.content or "").strip()
        evidence_map, why = parse_llm_leg_evidence(text, leg_count=len(selected))
        if not evidence_map:
            return {}, None, "LLM explanation parse failed; using deterministic evidence."
        return evidence_map, why, None
    except Exception as exc:
        return {}, None, f"LLM explanation failed: {exc}"


def llm_validate_selection(
    selected: List[CandidateLeg],
    league: str,
) -> Tuple[bool, Optional[str]]:
    """
    Optional LLM-in-the-loop validation. Gated by RAG_LLM_VALIDATION env var.
    Returns (passed, concern_text). If validation is disabled or unavailable, always passes.
    """
    if not env_bool("RAG_LLM_VALIDATION", default=False):
        return True, None
    if not client:
        return True, None

    legs_text = []
    for i, leg in enumerate(selected, start=1):
        point = "none" if leg.point is None else f"{leg.point:g}"
        legs_text.append(
            f"LEG {i}: {leg.fixture} | {leg.market_key} | {leg.outcome} | point={point} | odds={leg.odds:.2f}"
        )

    system_msg = (
        "You are a betting validation assistant. Review the proposed parlay legs below.\n"
        "Flag ONLY clear logical issues:\n"
        "- Contradictory legs (e.g., same-match Over and Under on same line)\n"
        "- Correlated exposure that compounds risk (e.g., two legs on the same team outcome)\n"
        "- Clearly unrealistic lines given the market context\n"
        "If all legs look reasonable, respond with exactly: PASS\n"
        "Otherwise respond with: CONCERN: <1-2 sentence explanation>\n"
        "Do NOT second-guess the statistical model or odds values."
    )
    user_msg = f"League: {league}\nCombined odds: {combined_odds(selected):.2f}x\n\n" + "\n".join(legs_text)

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=150,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text.upper().startswith("PASS"):
            return True, None
        concern = text.replace("CONCERN:", "").strip() if "CONCERN" in text.upper() else text
        return False, concern
    except Exception:
        return True, None


def _leg_standalone_confidence(leg: CandidateLeg, league: str) -> Dict:
    """Run the standalone market projection for a parlay leg to get confidence,
    model probability, and check for side agreement.

    Returns dict with: confidence, model_prob, projected, side_agrees, standalone_side
    """
    g = market_group_from_key(leg.market_key)
    result: Dict = {
        "confidence": None, "model_prob": None, "projected": None,
        "side_agrees": True, "standalone_side": None, "warning": None,
    }

    try:
        if g == "corners":
            proj_total, _, _ = projected_total_corners(leg.home_team, leg.away_team, league)
            if proj_total is not None and leg.point is not None:
                direction = leg.outcome.lower()
                is_over = "over" in direction
                h_var = get_team_recent_variance(leg.home_team, league)
                a_var = get_team_recent_variance(leg.away_team, league)
                comb_var_val = None
                hv, av = h_var.get("corners_for_var"), a_var.get("corners_for_var")
                if hv is not None and av is not None:
                    comb_var_val = hv + av
                model_p = over_prob(proj_total, leg.point, comb_var_val) if is_over else under_prob(proj_total, leg.point, comb_var_val)
                implied_p = implied_prob(leg.odds)
                ve = value_edge(model_p, implied_p)
                standalone_side = "Over" if proj_total > leg.point else "Under"
                result["projected"] = proj_total
                result["model_prob"] = model_p
                result["standalone_side"] = standalone_side
                result["side_agrees"] = (is_over and standalone_side == "Over") or (not is_over and standalone_side == "Under")
                result["confidence"] = confidence_from_edge(abs(ve), stat_group="corners", model_prob=model_p, value_edge_pct=ve)

        elif g == "totals":
            proj_total, _, _ = projected_total_goals(leg.home_team, leg.away_team, league)
            if proj_total is not None and leg.point is not None:
                direction = leg.outcome.lower()
                is_over = "over" in direction
                h_var = get_team_recent_variance(leg.home_team, league)
                a_var = get_team_recent_variance(leg.away_team, league)
                comb_var_val = None
                hv, av = h_var.get("goals_var"), a_var.get("goals_var")
                if hv is not None and av is not None:
                    comb_var_val = hv + av
                model_p = over_prob(proj_total, leg.point, comb_var_val) if is_over else under_prob(proj_total, leg.point, comb_var_val)
                implied_p = implied_prob(leg.odds)
                ve = value_edge(model_p, implied_p)
                standalone_side = "Over" if proj_total > leg.point else "Under"
                result["projected"] = proj_total
                result["model_prob"] = model_p
                result["standalone_side"] = standalone_side
                result["side_agrees"] = (is_over and standalone_side == "Over") or (not is_over and standalone_side == "Under")
                result["confidence"] = confidence_from_edge(abs(ve), stat_group="goals", model_prob=model_p, value_edge_pct=ve)

        elif g == "cards":
            proj_total, _, _, ref_mod = projected_total_cards(leg.home_team, leg.away_team, league)
            if proj_total is not None and leg.point is not None:
                direction = leg.outcome.lower()
                is_over = "over" in direction
                h_var = get_team_recent_variance(leg.home_team, league)
                a_var = get_team_recent_variance(leg.away_team, league)
                comb_var_val = None
                hv, av = h_var.get("cards_var"), a_var.get("cards_var")
                if hv is not None and av is not None:
                    comb_var_val = hv + av
                model_p = over_prob(proj_total, leg.point, comb_var_val) if is_over else under_prob(proj_total, leg.point, comb_var_val)
                implied_p = implied_prob(leg.odds)
                ve = value_edge(model_p, implied_p)
                standalone_side = "Over" if proj_total > leg.point else "Under"
                result["projected"] = proj_total
                result["model_prob"] = model_p
                result["standalone_side"] = standalone_side
                result["side_agrees"] = (is_over and standalone_side == "Over") or (not is_over and standalone_side == "Under")
                result["confidence"] = confidence_from_edge(abs(ve), stat_group="cards", model_prob=model_p, value_edge_pct=ve)

        elif g == "sot":
            proj_total, _, _ = projected_total_sot(leg.home_team, leg.away_team, league)
            if proj_total is not None and leg.point is not None:
                direction = leg.outcome.lower()
                is_over = "over" in direction
                h_var = get_team_recent_variance(leg.home_team, league)
                a_var = get_team_recent_variance(leg.away_team, league)
                comb_var_val = None
                hv, av = h_var.get("sot_for_var"), a_var.get("sot_for_var")
                if hv is not None and av is not None:
                    comb_var_val = hv + av
                model_p = over_prob(proj_total, leg.point, comb_var_val) if is_over else under_prob(proj_total, leg.point, comb_var_val)
                implied_p = implied_prob(leg.odds)
                ve = value_edge(model_p, implied_p)
                standalone_side = "Over" if proj_total > leg.point else "Under"
                result["projected"] = proj_total
                result["model_prob"] = model_p
                result["standalone_side"] = standalone_side
                result["side_agrees"] = (is_over and standalone_side == "Over") or (not is_over and standalone_side == "Under")
                result["confidence"] = confidence_from_edge(abs(ve), stat_group="sot", model_prob=model_p, value_edge_pct=ve)

        elif g == "btts":
            btts_result = projected_btts_prob(leg.home_team, leg.away_team, league)
            p_btts = btts_result[0] if btts_result else None
            if p_btts is not None:
                direction = leg.outcome.lower().strip()
                model_p = p_btts if direction == "yes" else (1.0 - p_btts)
                implied_p = implied_prob(leg.odds)
                ve = value_edge(model_p, implied_p)
                standalone_side = "Yes" if p_btts >= 0.50 else "No"
                result["projected"] = p_btts
                result["model_prob"] = model_p
                result["standalone_side"] = standalone_side
                result["side_agrees"] = direction == standalone_side.lower()
                result["confidence"] = confidence_from_edge(abs(ve), stat_group="goals", model_prob=model_p, value_edge_pct=ve)
                if not result["side_agrees"] and abs(p_btts - 0.50) > 0.10:
                    result["warning"] = f"Model favors BTTS {standalone_side} ({p_btts:.0%})"

        elif g == "moneyline":
            ml_result = projected_moneyline_probs(leg.home_team, leg.away_team, league)
            if ml_result and ml_result[0] is not None:
                p_home, p_draw, p_away = ml_result[0], ml_result[1], ml_result[2]
                direction = leg.outcome.lower().strip()
                home_lower = leg.home_team.lower()
                if direction.startswith(home_lower) or "home" in direction:
                    model_p = p_home
                elif "draw" in direction:
                    model_p = p_draw
                else:
                    model_p = p_away
                implied_p = implied_prob(leg.odds)
                ve = value_edge(model_p, implied_p)
                # Standalone would pick highest probability
                best_side = max([("Home", p_home), ("Draw", p_draw), ("Away", p_away)], key=lambda x: x[1])
                result["model_prob"] = model_p
                result["standalone_side"] = best_side[0]
                result["confidence"] = confidence_from_edge(abs(ve), stat_group="goals", model_prob=model_p, value_edge_pct=ve)

        elif g == "spreads":
            proj_diff, _, _ = projected_goal_difference(leg.home_team, leg.away_team, league)
            if proj_diff is not None and leg.point is not None:
                from math import erf, sqrt
                margin_std = SCORING_WEIGHTS["prob"].get("margin_std_default", 1.25)
                # Home covers if diff > -point
                cover_z = (proj_diff + leg.point) / margin_std
                model_p = 0.5 * (1.0 + erf(cover_z / sqrt(2.0)))
                if leg.outcome.lower().startswith(leg.away_team.lower()):
                    model_p = 1.0 - model_p
                implied_p = implied_prob(leg.odds)
                ve = value_edge(model_p, implied_p)
                result["projected"] = proj_diff
                result["model_prob"] = model_p
                result["confidence"] = confidence_from_edge(abs(ve), stat_group="goals", model_prob=model_p, value_edge_pct=ve)

    except Exception:
        pass  # graceful — return empty result if anything fails

    return result


def render_parlay(
    user_q: str,
    c: ConstraintSpec,
    events: List[Dict],
    selected: List[CandidateLeg],
    notes: List[str],
    llm_evidence: Optional[Dict[int, List[str]]] = None,
    llm_why: Optional[str] = None,
) -> str:
    combo = combined_odds(selected)
    available_groups = available_market_groups(build_candidates(events))
    applied, unmet = summarize_constraints(c, available_groups, selected)

    lines: List[str] = []
    lines.append("Parlay recommendation:")

    warnings: List[str] = []
    leg_confidence_cache: List[Optional[str]] = []  # cache per-leg confidence for summary

    for i, leg in enumerate(selected, start=1):
        bm = f" ({leg.bookmaker})" if leg.bookmaker else ""
        lg_badge = f"[{leg.league}] " if leg.league else ""
        league = _leg_league(leg, c.league)

        # Get standalone confidence for this leg
        sc = _leg_standalone_confidence(leg, league)
        conf_label = sc.get("confidence") or "—"
        model_p = sc.get("model_prob")
        conf_str = f" ({conf_label} confidence" + (f", model {model_p:.0%}" if model_p else "") + ")"
        leg_confidence_cache.append(conf_label if conf_label != "—" else None)

        lines.append(f"- Leg {i}: {lg_badge}{leg.fixture} | {leg_label(leg)} @ {leg.odds:.2f}{bm}{conf_str}")

        # Cross-validation warning
        if not sc.get("side_agrees", True) and sc.get("warning"):
            lines.append(f"  ⚠ {sc['warning']}")
            warnings.append(f"Leg {i}: {sc['warning']}")
        elif not sc.get("side_agrees", True) and sc.get("standalone_side"):
            g = market_group_from_key(leg.market_key)
            lines.append(f"  ⚠ Standalone {g} model favors {sc['standalone_side']}")
            warnings.append(f"Leg {i}: standalone {g} model favors {sc['standalone_side']}")

        ev_lines = (llm_evidence or {}).get(i) or leg_evidence(leg, league)
        for ev in ev_lines[:3]:
            lines.append(f"  Evidence: {ev}")

    lines.append(f"- Estimated combined odds: {combo:.2f}x")

    # Overall parlay confidence summary (from cached values, no recomputation)
    conf_counts = Counter(cf for cf in leg_confidence_cache if cf)
    if conf_counts:
        conf_summary = ", ".join(f"{cnt} {label}" for label, cnt in
                                 sorted(conf_counts.items(), key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x[0], 3)))
        lines.append(f"- Confidence breakdown: {conf_summary}")

    # League distribution summary for cross-league parlays
    if any(leg.league for leg in selected):
        league_counts = Counter(leg.league for leg in selected if leg.league)
        if len(league_counts) > 1:
            dist = ", ".join(f"{cnt}x {lg}" for lg, cnt in league_counts.most_common())
            lines.append(f"- League distribution: {dist}")

    if warnings:
        lines.append(f"- ⚠ Cross-check warnings: {len(warnings)} leg(s) disagree with standalone projections")

    if llm_why:
        lines.append(f"- Why this parlay: {llm_why}")
    lines.append("- Constraint report:")
    lines.append(f"  Applied: {('; '.join(applied) if applied else 'none')}")
    lines.append(f"  Unmet: {('; '.join(unmet) if unmet else 'none')}")

    if notes:
        lines.append("- Notes:")
        for n in notes:
            lines.append(f"  {n}")

    if unmet:
        lines.append(
            "- Fallback behavior: kept best possible legs from available markets instead of returning no answer."
        )

    return "\n".join(lines)


def is_schedule_query(user_q: str) -> bool:
    q = user_q.lower()
    schedule_terms = ["what matches", "games are on", "matches are on", "schedule", "who plays",
                      "kickoff time", "kick-off time", "kick off time"]
    if any(t in q for t in schedule_terms):
        return True
    # "show me ... fixtures" / "list ... fixtures" / "what fixtures" — the fixture IS the object
    if re.search(r"\b(show|list)\b.{0,20}\bfixtures?\b", q):
        return True
    # "what/which fixtures" only when NOT followed by "are best/worst" (that's analysis)
    if re.search(r"\b(what|which|upcoming)\s+\w*\s*fixtures?\b", q):
        if not re.search(r"\bfixtures?\s+are\s+(best|worst|good)", q):
            return True
    return False


def is_capability_query(user_q: str) -> bool:
    q = user_q.lower()
    return bool(re.search(r"\b(available|availability|have|offer|supported)\b", q) and re.search(r"\b(corner|card|booking|spread|total|moneyline|ml|h2h)\b", q))


def requested_groups_from_query(user_q: str) -> Set[str]:
    q = user_q.lower()
    out: Set[str] = set()
    if re.search(r"\bcorner", q):
        out.add("corners")
    if re.search(r"\b(card|booking)", q):
        out.add("cards")
    if re.search(r"\b(shot|sot)\b", q) or "shots on target" in q:
        out.add("sot")
    if re.search(r"\bspread|handicap", q):
        out.add("spreads")
    if re.search(r"\btotal|over|under", q):
        out.add("totals")
    if re.search(r"\bmoneyline\b|\bmoney\s*line\b|\bml\b|\bh2h\b", q):
        out.add("moneyline")
    if re.search(r"\bbtts\b", q) or re.search(r"both\s+teams?\s+(?:to\s+)?scor", q):
        out.add("btts")
    return out


def filter_events_by_mentioned_teams(user_q: str, events: List[Dict]) -> List[Dict]:
    q = re.sub(r"[^a-z0-9\s]", " ", user_q.lower())
    mentions: Set[str] = set()
    for ev in events:
        for t in [ev.get("home_team"), ev.get("away_team")]:
            if not t:
                continue
            aliases = team_name_aliases(str(t))
            if any(a and (a in q) for a in aliases):
                mentions.add(str(t))
    if not mentions:
        return events
    out = [ev for ev in events if (ev.get("home_team") in mentions) or (ev.get("away_team") in mentions)]
    return out or events


def query_mentions_any_event_team(user_q: str, events: List[Dict]) -> bool:
    q = re.sub(r"[^a-z0-9\s]", " ", user_q.lower())
    for ev in events:
        for t in [ev.get("home_team"), ev.get("away_team")]:
            if not t:
                continue
            aliases = team_name_aliases(str(t))
            if any(a and (a in q) for a in aliases):
                return True
    return False


def extract_mentioned_event_teams(user_q: str, events: List[Dict]) -> Set[str]:
    q = re.sub(r"[^a-z0-9\s]", " ", user_q.lower())
    out: Set[str] = set()
    for ev in events:
        for t in [ev.get("home_team"), ev.get("away_team")]:
            if not t:
                continue
            aliases = team_name_aliases(str(t))
            if any(a and (a in q) for a in aliases):
                out.add(str(t))
    return out


def is_singular_fixture_request(user_q: str) -> bool:
    q = user_q.lower()
    if re.search(r"\b(games|matches|fixtures)\b", q):
        return False
    if re.search(r"\b(the|this|that)\s+(game|match)\b", q):
        return True
    if re.search(r"\bfor\s+the\s+.+\s+game\b", q):
        return True
    return False


def event_sort_key(ev: Dict) -> str:
    return str(ev.get("commence_time") or "")


def is_parlay_query(user_q: str) -> bool:
    q = user_q.lower()
    return bool(re.search(r"\b(parlay|accumulator|acca|combo|multi[- ]?bet|multi[- ]?leg)\b", q)
                or re.search(r"\b\d+[- ]?legs?\b", q))


def wants_totals_only_parlay(user_q: str) -> bool:
    q = user_q.lower()
    if not is_parlay_query(user_q):
        return False
    if "totals only" in q:
        return True
    if re.search(r"parlay.*\b(goal|goals|xg|total|totals)\b", q) and "total" in q:
        return True
    if re.search(r"parlay.*\b(corner|corners)\b", q) and "total" in q:
        return True
    if re.search(r"parlay.*\b(card|cards|booking|bookings)\b", q) and "total" in q:
        return True
    return False


def is_comparison_query(user_q: str) -> bool:
    q = user_q.lower()
    # Explicit comparison phrases
    has_compare = bool(
        re.search(r"(which team|who)\s+.*(more|higher|most|win|get|lead)", q)
        or re.search(r"(which team|who)\s+.*(corner|card|shot|sot|foul)", q)
    )
    if has_compare:
        return True
    # "for all games" is only a comparison if there's a comparison signal
    # (not just "for all games, what are the lines" which is totals/team totals)
    if re.search(r"for all.*games", q):
        has_compare_signal = bool(
            re.search(r"\b(more|most|higher|fewer|less|which team|who gets|who has|compare|versus)\b", q)
        )
        has_totals_signal = bool(
            re.search(r"\b(line|total|per[- ]team|team\s+line|team\s+total|over|under)\b", q)
        )
        # Only treat as comparison if there's a compare signal and no totals signal
        return has_compare_signal and not has_totals_signal
    return False


def wants_deep_explanation(user_q: str) -> bool:
    q = user_q.lower()
    return bool(
        re.search(r"\b(why|reason|reasoning|explain|explanation)\b", q)
        or re.search(r"\b(solid|detailed|deep|thorough)\b", q)
    )


def requested_stat_group(user_q: str) -> Optional[str]:
    q = user_q.lower()
    if "goal" in q or re.search(r"\bxg\b", q):
        return "goals"
    if "corner" in q:
        return "corners"
    if "card" in q or "booking" in q:
        return "cards"
    if re.search(r"\b(shot|sot)\b", q) or "shots on target" in q:
        return "sot"
    return None


def is_totals_line_query(user_q: str) -> bool:
    q = user_q.lower()
    has_stat = bool(re.search(r"\b(goal|xg|corner|card|booking|shot|sot)s?\b", q) or "shots on target" in q)
    asks_total_or_line = bool(
        re.search(r"\b(how many|total|totals|line|which line|what line|should i take|take)\b", q)
        or re.search(r"\b(over|under)\b", q)
        or re.search(r"\b(interval|range|band)s?\b", q)
    )
    return has_stat and asks_total_or_line


def is_spreads_line_query(user_q: str) -> bool:
    q = user_q.lower()
    has_spread = bool(re.search(r"\b(spread|handicap|score line)\b", q))
    asks_line = bool(
        re.search(r"\b(which|what|best|recommend|should i take|take|pick|advice)\b", q)
        or re.search(r"\b(line|give me)\b", q)
    )
    return has_spread and asks_line


def is_moneyline_query(user_q: str) -> bool:
    q = user_q.lower()
    if re.search(r"\bmoney\s*line\b", q):
        return True
    if re.search(r"\b1x2\b", q):
        return True
    if re.search(r"\bmatch\s+winner\b", q):
        return True
    if re.search(r"\bwho\s+(?:will|would|is\s+going\s+to)\s+win\b", q):
        return True
    if re.search(r"\bwin\s+(?:the\s+)?(?:match|game)\b", q):
        return True
    if re.search(r"\bh2h\b", q):
        return True
    if re.search(r"\bthree[- ]?way\b", q):
        return True
    if re.search(r"\bwin\s*,?\s*(?:draw|lose)\b", q):
        return True
    if re.search(r"\bmatch\s+result\b", q):
        return True
    return False


def is_correct_score_query(user_q: str) -> bool:
    q = user_q.lower()
    if re.search(r"\bcorrect\s+score", q):
        return True
    if re.search(r"\bexact\s+score", q):
        return True
    if re.search(r"\bfinal\s+score\b.*\bpredict", q) or re.search(r"\bpredict.*\bfinal\s+score\b", q):
        return True
    if re.search(r"\bscore\s*line", q):
        return True
    if re.search(r"\bwhat\s+(?:will|would)\s+the\s+score\s+be\b", q):
        return True
    if re.search(r"\bmost\s+likely\s+score\b", q):
        return True
    if re.search(r"\b\d+\s*-\s*\d+\s+(?:result|prediction|correct)\b", q):
        return True
    return False


def is_btts_query(user_q: str) -> bool:
    q = user_q.lower()
    return bool(re.search(r"\bbtts\b", q) or re.search(r"both\s+teams?\s+(?:to\s+)?scor", q))


def is_team_totals_query(user_q: str) -> bool:
    q = user_q.lower()
    has_team = bool(re.search(
        r"\b(team\s+line|team\s+total|individual|per[- ]team|each\s+team|per\s+side)\b", q
    ))
    has_stat = bool(re.search(r"\b(corner|card|booking)s?\b", q))
    return has_team and has_stat


def is_player_prop_query(user_q: str) -> bool:
    q = user_q.lower()
    # "shots on target" / "sot" can be team-level (comparison / totals) or
    # player-level.  Only treat as player prop when there is NO team-level
    # signal (comparison or totals patterns).
    sot_hit = bool(re.search(r"\bshots?\s+on\s+target\b|\bsot\b", q))
    team_level_signal = bool(
        re.search(r"\b(which team|for all|for every|for each)\b", q)
        or re.search(r"\b(over|under|o/u|total|totals|line)\b", q)
        or re.search(r"\b(more|higher|most|winner)\b", q)
    )
    if sot_hit and team_level_signal:
        # Suppress the player-prop classification; let comparison / totals
        # handlers pick it up instead.
        sot_hit = False
    has_player_keyword = bool(
        re.search(
            r"\b(player|goalscorer|goal\s*scorer|anytime|first\s*scorer"
            r"|assist[s]?"
            r"|who\s+scores|who\s+gets?\s+booked|who\s+gets?\s+carded"
            r"|player\s+prop|player\s+card|player\s+booking"
            r"|which\s+player)",
            q,
        )
    )
    return has_player_keyword or sot_hit


def projected_total_corners(home: str, away: str, league: str, knockout_ctx: Optional[KnockoutContext] = None) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    pw = SCORING_WEIGHTS["projection"]
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

    if season_total is not None and recent_total is not None:
        # Dynamic blend: increase recent weight when form diverges from season avg
        base_season_w, base_recent_w = _stat_blend("corners")
        divergence = abs(recent_total - season_total) / max(season_total, 1.0)
        # When divergence > 20%, shift blend toward recent (up to 60/40)
        if divergence > 0.20:
            shift = min(0.30, (divergence - 0.20) * 1.0)  # max 30% shift
            blend_season = base_season_w - shift
            blend_recent = base_recent_w + shift
        else:
            blend_season = base_season_w
            blend_recent = base_recent_w
        blended = blend_season * season_total + blend_recent * recent_total
    elif season_total is not None:
        blended = season_total
    elif recent_total is not None:
        blended = recent_total
    else:
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


def projected_total_cards(
    home: str,
    away: str,
    league: str,
    knockout_ctx: Optional[KnockoutContext] = None,
    ref_mod: Optional[RefereeModifier] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float], RefereeModifier]:
    pw = SCORING_WEIGHTS["projection"]
    rw = SCORING_WEIGHTS["referee"]
    hm = _profile_meta(home, league)
    am = _profile_meta(away, league)

    if ref_mod is None:
        ref_mod = get_referee_modifier(home, away, league,
                                       weight=rw.get("modifier_weight", 0.25))

    # Season projection (referee modifier applied inside projected_cards)
    h_proj, a_proj = projected_cards(hm, am, referee_modifier=ref_mod.multiplier)
    season_total = (h_proj + a_proj) if (h_proj is not None and a_proj is not None) else None

    hr = _recent_stats(home, league, last_n=6)
    ar = _recent_stats(away, league, last_n=6)
    h_recent_cards = hr.get("cards_avg")
    a_recent_cards = ar.get("cards_avg")
    h_recent_opp_induced = ar.get("cards_induced_avg")
    a_recent_opp_induced = hr.get("cards_induced_avg")
    # Blend each team's recent cards with opponent's recent card-inducing tendency
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

    if season_total is not None and recent_total is not None:
        # Dynamic blend: increase recent weight when form diverges from season avg
        base_season_w, base_recent_w = _stat_blend("cards")
        divergence = abs(recent_total - season_total) / max(season_total, 1.0)
        # When divergence > 20%, shift blend toward recent (up to 60/40)
        if divergence > 0.20:
            shift = min(0.30, (divergence - 0.20) * 1.0)  # max 30% shift
            blend_season = base_season_w - shift
            blend_recent = base_recent_w + shift
        else:
            blend_season = base_season_w
            blend_recent = base_recent_w
        blended = blend_season * season_total + blend_recent * recent_total
    elif season_total is not None:
        blended = season_total
    elif recent_total is not None:
        blended = recent_total
    else:
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

    # Foul-based card estimate: predicted_fouls × referee cards_per_foul
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
    """Per-team corner projection with season, recent, and style-pressure anchors."""
    tlw = SCORING_WEIGHTS.get("team_lines", {})
    cmw = tlw.get("corners_model", {})
    csw = tlw.get("corners_style", {})
    hm = _profile_meta(team if is_home else opponent, league)
    am = _profile_meta(opponent if is_home else team, league)
    team_meta = hm if is_home else am
    opp_meta = am if is_home else hm

    h_proj, a_proj = projected_corners(hm, am)
    season_proj = h_proj if is_home else a_proj

    venue = "home" if is_home else "away"
    opp_venue = "away" if is_home else "home"
    recent_stats = _recent_stats(team, league, last_n=6, venue=venue)
    if recent_stats.get("corners_for_avg") is None:
        recent_stats = _recent_stats(team, league, last_n=6)
    opp_recent = _recent_stats(opponent, league, last_n=6, venue=opp_venue)
    if opp_recent.get("corners_against_avg") is None:
        opp_recent = _recent_stats(opponent, league, last_n=6)

    recent_team_for = recent_stats.get("corners_for_avg")
    recent_opp_allow = opp_recent.get("corners_against_avg")
    recent_own_w = tlw.get("corners_recent_own", 0.50)
    if recent_team_for is not None and recent_opp_allow is not None:
        recent_proj = recent_own_w * recent_team_for + (1.0 - recent_own_w) * recent_opp_allow
    else:
        recent_proj = recent_team_for if recent_team_for is not None else recent_opp_allow

    style_proj = _weighted_sum([
        (csw.get("shots", 0.0), recent_stats.get("shots_for_avg") or safe_float(team_meta.get("shots_for_pm"))),
        (csw.get("sot", 0.0), recent_stats.get("sot_for_avg") or safe_float(team_meta.get("sot_for_pm"))),
        (csw.get("control", 0.0), recent_stats.get("control_avg") or safe_float(team_meta.get("control_index"))),
        (csw.get("dominance", 0.0), safe_float(team_meta.get("dominance_index"))),
        (csw.get("possession", 0.0), recent_stats.get("possession_avg") or safe_float(team_meta.get("possession"))),
        (
            csw.get("opp_sot_against", 0.0),
            opp_recent.get("sot_against_avg")
            or safe_float(opp_meta.get("sot_against_away_pm" if is_home else "sot_against_home_pm"))
            or safe_float(opp_meta.get("sot_against_pm")),
        ),
    ])

    blended = _weighted_component_blend([
        (season_proj, cmw.get("season", 0.43)),
        (recent_proj, cmw.get("recent", 0.34)),
        (style_proj, cmw.get("style", 0.23)),
    ])
    if blended is None:
        return None, None, None

    trend = recent_stats.get("corners_for_slope")
    if trend is not None:
        trend_cap = cmw.get("trend_cap", 0.25)
        blended += _clamp(cmw.get("trend_nudge", 0.06) * trend, -trend_cap, trend_cap)

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

    return blended, season_proj, recent_proj


def projected_team_cards(
    team: str, opponent: str, league: str, is_home: bool,
    ref_mod: "RefereeModifier" = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Per-team card projection from match-total anchor + team share + stabilizer."""
    tlw = SCORING_WEIGHTS.get("team_lines", {})
    cdw = tlw.get("cards_direct", {})
    cow = tlw.get("cards_output", {})
    csw = tlw.get("cards_share", {})
    hm = _profile_meta(team if is_home else opponent, league)
    am = _profile_meta(opponent if is_home else team, league)
    team_meta = hm if is_home else am
    opp_meta = am if is_home else hm
    home_name = team if is_home else opponent
    away_name = opponent if is_home else team

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

    recent_stats = _recent_stats(team, league, last_n=6)
    opp_recent = _recent_stats(opponent, league, last_n=6)
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

    total_blended, season_total, recent_total, ref_mod = projected_total_cards(
        home_name, away_name, league, ref_mod=ref_mod
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
    blended = _weighted_component_blend([
        (share_anchor, cow.get("share_weight", 0.50)),
        (stabilizer, cow.get("stabilizer_weight", 0.50)),
    ])
    if blended is None:
        return None, None, None

    return blended, season_proj, recent_proj


def _trend_adjustment(hr: Dict, ar: Dict, slope_key: str) -> float:
    """Compute trend adjustment from home/away recent slopes.
    Positive slope = team improving → nudge projection up."""
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

    if season_total is not None and recent_total is not None:
        # Dynamic blend: increase recent weight when form diverges from season avg
        base_season_w, base_recent_w = _stat_blend("goals")
        divergence = abs(recent_total - season_total) / max(season_total, 1.0)
        # When divergence > 20%, shift blend toward recent (up to 60/40)
        if divergence > 0.20:
            shift = min(0.30, (divergence - 0.20) * 1.0)  # max 30% shift
            blend_season = base_season_w - shift
            blend_recent = base_recent_w + shift
        else:
            blend_season = base_season_w
            blend_recent = base_recent_w
        blended = blend_season * season_total + blend_recent * recent_total
    elif season_total is not None:
        blended = season_total
    elif recent_total is not None:
        blended = recent_total
    else:
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


def projected_btts_prob(home: str, away: str, league: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Project P(BTTS Yes) = P(Home >= 1) * P(Away >= 1) using independent Poisson/NegBin.
    Returns (p_btts_yes, h_proj, a_proj, h_season, a_season).
    """
    pw = SCORING_WEIGHTS["projection"]
    hm = _profile_meta(home, league)
    am = _profile_meta(away, league)

    h_season, a_season = projected_goals(hm, am)

    hr = _recent_stats(home, league, last_n=6)
    ar = _recent_stats(away, league, last_n=6)
    h_xg = hr.get("xg_for_avg")
    a_xg = ar.get("xg_for_avg")

    def _blend(season_val, recent_val):
        if season_val is not None and recent_val is not None:
            return pw["blend_season"] * season_val + pw["blend_recent"] * recent_val
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

    # Dixon-Coles bivariate model for BTTS (accounts for low-score correlation)
    try:
        from prob_models import dixon_coles_btts_prob
        p_btts_yes = dixon_coles_btts_prob(h_proj, a_proj)
    except ImportError:
        # Fallback to independent Poisson
        p_home_scores = over_prob(h_proj, 0.5, h_var)
        p_away_scores = over_prob(a_proj, 0.5, a_var)
        p_btts_yes = p_home_scores * p_away_scores

    return p_btts_yes, h_proj, a_proj, h_season, a_season


def projected_correct_score_probs(
    home: str, away: str, league: str, max_goals: int = 6,
) -> Tuple[Optional[Dict[Tuple[int, int], float]], Optional[float], Optional[float]]:
    """
    Compute P(Home=i, Away=j) for all i,j in [0, max_goals] using independent
    Poisson/NegBin per team.  Returns (prob_matrix, h_proj, a_proj).
    """
    pw = SCORING_WEIGHTS["projection"]
    hm = _profile_meta(home, league)
    am = _profile_meta(away, league)
    h_season, a_season = projected_goals(hm, am)

    hr = _recent_stats(home, league, last_n=6)
    ar = _recent_stats(away, league, last_n=6)
    h_xg = hr.get("xg_for_avg")
    a_xg = ar.get("xg_for_avg")

    def _blend(season_val, recent_val):
        if season_val is not None and recent_val is not None:
            return pw["blend_season"] * season_val + pw["blend_recent"] * recent_val
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

    # Per-team variance for distribution selection (Bayesian blend: season + recent)
    h_var = get_blended_variance(home, league, "goals_var")
    a_var = get_blended_variance(away, league, "goals_var")

    def _team_pmf(lam, var, k):
        if var is not None and var > lam and lam > 0:
            r, p = negbin_from_mean_var(lam, var)
            return negbin_pmf(k, r, p)
        return poisson_pmf(k, lam)

    # Dixon-Coles bivariate model (corrects for low-score correlation)
    try:
        from prob_models import dixon_coles_scoreline_matrix
        matrix = dixon_coles_scoreline_matrix(h_proj, a_proj, rho=-0.10, max_goals=max_goals)
        probs: Dict[Tuple[int, int], float] = {}
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                probs[(i, j)] = matrix[i][j]
    except ImportError:
        # Fallback to independent Poisson
        probs = {}
        for i in range(max_goals + 1):
            p_h = _team_pmf(h_proj, h_var, i)
            for j in range(max_goals + 1):
                p_a = _team_pmf(a_proj, a_var, j)
                probs[(i, j)] = p_h * p_a

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
    ha = proj_w.get("home_advantage", 0.0) * 2  # full spread: +ha home, -ha away = 2*ha diff

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


def extract_total_line_options(event: Dict, stat_group: str) -> List[Dict]:
    out: List[Dict] = []
    for bm in event.get("bookmakers", []) or []:
        bm_name = str(bm.get("title") or "")
        for mk in bm.get("markets", []) or []:
            key = str(mk.get("key") or "")
            k = key.lower()
            if not is_full_game_market(key):
                continue
            if stat_group == "corners":
                if ("corner" not in k) or ("total" not in k):
                    continue
                # Exclude per-team corner lines — we want match totals only
                if "team_total" in k or "home_corner" in k or "away_corner" in k:
                    continue
            elif stat_group == "cards":
                if (("card" not in k) and ("booking" not in k) and ("yellow" not in k)) or ("total" not in k):
                    continue
                # Exclude per-team card lines — we want match totals only
                if "team_total" in k or "home_card" in k or "away_card" in k or "home_team_total" in k or "away_team_total" in k:
                    continue
            elif stat_group == "goals":
                # Match totals goals markets. Avoid corners/cards/teams/goalscorer props.
                if "total" not in k:
                    continue
                if "corner" in k or "card" in k or "booking" in k:
                    continue
                if "team_total" in k or "player" in k or "goalscorer" in k:
                    continue
                if "shot" in k or "yellow" in k:
                    continue
            elif stat_group == "sot":
                if "shot" not in k:
                    continue
                # Must be a totals-style market (over/under), not 1x2 or handicap
                if "total" not in k and "over" not in k and "under" not in k:
                    continue
                # Exclude per-team SoT and 1x2/handicap shot markets
                if "team_total" in k or "home" in k or "away" in k:
                    continue
                if "1x2" in k or "handicap" in k or "double" in k:
                    continue
            else:
                continue
            for outcome in mk.get("outcomes", []) or []:
                name = str(outcome.get("name") or "").lower().strip()
                if name not in {"over", "under"}:
                    continue
                point = safe_float(outcome.get("point"))
                price = safe_float(outcome.get("price"))
                if point is None or price is None or price <= 1.0:
                    continue
                out.append(
                    {
                        "market_key": key,
                        "bookmaker": bm_name,
                        "side": name,
                        "point": point,
                        "odds": price,
                    }
                )
    return out


def extract_market_implied_total(event: Dict, stat_group: str) -> Optional[float]:
    """Extract the market-implied total from bookmaker over/under odds.

    Finds the over/under line where the vig-removed P(over) is closest to 50%.
    That line approximates the market's expected total for the stat.

    Returns None if no suitable lines are found.
    """
    options = extract_total_line_options(event, stat_group)
    if not options:
        return None

    # Group by (point, bookmaker) to find over/under pairs
    pairs: Dict[Tuple[float, str], Dict[str, float]] = {}
    for opt in options:
        pt = float(opt["point"])
        bm = str(opt.get("bookmaker", ""))
        key = (pt, bm)
        if key not in pairs:
            pairs[key] = {}
        pairs[key][opt["side"]] = float(opt["odds"])

    best_line: Optional[float] = None
    best_dist: float = 999.0

    for (pt, _bm), sides in pairs.items():
        over_odds = sides.get("over")
        under_odds = sides.get("under")
        if over_odds and under_odds:
            # Use vig-removed probabilities
            p_over, _p_under = remove_vig_two_way(over_odds, under_odds)
        elif over_odds:
            p_over = implied_prob(over_odds)
        elif under_odds:
            p_over = 1.0 - implied_prob(under_odds)
        else:
            continue

        dist = abs(p_over - 0.50)
        if dist < best_dist:
            best_dist = dist
            best_line = pt

    if best_line is None:
        return None

    # Refine: if P(over best_line) > 50%, actual total is slightly above the line
    # If P(over best_line) < 50%, actual total is slightly below the line
    # Use the closest-to-50% line as the market-implied total
    return best_line


def extract_team_total_line_options(
    event: Dict, team_name: str, stat_group: str,
) -> List[Dict]:
    """Extract over/under lines for a specific team from team_totals markets.

    The Odds API only offers team_totals for goals — corners/cards team totals
    don't exist, so this returns empty for those stat groups (triggering model-only
    fallback in the renderer).
    """
    out: List[Dict] = []
    aliases = team_name_aliases(team_name)
    for bm in event.get("bookmakers", []) or []:
        bm_name = str(bm.get("title") or "")
        for mk in bm.get("markets", []) or []:
            key = str(mk.get("key") or "")
            k = key.lower()
            if not is_full_game_market(key):
                continue
            if "team_total" not in k:
                continue
            if stat_group == "corners" and "corner" not in k:
                continue
            if stat_group == "cards" and "card" not in k and "booking" not in k:
                continue
            # Determine if this market is for the requested team.
            # API-Football keys: team_totals_home_corners, team_totals_away_cards, etc.
            # Match "home" in key to the event's home team, "away" to away team.
            home_in_event = str(event.get("home_team") or "").lower().strip()
            away_in_event = str(event.get("away_team") or "").lower().strip()
            team_lower = team_name.lower().strip()

            key_is_home = "home" in k
            key_is_away = "away" in k
            team_is_home = (team_lower == home_in_event
                            or any(a in home_in_event for a in aliases))
            team_is_away = (team_lower == away_in_event
                            or any(a in away_in_event for a in aliases))

            # If key says "home" but team is away (or vice versa), skip
            if key_is_home and not team_is_home:
                continue
            if key_is_away and not team_is_away:
                continue
            # If key doesn't specify home/away, fall back to old text matching
            if not key_is_home and not key_is_away:
                # Legacy path for Odds API format
                pass

            for outcome in mk.get("outcomes", []) or []:
                name = str(outcome.get("name") or "").strip()
                side = None
                if name.lower() in ("over", "under"):
                    side = name.lower()
                if side is None:
                    desc = str(outcome.get("description") or "").strip()
                    combined_text = f"{name} {desc}".lower()
                    if "over" in combined_text:
                        side = "over"
                    elif "under" in combined_text:
                        side = "under"
                if side is None:
                    continue
                point = safe_float(outcome.get("point"))
                price = safe_float(outcome.get("price"))
                if point is None or price is None or price <= 1.0:
                    continue
                out.append({
                    "market_key": key, "bookmaker": bm_name,
                    "side": side, "point": point, "odds": price,
                    "team": team_name,
                })
    return out


def choose_best_total_line(options: List[Dict], projection_total: float,
                           combined_var: Optional[float] = None) -> Optional[Dict]:
    if not options:
        return None
    tw = SCORING_WEIGHTS["totals_line"]
    pw = SCORING_WEIGHTS["prob"]

    filtered: List[Dict] = []
    for opt in options:
        odds = safe_float(opt.get("odds"))
        if odds is None or odds < tw["min_odds_hard"]:
            continue
        filtered.append(opt)
    if not filtered:
        return None

    primary = [o for o in filtered if "alternate" not in str(o.get("market_key") or "").lower()]
    options_use = primary if primary else filtered

    # Pre-compute vig-free implied probabilities for over/under pairs
    _pair_map: Dict[Tuple, Dict[str, float]] = {}
    for _o in options_use:
        _bm = str(_o.get("bookmaker") or "")
        _pt = safe_float(_o.get("point"))
        _sd = str(_o.get("side") or "").lower()
        _od = safe_float(_o.get("odds"))
        if _pt is not None and _od is not None:
            _pair_map.setdefault((_bm, _pt), {})[_sd] = _od
    _fair_implied: Dict[Tuple, float] = {}
    for (_bm, _pt), _sides in _pair_map.items():
        _ov = _sides.get("over")
        _un = _sides.get("under")
        if _ov is not None and _un is not None:
            _f_ov, _f_un = remove_vig_two_way(_ov, _un)
            _fair_implied[(_bm, _pt, "over")] = _f_ov
            _fair_implied[(_bm, _pt, "under")] = _f_un

    best = None
    best_score = None
    for opt in options_use:
        side = str(opt.get("side") or "")
        point = safe_float(opt.get("point"))
        odds = safe_float(opt.get("odds"))
        if point is None or odds is None:
            continue
        edge = projection_total - point if side == "over" else point - projection_total

        dist = abs(projection_total - point)
        proximity_penalty = dist * tw["proximity_penalty"]
        edge_term = edge * tw["edge_weight"]

        if odds < tw["min_odds_preferred"]:
            odds_term = -tw["short_price_penalty"] * (tw["min_odds_preferred"] - odds)
        elif tw["ideal_odds_low"] <= odds <= tw["ideal_odds_high"]:
            odds_term = tw["ideal_range_bonus"] * (odds - 1.0)
        else:
            odds_term = tw["outside_ideal_bonus"] * (odds - 1.0)

        far_penalty = 0.0
        if dist >= tw["far_threshold"]:
            far_penalty = (dist - tw["far_threshold"]) * tw["far_penalty"]

        # --- Probability-aware scoring (vig-adjusted) ---
        model_p = (over_prob(projection_total, point, combined_var) if side == "over"
                   else under_prob(projection_total, point, combined_var))
        fair_key = (str(opt.get("bookmaker") or ""), point, side.lower())
        implied_p = _fair_implied.get(fair_key) or implied_prob(odds)
        val_edge = value_edge(model_p, implied_p)
        ev = expected_value(model_p, odds)

        prob_term = val_edge * pw["value_edge_weight"]
        ev_penalty = abs(ev) * pw["negative_ev_penalty"] if ev < 0 else 0.0

        # Model probability is the DOMINANT signal — a bet that cashes is worth
        # more than a bet with a big edge that doesn't. Scale it aggressively
        # so an 85% prob bet always beats a 55% prob bet regardless of edge.
        model_prob_bonus = (model_p - 0.50) * pw["model_prob_weight"]

        # Minimum probability gate — reject bets below 55% entirely
        if model_p < pw["min_model_prob"]:
            model_prob_bonus -= 5.0  # heavy penalty, effectively disqualifies

        score = (model_prob_bonus + prob_term + odds_term
                 + edge_term * 0.3 - proximity_penalty * 0.3
                 - far_penalty - ev_penalty)

        # Attach probability data for rendering
        opt["_model_prob"] = model_p
        opt["_implied_prob"] = implied_p
        opt["_value_edge"] = val_edge
        opt["_ev"] = ev

        if best is None or score > float(best_score):
            best = opt
            best_score = score
    return best


def _best_model_only_line(projection: float,
                          variance: Optional[float] = None) -> Tuple[str, float, float]:
    """Pick the strongest (side, line, model_prob) for a model-only recommendation.

    Evaluates half-integer lines (X.5, no push) within ±1.0 of the projection
    and returns the side+line combination with the highest model probability.
    """
    best_side, best_line, best_prob = "Over", round(projection * 2) / 2, 0.5

    base = int(projection)
    candidates = [base - 0.5, base + 0.5, base + 1.5]
    if base >= 2:
        candidates.append(base - 1.5)

    for line in candidates:
        if line < 0.5 or abs(line - projection) > 1.0:
            continue
        p_over = over_prob(projection, line, variance)
        p_under = 1.0 - p_over
        side = "Over" if p_over >= p_under else "Under"
        prob = max(p_over, p_under)
        if prob > best_prob:
            best_side, best_line, best_prob = side, line, prob

    return best_side, best_line, best_prob


def _best_model_only_spread(proj_diff: float, home: str, away: str) -> Tuple[str, str, float]:
    """Pick the strongest (team, handicap_display, cover_prob) for model-only spread.

    Evaluates half-integer handicaps within ±1.0 of |proj_diff|.
    """
    margin_std = SCORING_WEIGHTS["prob"]["margin_std_default"]
    best_team = home if proj_diff >= 0 else away
    best_hc = "-0.5" if proj_diff >= 0 else "+0.5"
    best_prob = 0.5

    for handicap_abs in [0.5, 1.5, 2.5]:
        if abs(handicap_abs - abs(proj_diff)) > 1.0:
            continue
        # Home giving -handicap
        p_home = 1.0 - _normal_cdf((handicap_abs - proj_diff) / margin_std)
        # Away receiving +handicap
        p_away = 1.0 - _normal_cdf((proj_diff - handicap_abs) / margin_std)

        if p_home > best_prob:
            best_team, best_hc, best_prob = home, f"{-handicap_abs:+g}", p_home
        if p_away > best_prob:
            best_team, best_hc, best_prob = away, f"+{handicap_abs:g}", p_away

    return best_team, best_hc, best_prob


def extract_btts_odds(event: Dict) -> List[Dict]:
    """Extract BTTS Yes/No odds from bookmaker data for one event."""
    out: List[Dict] = []
    for bm in event.get("bookmakers", []) or []:
        bm_name = str(bm.get("title") or "")
        for mk in bm.get("markets", []) or []:
            key = str(mk.get("key") or "")
            if market_group_from_key(key) != "btts":
                continue
            if not is_full_game_market(key):
                continue
            for outcome in mk.get("outcomes", []) or []:
                name = str(outcome.get("name") or "").strip()
                price = safe_float(outcome.get("price"))
                if not name or price is None or price <= 1.0:
                    continue
                out.append({
                    "market_key": key,
                    "bookmaker": bm_name,
                    "side": name,
                    "odds": price,
                })
    return out


def choose_best_btts_side(options: List[Dict], p_btts_yes: float) -> Optional[Dict]:
    """Score BTTS Yes/No options and return the best one with probability data attached.

    BTTS is a directional bet — the model's conviction (probability) must drive the
    recommendation, not just value edge. A 72% BTTS Yes probability should almost always
    recommend Yes, regardless of whether No has a marginally better edge. Value edge is
    used as a tiebreaker and to flag when the model-preferred side is overpriced.
    """
    if not options:
        return None
    pw = SCORING_WEIGHTS["prob"]

    # Build vig-free implied probs from Yes/No pairs per bookmaker
    _pair_map: Dict[str, Dict[str, float]] = {}
    for o in options:
        bm = str(o.get("bookmaker") or "")
        side = str(o.get("side") or "").lower().strip()
        odds = safe_float(o.get("odds"))
        if odds is not None and side:
            _pair_map.setdefault(bm, {})[side] = odds
    _fair_implied: Dict[Tuple[str, str], float] = {}
    for bm, sides in _pair_map.items():
        yes_odds = sides.get("yes")
        no_odds = sides.get("no")
        if yes_odds is not None and no_odds is not None:
            f_yes, f_no = remove_vig_two_way(yes_odds, no_odds)
            _fair_implied[(bm, "yes")] = f_yes
            _fair_implied[(bm, "no")] = f_no

    # Step 1: Determine the model's preferred side based on probability
    model_preferred_side = "yes" if p_btts_yes >= 0.50 else "no"
    model_conviction = max(p_btts_yes, 1.0 - p_btts_yes)

    # Step 2: Score all options, but heavily weight model conviction
    best = None
    best_score = None
    for opt in options:
        side = str(opt.get("side") or "").lower().strip()
        odds = safe_float(opt.get("odds"))
        if not side or odds is None or odds <= 1.0:
            continue

        model_p = p_btts_yes if side == "yes" else (1.0 - p_btts_yes)
        bm = str(opt.get("bookmaker") or "")
        implied_p = _fair_implied.get((bm, side)) or implied_prob(odds)
        val_edge = value_edge(model_p, implied_p)
        ev = expected_value(model_p, odds)

        # Score: model probability is the PRIMARY driver (weight 4.0)
        # Value edge is secondary (weight 1.0)
        # Penalize going against model conviction heavily
        conviction_score = (model_p - 0.50) * pw.get("model_prob_weight", 4.0)
        edge_bonus = val_edge * pw.get("value_edge_weight", 2.0) if val_edge > 0 else 0.0
        ev_penalty = abs(ev) * pw["negative_ev_penalty"] if ev < 0 else 0.0

        # Strong penalty for recommending opposite of model conviction
        against_model_penalty = 0.0
        if side != model_preferred_side and model_conviction >= 0.55:
            against_model_penalty = (model_conviction - 0.50) * 6.0

        score = conviction_score + edge_bonus - ev_penalty - against_model_penalty

        opt["_model_prob"] = model_p
        opt["_implied_prob"] = implied_p
        opt["_value_edge"] = val_edge
        opt["_ev"] = ev

        if best is None or score > best_score:
            best = opt
            best_score = score
    return best


def extract_correct_score_odds(event: Dict) -> Dict[Tuple[int, int], List[Dict]]:
    """Extract correct score odds from event bookmaker data.

    Returns {(home_goals, away_goals): [{"odds": float, "bookmaker": str, "key": str}, ...]}.
    """
    result: Dict[Tuple[int, int], List[Dict]] = {}
    for bm in event.get("bookmakers", []) or []:
        bm_name = str(bm.get("title") or "")
        for mk in bm.get("markets", []) or []:
            key = str(mk.get("key") or "").lower()
            if "correctscore" not in key and "correct_score" not in key and "exactscore" not in key:
                continue
            for oc in mk.get("outcomes", []) or []:
                name = str(oc.get("name") or "")
                odds = safe_float(oc.get("price"))
                if odds is None or odds <= 1.0:
                    continue
                m = re.match(r"(\d+)\s*[-:]\s*(\d+)", name.strip())
                if not m:
                    continue
                score = (int(m.group(1)), int(m.group(2)))
                result.setdefault(score, []).append({
                    "odds": odds,
                    "bookmaker": bm_name,
                    "key": key,
                })
    return result


def choose_best_correct_scores(
    model_probs: Dict[Tuple[int, int], float],
    market_odds: Dict[Tuple[int, int], List[Dict]],
    top_n: int = 5,
) -> List[Dict]:
    """Rank correct score bets by model probability + value edge.  Returns up to *top_n*."""
    candidates: List[Dict] = []
    sorted_scores = sorted(model_probs.items(), key=lambda x: x[1], reverse=True)

    for score, model_p in sorted_scores[:15]:
        entry: Dict = {
            "score": score,
            "score_str": f"{score[0]}-{score[1]}",
            "model_prob": model_p,
            "best_odds": None,
            "bookmaker": None,
            "implied_prob": None,
            "value_edge": None,
            "ev": None,
        }
        if score in market_odds and market_odds[score]:
            best = max(market_odds[score], key=lambda x: x["odds"])
            entry["best_odds"] = best["odds"]
            entry["bookmaker"] = best["bookmaker"]
            entry["implied_prob"] = implied_prob(best["odds"])
            entry["value_edge"] = value_edge(model_p, entry["implied_prob"])
            entry["ev"] = expected_value(model_p, best["odds"])
        candidates.append(entry)

    def _rank(c):
        base = c["model_prob"] * 10.0
        if c["value_edge"] is not None:
            base += max(c["value_edge"], 0) * 5.0
        if c["ev"] is not None and c["ev"] > 0:
            base += c["ev"] * 2.0
        return -base

    candidates.sort(key=_rank)
    return candidates[:top_n]


def extract_moneyline_odds(event: Dict) -> List[Dict]:
    """Extract home/draw/away moneyline odds from event bookmaker data."""
    home = str(event.get("home_team") or "")
    away = str(event.get("away_team") or "")
    results: List[Dict] = []
    for bm in event.get("bookmakers", []) or []:
        bm_name = str(bm.get("title") or "")
        for mk in bm.get("markets", []) or []:
            key = str(mk.get("key") or "").lower()
            if market_group_from_key(key) != "moneyline":
                continue
            if not is_full_game_market(key):
                continue
            for oc in mk.get("outcomes", []) or []:
                name = str(oc.get("name") or "")
                odds = safe_float(oc.get("price"))
                if odds is None or odds <= 1.0:
                    continue
                name_l = name.lower().strip()
                if name_l == "draw" or name_l == "x":
                    side, team = "draw", "Draw"
                elif name_l == home.lower().strip() or name_l.startswith("home") or name_l == "1":
                    side, team = "home", home
                elif name_l == away.lower().strip() or name_l.startswith("away") or name_l == "2":
                    side, team = "away", away
                else:
                    continue
                results.append({"side": side, "odds": odds, "bookmaker": bm_name, "team": team})
    return results


def choose_best_moneyline_side(
    p_home: float, p_draw: float, p_away: float,
    market_odds: List[Dict],
    home: str, away: str,
) -> Dict:
    """Score home/draw/away and pick the best value moneyline bet."""
    model_probs = {"home": p_home, "draw": p_draw, "away": p_away}
    team_labels = {"home": home, "draw": "Draw", "away": away}

    # Group odds by side, pick best (highest) per side
    best_per_side: Dict[str, Dict] = {}
    for entry in market_odds:
        s = entry["side"]
        if s not in best_per_side or entry["odds"] > best_per_side[s]["odds"]:
            best_per_side[s] = entry

    # Remove vig (3-way proportional)
    fair_probs: Dict[str, float] = {}
    if all(s in best_per_side for s in ("home", "draw", "away")):
        raw = {s: 1.0 / best_per_side[s]["odds"] for s in ("home", "draw", "away")}
        total = sum(raw.values())
        fair_probs = {s: raw[s] / total for s in ("home", "draw", "away")}

    candidates: List[Dict] = []
    for side in ("home", "draw", "away"):
        entry: Dict = {
            "side": side,
            "team": team_labels[side],
            "model_prob": model_probs[side],
            "best_odds": None,
            "bookmaker": None,
            "implied_prob": None,
            "fair_implied": None,
            "value_edge": None,
            "ev": None,
        }
        if side in best_per_side:
            bps = best_per_side[side]
            entry["best_odds"] = bps["odds"]
            entry["bookmaker"] = bps["bookmaker"]
            entry["implied_prob"] = implied_prob(bps["odds"])
            if side in fair_probs:
                entry["fair_implied"] = fair_probs[side]
                entry["value_edge"] = model_probs[side] - fair_probs[side]
            else:
                entry["value_edge"] = value_edge(model_probs[side], entry["implied_prob"])
            entry["ev"] = expected_value(model_probs[side], bps["odds"])
        candidates.append(entry)

    # Pick best: model probability is the primary driver, value edge is secondary.
    # A Draw at +2% edge should NOT beat a Home Win at 55% probability.
    # Score = model_prob * 3.0 + value_edge_bonus - negative_ev_penalty
    best = None
    best_score = -999.0
    for c in candidates:
        model_p = c["model_prob"]
        ve = c["value_edge"]
        ev_val = c["ev"]

        # Model conviction is the dominant signal
        conviction = (model_p - 0.33) * 3.0  # 3-way: baseline is 33%, not 50%

        # Value edge bonus (only when positive)
        edge_bonus = (ve * 2.0) if ve is not None and ve > 0 else 0.0

        # Penalty for negative EV
        ev_penalty = abs(ev_val) * 2.0 if ev_val is not None and ev_val < 0 else 0.0

        # Penalty for recommending Draw (draws are inherently harder to predict
        # and rarely the right recommendation unless model is very confident)
        draw_penalty = 0.0
        if c["side"] == "draw" and model_p < 0.32:
            draw_penalty = 0.5

        score = conviction + edge_bonus - ev_penalty - draw_penalty

        if score > best_score:
            best_score = score
            best = c

    return {"recommended": best, "all_sides": candidates}


def extract_spread_line_options(event: Dict) -> List[Dict]:
    """Extract available handicap/spread lines from bookmaker data for one event."""
    out: List[Dict] = []
    for bm in event.get("bookmakers", []) or []:
        bm_name = str(bm.get("title") or "")
        for mk in bm.get("markets", []) or []:
            key = str(mk.get("key") or "")
            k = key.lower()
            if not is_full_game_market(key):
                continue
            if "spread" not in k and "handicap" not in k:
                continue
            for oc in mk.get("outcomes", []) or []:
                name = str(oc.get("name") or "")
                point = oc.get("point")
                price = oc.get("price")
                try:
                    point = float(point) if point is not None else None
                    price = float(price) if price is not None else None
                except (TypeError, ValueError):
                    continue
                if point is None or price is None or price < 1.10:
                    continue
                out.append({
                    "market_key": key,
                    "bookmaker": bm_name,
                    "team": name,
                    "point": point,
                    "odds": price,
                })
    return out


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def choose_best_spread_line(options: List[Dict], projected_diff: float, home_team: str) -> Optional[Dict]:
    """Score available spread lines and return the best one."""
    if not options:
        return None
    sw = SCORING_WEIGHTS["spreads_line"]
    pw = SCORING_WEIGHTS["prob"]
    margin_std = pw["margin_std_default"]

    filtered = [o for o in options if (o.get("odds") or 0) >= sw["min_odds_hard"]]
    if not filtered:
        return None

    primary = [o for o in filtered if "alternate" not in str(o.get("market_key") or "").lower()]
    options_use = primary if primary else filtered

    # Pre-compute vig-free implied probabilities for spread pairs.
    # Each option has (team, signed_point, odds). The matching pair for
    # "TeamA at point X" is "TeamB at point -X" from the same bookmaker.
    # Build a lookup: (bookmaker, team, signed_point) -> odds
    _sp_lookup: Dict[Tuple, float] = {}
    for _o in options_use:
        _bm = str(_o.get("bookmaker") or "")
        _tm = str(_o.get("team") or "").lower().strip()
        _pt = _o.get("point") or 0
        _od = _o.get("odds")
        if _od is not None and _tm:
            _sp_lookup[(_bm, _tm, _pt)] = float(_od)

    # Build fair implied probabilities by pairing each option with its
    # counterpart: same bookmaker, same abs(point), opposite sign, different team.
    # Key: (bookmaker, signed_point, team) -> fair_implied_probability
    _sp_fair_signed: Dict[Tuple, float] = {}
    _sp_done: set = set()
    for _o in options_use:
        _bm = str(_o.get("bookmaker") or "")
        _tm = str(_o.get("team") or "").lower().strip()
        _pt = _o.get("point") or 0
        _od = float(_o.get("odds") or 0)

        key_self = (_bm, _pt, _tm)
        if key_self in _sp_done:
            continue

        opp_entry = None
        for (_bm2, _tm2, _pt2), _od2 in _sp_lookup.items():
            if (_bm2 == _bm
                    and _tm2 != _tm
                    and abs(abs(_pt2) - abs(_pt)) < 0.001
                    and ((_pt < 0 and _pt2 > 0) or (_pt > 0 and _pt2 < 0))):
                opp_entry = (_tm2, _pt2, _od2)
                break

        if opp_entry:
            _tm2, _pt2, _od2 = opp_entry
            _fa, _fb = remove_vig_two_way(_od, _od2)
            _sp_fair_signed[key_self] = _fa
            _sp_fair_signed[(_bm, _pt2, _tm2)] = _fb
            _sp_done.add(key_self)
            _sp_done.add((_bm, _pt2, _tm2))

    # Wrapper to look up fair implied by signed point
    def _get_sp_fair(bookmaker: str, point: float, team: str) -> Optional[float]:
        return _sp_fair_signed.get((bookmaker, point, team.lower().strip()))

    _sp_fair = {}  # legacy compat: not used in lookup below

    best, best_score = None, None
    for opt in options_use:
        team = str(opt.get("team") or "")
        point = opt["point"]
        odds = opt["odds"]

        is_home = team.lower().strip() == home_team.lower().strip()
        sign = 1.0 if is_home else -1.0
        edge = sign * projected_diff + point

        edge_term = edge * sw["edge_weight"]
        dist = abs(point)
        proximity_term = dist * sw["proximity_penalty"]

        if odds < sw["min_odds_preferred"]:
            odds_term = -sw["short_price_penalty"] * (sw["min_odds_preferred"] - odds)
        elif sw["ideal_odds_low"] <= odds <= sw["ideal_odds_high"]:
            odds_term = sw["ideal_range_bonus"] * (odds - 1.0)
        else:
            odds_term = sw["outside_ideal_bonus"] * (odds - 1.0)

        # --- Spread cover probability (normal approximation of goal difference, vig-adjusted) ---
        margin_mean = sign * projected_diff
        cover_threshold = -point
        model_p = 1.0 - _normal_cdf((cover_threshold - margin_mean) / margin_std)
        implied_p = _get_sp_fair(str(opt.get("bookmaker") or ""), point, team) or implied_prob(odds)
        val_edge = value_edge(model_p, implied_p)
        ev = expected_value(model_p, odds)

        prob_term = val_edge * pw["value_edge_weight"]
        ev_penalty = abs(ev) * pw["negative_ev_penalty"] if ev < 0 else 0.0

        # Model probability dominant — pick the line most likely to cash
        model_prob_bonus = (model_p - 0.50) * pw["model_prob_weight"]
        if model_p < pw["min_model_prob"]:
            model_prob_bonus -= 5.0

        score = (model_prob_bonus + prob_term + odds_term
                 + edge_term * 0.3 - proximity_term * 0.3
                 - ev_penalty)

        opt["_model_prob"] = model_p
        opt["_implied_prob"] = implied_p
        opt["_value_edge"] = val_edge
        opt["_ev"] = ev

        if best is None or score > best_score:
            best = opt
            best_score = score
    return best


def confidence_from_edge(edge: float, stat_group: Optional[str] = None,
                         model_prob: Optional[float] = None,
                         value_edge_pct: Optional[float] = None,
                         is_european: bool = False,
                         has_domestic_profile: bool = True) -> str:
    # Probability-calibrated confidence when available
    if model_prob is not None and value_edge_pct is not None:
        if model_prob >= 0.65 and value_edge_pct >= 0.08:
            conf = "high"
        elif model_prob >= 0.55 and value_edge_pct >= 0.03:
            conf = "medium"
        else:
            conf = "low"
    else:
        # Fallback: existing threshold logic
        cw = SCORING_WEIGHTS["confidence"]
        if stat_group == "cards":
            high, medium = cw.get("cards_gap_high", cw["edge_high"]), cw.get("cards_gap_medium", cw["edge_medium"])
        elif stat_group == "corners":
            high, medium = cw.get("corners_gap_high", cw["edge_high"]), cw.get("corners_gap_medium", cw["edge_medium"])
        elif stat_group == "sot":
            high, medium = cw.get("sot_gap_high", cw["edge_high"]), cw.get("sot_gap_medium", cw["edge_medium"])
        else:
            high, medium = cw["edge_high"], cw["edge_medium"]
        if edge >= high:
            conf = "high"
        elif edge >= medium:
            conf = "medium"
        else:
            conf = "low"
    # European competition penalty for non-top-5 teams (no domestic profile)
    if is_european and not has_domestic_profile:
        if conf == "high":
            conf = "medium"
        elif conf == "medium":
            conf = "low"
    return conf


def _euro_data_source_note(team: str, league: str) -> Optional[str]:
    """Return a data-source note for European competition evidence rendering."""
    if league not in EUROPEAN_COMPETITIONS:
        return None
    domestic_lg = resolve_domestic_league(team)
    if domestic_lg:
        euro_meta = get_team_profile_meta(team, league)
        domestic_meta = get_team_profile_meta(team, domestic_lg)
        d_n = int(_numeric(domestic_meta.get("matches_played")) or 0) if domestic_meta else 0
        e_n = int(_numeric(euro_meta.get("matches_played")) or 0) if euro_meta else 0
        ew = SCORING_WEIGHTS["european"]
        if e_n >= ew["min_euro_fixtures"]:
            return (f"Data: Domestic-anchored ({int(ew['domestic_weight']*100)}% {domestic_lg} + "
                    f"{int(ew['euro_weight']*100)}% {league}) — {d_n} domestic + {e_n} European fixtures")
        return f"Data: {domestic_lg} profile only ({d_n} fixtures) — insufficient {league} data ({e_n} fixtures)"
    euro_meta = get_team_profile_meta(team, league)
    e_n = int(_numeric(euro_meta.get("matches_played")) or 0) if euro_meta else 0
    return f"Data: European competition only ({e_n} {league} fixtures) — lower confidence"


def render_totals_line_answer(user_q: str, league: str, events: List[Dict]) -> str:
    # Delegate to goal intervals if user asks for intervals/ranges/bands
    q_lower = user_q.lower()
    if (re.search(r"\b(interval|range|band)s?\b", q_lower)
            and re.search(r"\b(goal|xg)s?\b", q_lower)):
        return render_goal_intervals(user_q, league, events)

    group = requested_stat_group(user_q)
    if group not in {"goals", "corners", "cards", "sot"}:
        return "I can estimate totals lines for goals/corners/cards/shots-on-target when the stat type is clear."

    title = {"goals": "Goals Totals", "corners": "Corner Totals", "cards": "Card Totals", "sot": "SoT Totals"}[group]
    lines: List[str] = [f"{title} advice by fixture:"]

    for ev in events:
        home = str(ev.get("home_team") or "Home")
        away = str(ev.get("away_team") or "Away")
        fixture = f"{home} vs {away}"

        # Knockout round context for European competitions
        ko_ctx = None
        if league in EUROPEAN_COMPETITIONS:
            try:
                ko_date = ev.get("commence_time", "")
                ko_ctx = get_knockout_context(home, away, league, ko_date)
            except Exception:
                ko_ctx = None

        # Stake intensity context for domestic leagues
        stake_ctx = None
        if league not in EUROPEAN_COMPETITIONS:
            try:
                stake_ctx = get_stake_context(home, away, league)
            except Exception:
                stake_ctx = None

        _cards_ref_mod = None
        if group == "goals":
            proj_total, season_total, recent_total = projected_total_goals(home, away, league, knockout_ctx=ko_ctx)
        elif group == "corners":
            proj_total, season_total, recent_total = projected_total_corners(home, away, league, knockout_ctx=ko_ctx)
        elif group == "sot":
            proj_total, season_total, recent_total = projected_total_sot(home, away, league, knockout_ctx=ko_ctx)
        else:
            proj_total, season_total, recent_total, _cards_ref_mod = projected_total_cards(home, away, league, knockout_ctx=ko_ctx)

        # Apply stake intensity modifier (domestic leagues only)
        if stake_ctx and stake_ctx.source != "none" and proj_total is not None:
            stat_mod = {"goals": stake_ctx.goals_modifier, "corners": stake_ctx.corners_modifier,
                        "cards": stake_ctx.cards_modifier, "sot": stake_ctx.sot_modifier}.get(group, 1.0)
            if abs(stat_mod - 1.0) >= 0.01:
                proj_total *= stat_mod

        # Lineup adjustment (when lineups are available, typically 30 min before kickoff)
        lineup_ctx = None
        if _HAS_LINEUP and proj_total is not None:
            try:
                fx_date = ev.get("commence_time", "")[:10] if ev.get("commence_time") else None
                lineup_ctx = get_lineup_context(home, away, league, fx_date)
                if lineup_ctx and lineup_ctx.is_available:
                    h_adj = lineup_ctx.home_adjustment.get(group, 1.0) if lineup_ctx.home_adjustment else 1.0
                    a_adj = lineup_ctx.away_adjustment.get(group, 1.0) if lineup_ctx.away_adjustment else 1.0
                    combined_adj = h_adj * a_adj
                    if abs(combined_adj - 1.0) >= 0.01:
                        proj_total *= combined_adj
            except Exception:
                lineup_ctx = None

        if proj_total is None:
            lines.append(f"- {fixture}: insufficient profile data to project totals.")
            continue

        # Market-implied baseline blend: anchor model projection toward market consensus
        _market_blend_applied = False
        _original_proj = proj_total
        market_total = extract_market_implied_total(ev, group)
        if market_total is not None:
            mbw = SCORING_WEIGHTS["market_blend"]
            proj_total = mbw["model_weight"] * proj_total + mbw["market_weight"] * market_total
            _market_blend_applied = True

        # Knockout context evidence
        if ko_ctx and ko_ctx.is_knockout:
            ko_line = knockout_evidence_line(ko_ctx)
            if ko_line:
                lines.append(f"  {ko_line}")

        # Stake intensity evidence (domestic leagues)
        if stake_ctx and stake_ctx.source != "none":
            stake_line = stake_evidence_line(stake_ctx)
            if stake_line:
                lines.append(f"  {stake_line}")

        # Lineup adjustment evidence
        if lineup_ctx and lineup_ctx.is_available:
            lu_line = lineup_evidence_line(lineup_ctx)
            if lu_line:
                lines.append(f"  {lu_line}")

        # European data source note
        if league in EUROPEAN_COMPETITIONS:
            for _t in (home, away):
                _note = _euro_data_source_note(_t, league)
                if _note:
                    lines.append(f"  {_t}: {_note}")

        # Compute combined variance for probability modeling (Bayesian blend)
        var_key = {"goals": "goals_var", "corners": "corners_for_var", "cards": "cards_var", "sot": "sot_for_var"}[group]
        h_v = get_blended_variance(home, league, var_key)
        a_v = get_blended_variance(away, league, var_key)
        combined_var = (h_v + a_v) if (h_v is not None and a_v is not None) else None

        options = extract_total_line_options(ev, group)
        best = choose_best_total_line(options, proj_total, combined_var)
        lines.append(f"- {fixture}: projected total {proj_total:.2f}.")

        if season_total is not None:
            lines.append(f"  Season model: {season_total:.2f}.")
        if recent_total is not None:
            lines.append(f"  Recent 6-match: {recent_total:.2f}.")
        if _market_blend_applied:
            lines.append(
                f"  Market-implied total: {market_total:.1f} "
                f"(blended 70/30 with model {_original_proj:.2f})"
            )

        # Referee profile for cards totals
        if group == "cards":
            if _cards_ref_mod and _cards_ref_mod.source == "profile":
                ref_name = (_cards_ref_mod.referee_name or "Unknown").title()
                ref_avg = _cards_ref_mod.avg_cards_per_match
                ref_strict = _cards_ref_mod.strictness_ratio
                ref_n = _cards_ref_mod.sample_size
                if ref_strict >= 1.10:
                    label = "strict"
                elif ref_strict <= 0.90:
                    label = "lenient"
                else:
                    label = "average"
                lines.append(
                    f"  Referee: {ref_name} — {ref_avg:.2f} cards/match ({label}, "
                    f"{ref_strict:.2f}x league avg, {ref_n} matches)."
                )
                ref_cpf = _cards_ref_mod.cards_per_foul
                if ref_cpf > 0:
                    lines.append(
                        f"  Ref cards/foul: {ref_cpf:.3f} — "
                        f"avg fouls/match: {_cards_ref_mod.avg_fouls_per_match:.1f}."
                    )
            else:
                lines.append("  Referee: not yet assigned or API-Football key not set.")

        if best:
            side = str(best.get("side") or "").title()
            point = float(best.get("point"))
            odds = float(best.get("odds"))
            edge = (proj_total - point) if side.lower() == "over" else (point - proj_total)
            model_p = best.get("_model_prob")
            implied_p = best.get("_implied_prob")
            val_edge = best.get("_value_edge")
            ev = best.get("_ev")
            conf = confidence_from_edge(edge, stat_group=group, model_prob=model_p, value_edge_pct=val_edge)
            lines.append(
                f"  Recommended: {side} {point:g} @ {odds:.2f} ({best.get('bookmaker')}, {best.get('market_key')})."
            )
            if model_p is not None and implied_p is not None and val_edge is not None:
                lines.append(
                    f"  Model: P({side} {point:g}) = {model_p:.1%} | Fair implied: {implied_p:.1%} | Value edge: {val_edge:+.1%}"
                )
            if ev is not None:
                lines.append(f"  Expected value: {ev:+.3f} per unit staked ({conf} confidence).")
            else:
                lines.append(f"  Model edge {edge:+.2f} ({conf} confidence).")
        else:
            side, anchor, mp = _best_model_only_line(proj_total, combined_var)
            lines.append(
                f"  Recommended: {side} {anchor:g} "
                f"(model-only, P({side} {anchor:g}) = {mp:.1%})."
            )

    lines.append("Method: Poisson probability model from local KB + available full-game market lines.")
    return "\n".join(lines)


# ---- Goal Interval Predictions ----

GOAL_INTERVAL_BANDS = [
    (0, 1, "0-1 goals"),
    (0, 2, "0-2 goals"),
    (1, 2, "1-2 goals"),
    (1, 3, "1-3 goals"),
    (2, 3, "2-3 goals"),
    (2, 4, "2-4 goals"),
    (2, 5, "2-5 goals"),
    (3, 4, "3-4 goals"),
    (3, 5, "3-5 goals"),
    (4, 5, "4-5 goals"),
    (6, 30, "6+ goals"),
]


def extract_goal_interval_odds(event: Dict) -> Dict[str, Dict]:
    """Extract goal range / goal band odds from bookmaker data.

    Returns {label: {"odds": float, "bookmaker": str, "market_key": str}} for
    each interval band found.  Empty dict when no goal-range market exists
    (graceful model-only fallback).
    """
    out: Dict[str, Dict] = {}
    for bm in event.get("bookmakers", []) or []:
        bm_name = str(bm.get("title") or "")
        for mk in bm.get("markets", []) or []:
            key = str(mk.get("key") or "")
            k = key.lower()
            if not is_full_game_market(key):
                continue
            # Match goal range / goal band market keys
            if not (("goal" in k and ("range" in k or "band" in k or "interval" in k))
                    or k in {"goals_range", "goal_range", "goal_band", "goal_interval"}):
                continue
            for outcome in mk.get("outcomes", []) or []:
                name = str(outcome.get("name") or "").strip()
                price = safe_float(outcome.get("price"))
                if price is None or price <= 1.0:
                    continue
                # Map outcome name to our standard bands
                label = _match_interval_label(name)
                if label and (label not in out or price > out[label]["odds"]):
                    out[label] = {"odds": price, "bookmaker": bm_name, "market_key": key}
    return out


def _match_interval_label(outcome_name: str) -> Optional[str]:
    """Map bookmaker outcome name like '0-1', '2-3', '4-5', '6+' to our band labels."""
    n = outcome_name.lower().strip()
    for low, high, label in GOAL_INTERVAL_BANDS:
        if high >= 30:
            # Open-ended: match "6+", "6 or more", "6-7", etc.
            if f"{low}+" in n or f"{low} or more" in n or f"{low}-" in n:
                return label
        else:
            if f"{low}-{high}" in n or f"{low} - {high}" in n:
                return label
    return None


def render_goal_intervals(_user_q: str, league: str, events: List[Dict]) -> str:
    """Goal interval probability breakdown per fixture."""
    lines: List[str] = ["Goal Interval Predictions by fixture:"]

    for ev in events:
        home = str(ev.get("home_team") or "Home")
        away = str(ev.get("away_team") or "Away")
        fixture = f"{home} vs {away}"

        proj_total, season_total, recent_total = projected_total_goals(home, away, league)
        if proj_total is None:
            lines.append(f"- {fixture}: insufficient profile data to project goal intervals.")
            continue

        # Market-implied baseline blend
        _market_blend_applied = False
        _original_proj = proj_total
        market_total = extract_market_implied_total(ev, "goals")
        if market_total is not None:
            mbw = SCORING_WEIGHTS["market_blend"]
            proj_total = mbw["model_weight"] * proj_total + mbw["market_weight"] * market_total
            _market_blend_applied = True

        # European data source note
        if league in EUROPEAN_COMPETITIONS:
            for _t in (home, away):
                _note = _euro_data_source_note(_t, league)
                if _note:
                    lines.append(f"  {_t}: {_note}")

        # Combined variance for NegBin
        h_v = get_blended_variance(home, league, "goals_var")
        a_v = get_blended_variance(away, league, "goals_var")
        combined_var = (h_v + a_v) if (h_v is not None and a_v is not None) else None

        lines.append(f"- {fixture}: projected total {proj_total:.2f}.")
        if season_total is not None:
            lines.append(f"  Season model: {season_total:.2f}.")
        if recent_total is not None:
            lines.append(f"  Recent 6-match: {recent_total:.2f}.")
        if _market_blend_applied:
            lines.append(
                f"  Market-implied total: {market_total:.1f} "
                f"(blended 70/30 with model {_original_proj:.2f})"
            )

        # Compute interval probabilities
        interval_odds = extract_goal_interval_odds(ev)
        lines.append("")
        lines.append(f"  Goal Intervals:")

        best_label = None
        best_prob = 0.0
        best_value_edge = None

        for low, high, label in GOAL_INTERVAL_BANDS:
            prob = interval_prob(proj_total, low, high, combined_var)

            odds_info = interval_odds.get(label)
            edge_str = ""
            v_edge = None
            if odds_info:
                odds = odds_info["odds"]
                imp_p = implied_prob(odds)
                v_edge = value_edge(prob, imp_p)
                edge_str = f"  @ {odds:.2f} ({odds_info['bookmaker']}) | Value edge: {v_edge:+.1%}"

            # Only recommend from bands with ≤3 goals (0-1 or 2-3)
            if high <= 3 and prob > best_prob:
                best_prob = prob
                best_label = label
                best_value_edge = v_edge

            lines.append(f"    {label:<14} {prob:>6.1%}{edge_str}")

        # Recommendation
        if best_label:
            conf = "high" if best_prob >= 0.45 else ("medium" if best_prob >= 0.30 else "low")
            rec = f"  Recommended: {best_label} ({best_prob:.1%} model probability, {conf} confidence)."
            if best_value_edge is not None:
                rec += f" Value edge: {best_value_edge:+.1%}."
            lines.append(rec)

        lines.append("")

    lines.append("Method: Poisson/NegBin interval probability model from local KB projections.")
    return "\n".join(lines)


def render_spreads_line_answer(user_q: str, league: str, events: List[Dict]) -> str:
    """Standalone handicap/spread recommendation per fixture."""
    lines: List[str] = ["Score Handicap advice by fixture:"]

    for ev in events:
        home = str(ev.get("home_team") or "Home")
        away = str(ev.get("away_team") or "Away")
        fixture = f"{home} vs {away}"

        proj_diff, season_diff, recent_diff = projected_goal_difference(home, away, league)
        if proj_diff is None:
            lines.append(f"- {fixture}: insufficient profile data to project goal difference.")
            continue

        favored = home if proj_diff >= 0 else away
        lines.append(f"- {fixture}: projected goal difference {proj_diff:+.2f} ({favored} favored).")
        if league in EUROPEAN_COMPETITIONS:
            for _t in (home, away):
                _note = _euro_data_source_note(_t, league)
                if _note:
                    lines.append(f"  {_t}: {_note}")
        if season_diff is not None:
            lines.append(f"  Reasoning: Season model diff {season_diff:+.2f}.")
        if recent_diff is not None:
            lines.append(f"  Reasoning: Recent 6-match xG diff {recent_diff:+.2f}.")

        options = extract_spread_line_options(ev)
        best = choose_best_spread_line(options, proj_diff, home)
        if best:
            team = best["team"]
            point = best["point"]
            odds = best["odds"]
            is_home = team.lower().strip() == home.lower().strip()
            sign = 1.0 if is_home else -1.0
            edge = sign * proj_diff + point
            model_p = best.get("_model_prob")
            implied_p = best.get("_implied_prob")
            val_edge = best.get("_value_edge")
            ev = best.get("_ev")
            conf = confidence_from_edge(abs(edge), stat_group="goals", model_prob=model_p, value_edge_pct=val_edge)
            lines.append(
                f"  Recommended: {team} {point:+g} @ {odds:.2f} ({best['bookmaker']}, {best['market_key']})."
            )
            if model_p is not None and implied_p is not None and val_edge is not None:
                lines.append(
                    f"  Model: P({team} covers {point:+g}) = {model_p:.1%} | Fair implied: {implied_p:.1%} | Value edge: {val_edge:+.1%}"
                )
            if ev is not None:
                lines.append(f"  Expected value: {ev:+.3f} per unit staked ({conf} confidence).")
            else:
                lines.append(f"  Model edge {edge:+.2f} ({conf} confidence).")
        else:
            sp_team, sp_hc, sp_cp = _best_model_only_spread(proj_diff, home, away)
            lines.append(
                f"  Recommended: {sp_team} {sp_hc} "
                f"(model-only, P(cover) = {sp_cp:.1%})."
            )

    lines.append("Method: normal probability model for goal-difference from local KB + available spread lines.")
    return "\n".join(lines)


def render_btts_answer(user_q: str, league: str, events: List[Dict]) -> str:
    """Standalone BTTS recommendation per fixture."""
    lines: List[str] = ["Both Teams To Score (BTTS) advice by fixture:"]

    for ev in events:
        home = str(ev.get("home_team") or "Home")
        away = str(ev.get("away_team") or "Away")
        fixture = f"{home} vs {away}"

        result = projected_btts_prob(home, away, league)
        p_btts, h_proj, a_proj, h_season, a_season = result

        if p_btts is None:
            lines.append(f"- {fixture}: insufficient profile data to project BTTS.")
            continue

        lines.append(f"- {fixture}: P(BTTS Yes) = {p_btts:.1%}.")
        if league in EUROPEAN_COMPETITIONS:
            for _t in (home, away):
                _note = _euro_data_source_note(_t, league)
                if _note:
                    lines.append(f"  {_t}: {_note}")
        if h_proj is not None and a_proj is not None:
            lines.append(f"  Blended projections: {home} {h_proj:.2f} goals, {away} {a_proj:.2f} goals.")
        if h_season is not None and a_season is not None:
            lines.append(f"  Season projections: {home} {h_season:.2f}, {away} {a_season:.2f}.")

        btts_options = extract_btts_odds(ev)
        best = choose_best_btts_side(btts_options, p_btts)

        if best:
            side = str(best.get("side") or "").title()
            odds = float(best.get("odds"))
            model_p = best.get("_model_prob")
            implied_p = best.get("_implied_prob")
            val_edge = best.get("_value_edge")
            ev_val = best.get("_ev")
            conf = confidence_from_edge(
                abs(val_edge) if val_edge else 0.0,
                stat_group="goals",
                model_prob=model_p,
                value_edge_pct=val_edge,
            )
            lines.append(
                f"  Recommended: BTTS {side} @ {odds:.2f} ({best.get('bookmaker')}, {best.get('market_key')})."
            )
            if model_p is not None and implied_p is not None and val_edge is not None:
                lines.append(
                    f"  Model: P(BTTS {side}) = {model_p:.1%} | Fair implied: {implied_p:.1%} | Value edge: {val_edge:+.1%}"
                )
            if ev_val is not None:
                lines.append(f"  Expected value: {ev_val:+.3f} per unit staked ({conf} confidence).")
            else:
                lines.append(f"  Confidence: {conf}.")
        else:
            # Model-only: recommend the side the model is more confident about
            rec = "Yes" if p_btts >= 0.50 else "No"
            rec_prob = p_btts if rec == "Yes" else (1.0 - p_btts)
            conf = "high" if rec_prob >= 0.65 else ("medium" if rec_prob >= 0.55 else "low")
            lines.append(
                f"  Recommended: BTTS {rec} — model gives {rec_prob:.0%} probability ({conf} confidence)."
            )
            lines.append(
                f"  (No BTTS odds available in current feed; model-only projection.)"
            )

    lines.append("Method: independent Poisson probability model for per-team scoring from local KB.")
    return "\n".join(lines)


def render_correct_score_answer(user_q: str, league: str, events: List[Dict]) -> str:
    """Standalone correct score prediction per fixture."""
    lines: List[str] = ["Correct Score Predictions by fixture:"]

    for ev in events:
        home = str(ev.get("home_team") or "Home")
        away = str(ev.get("away_team") or "Away")
        fixture = f"{home} vs {away}"

        probs, h_proj, a_proj = projected_correct_score_probs(home, away, league)
        if probs is None:
            lines.append(f"\n=== {fixture} ===")
            lines.append("  Insufficient profile data to project correct score.")
            continue

        lines.append(f"\n=== {fixture} ===")
        lines.append(f"  Goal projections: {home} {h_proj:.2f} — {away} {a_proj:.2f}")
        if league in EUROPEAN_COMPETITIONS:
            for _t in (home, away):
                _note = _euro_data_source_note(_t, league)
                if _note:
                    lines.append(f"  {_t}: {_note}")

        top_score = max(probs, key=probs.get)
        lines.append(f"  Most likely score: {top_score[0]}-{top_score[1]} ({probs[top_score]:.1%})")

        market_odds = extract_correct_score_odds(ev)
        ranked = choose_best_correct_scores(probs, market_odds, top_n=5)

        if market_odds:
            lines.append("  Top correct score bets (with odds):")
        else:
            lines.append("  Top scoreline probabilities (model-only — no correct score odds in feed):")

        for i, r in enumerate(ranked, 1):
            prob_str = f"{r['model_prob']:.1%}"
            if r["best_odds"] is not None:
                odds_str = f"@ {r['best_odds']:.2f}"
                bm_str = f" ({r['bookmaker']})" if r["bookmaker"] else ""
                edge_str = ""
                if r["value_edge"] is not None:
                    edge_str = f" | Edge: {r['value_edge']:+.1%}"
                ev_str = ""
                if r["ev"] is not None:
                    ev_str = f" | EV: {r['ev']:+.3f}"
                lines.append(f"    {i}. {r['score_str']}  —  {prob_str} {odds_str}{bm_str}{edge_str}{ev_str}")
            else:
                lines.append(f"    {i}. {r['score_str']}  —  {prob_str}")

        p_home_win = sum(p for (h, a), p in probs.items() if h > a)
        p_draw = sum(p for (h, a), p in probs.items() if h == a)
        p_away_win = sum(p for (h, a), p in probs.items() if h < a)
        lines.append(f"  Result probabilities: Home win {p_home_win:.1%} | Draw {p_draw:.1%} | Away win {p_away_win:.1%}")

    lines.append("\nMethod: independent Poisson/NegBin probability model per team from local KB.")
    return "\n".join(lines)


def render_moneyline_answer(user_q: str, league: str, events: List[Dict]) -> str:
    """Standalone moneyline recommendation per fixture."""
    lines: List[str] = ["Moneyline (1X2) advice by fixture:"]

    for ev in events:
        home = str(ev.get("home_team") or "Home")
        away = str(ev.get("away_team") or "Away")
        fixture = f"{home} vs {away}"

        p_home, p_draw, p_away, h_proj, a_proj = projected_moneyline_probs(home, away, league)
        if p_home is None:
            lines.append(f"\n=== {fixture} ===")
            lines.append("  Insufficient profile data to project moneyline.")
            continue

        lines.append(f"\n=== {fixture} ===")
        lines.append(f"  Goal projections: {home} {h_proj:.2f} — {away} {a_proj:.2f}")
        lines.append(f"  Model: P(Home) = {p_home:.1%}  P(Draw) = {p_draw:.1%}  P(Away) = {p_away:.1%}")
        if league in EUROPEAN_COMPETITIONS:
            for _t in (home, away):
                _note = _euro_data_source_note(_t, league)
                if _note:
                    lines.append(f"  {_t}: {_note}")

        # Quality signals for evidence
        hm = _profile_meta(home, league)
        am = _profile_meta(away, league)
        h_form = _numeric(hm.get("form_index_team"))
        a_form = _numeric(am.get("form_index_team"))
        h_dom = _numeric(hm.get("dominance_index"))
        a_dom = _numeric(am.get("dominance_index"))
        if h_form is not None and a_form is not None:
            lines.append(f"  Form index: {home} {h_form:.2f} vs {away} {a_form:.2f}")
        if h_dom is not None and a_dom is not None:
            lines.append(f"  Dominance index: {home} {h_dom:.2f} vs {away} {a_dom:.2f}")

        # Odds
        market_odds = extract_moneyline_odds(ev)
        result = choose_best_moneyline_side(p_home, p_draw, p_away, market_odds, home, away)
        rec = result["recommended"]

        if market_odds:
            lines.append(f"  Odds comparison:")
            for side_data in result["all_sides"]:
                if side_data["best_odds"] is not None:
                    ve_str = f" | Edge: {side_data['value_edge']:+.1%}" if side_data["value_edge"] is not None else ""
                    ev_str = f" | EV: {side_data['ev']:+.3f}" if side_data["ev"] is not None else ""
                    lines.append(
                        f"    {side_data['team']}: {side_data['model_prob']:.1%} model "
                        f"@ {side_data['best_odds']:.2f} ({side_data['bookmaker']}){ve_str}{ev_str}"
                    )

            conf = confidence_from_edge(
                abs(rec["value_edge"] or 0), stat_group="goals",
                model_prob=rec["model_prob"],
                value_edge_pct=rec["value_edge"],
            )
            lines.append(f"  >>> Recommended: {rec['team']} @ {rec['best_odds']:.2f} ({conf} confidence)")
        else:
            favored = max(
                [("home", p_home, home), ("draw", p_draw, "Draw"), ("away", p_away, away)],
                key=lambda x: x[1],
            )
            lines.append(f"  >>> Recommended: {favored[2]} ({favored[1]:.1%}) — no moneyline odds in feed (model-only)")

    lines.append("\nMethod: Poisson/NegBin scoreline matrix from local KB → 3-way outcome probabilities.")
    return "\n".join(lines)


def render_team_totals_answer(user_q: str, league: str, events: List[Dict]) -> str:
    """Per-team corner and card line recommendations for each fixture."""
    lines: List[str] = ["Per-Team Totals Lines by fixture:"]

    for ev in events:
        home = str(ev.get("home_team") or "Home")
        away = str(ev.get("away_team") or "Away")
        lines.append(f"\n=== {home} vs {away} ===")

        # Knockout round context for European competitions
        ko_ctx = None
        if league in EUROPEAN_COMPETITIONS:
            try:
                ko_date = ev.get("commence_time", "")
                ko_ctx = get_knockout_context(home, away, league, ko_date)
            except Exception:
                ko_ctx = None
            if ko_ctx and ko_ctx.is_knockout:
                ko_line = knockout_evidence_line(ko_ctx)
                if ko_line:
                    lines.append(f"  {ko_line}")
            for _t in (home, away):
                _note = _euro_data_source_note(_t, league)
                if _note:
                    lines.append(f"  {_t}: {_note}")

        # Stake intensity context for domestic leagues
        stake_ctx = None
        if league not in EUROPEAN_COMPETITIONS:
            try:
                stake_ctx = get_stake_context(home, away, league)
            except Exception:
                stake_ctx = None
            if stake_ctx and stake_ctx.source != "none":
                stake_line = stake_evidence_line(stake_ctx)
                if stake_line:
                    lines.append(f"  {stake_line}")

        # Lineup adjustment
        lineup_ctx = None
        if _HAS_LINEUP:
            try:
                fx_date = ev.get("commence_time", "")[:10] if ev.get("commence_time") else None
                lineup_ctx = get_lineup_context(home, away, league, fx_date)
                if lineup_ctx and lineup_ctx.is_available:
                    lu_line = lineup_evidence_line(lineup_ctx)
                    if lu_line:
                        lines.append(f"  {lu_line}")
            except Exception:
                lineup_ctx = None

        rw = SCORING_WEIGHTS["referee"]
        ref_mod = get_referee_modifier(home, away, league,
                                       weight=rw.get("modifier_weight", 0.25))

        # ---- CORNERS ----
        lines.append("  CORNERS:")
        _corner_projs = {}
        for team, opp, is_home in [(home, away, True), (away, home, False)]:
            c_bl, c_sea, c_rec = projected_team_corners(team, opp, league, is_home)
            # Apply knockout modifier
            if ko_ctx and ko_ctx.is_knockout and ko_ctx.corners_modifier != 1.0 and c_bl is not None:
                c_bl *= ko_ctx.corners_modifier
            # Apply lineup adjustment
            if lineup_ctx and lineup_ctx.is_available and c_bl is not None:
                adj = (lineup_ctx.home_adjustment or {}).get("corners", 1.0) if is_home \
                    else (lineup_ctx.away_adjustment or {}).get("corners", 1.0)
                if abs(adj - 1.0) >= 0.01:
                    c_bl *= adj
            if c_bl is None:
                lines.append(f"    {team}: insufficient data.")
                continue
            _corner_projs[team] = c_bl
            lines.append(f"    {team}: projected {c_bl:.2f} corners/match.")
            if c_sea is not None:
                lines.append(f"      Season model: {c_sea:.2f}.")
            if c_rec is not None:
                lines.append(f"      Recent/context anchor: {c_rec:.2f}.")
            t_var = get_blended_variance(team, league, "corners_for_var")
            options = extract_team_total_line_options(ev, team, "corners")
            best = choose_best_total_line(options, c_bl, t_var) if options else None
            if best:
                side = str(best.get("side") or "").title()
                pt = float(best.get("point"))
                odds = float(best.get("odds"))
                edge = (c_bl - pt) if side.lower() == "over" else (pt - c_bl)
                mp = best.get("_model_prob")
                ip = best.get("_implied_prob")
                ve = best.get("_value_edge")
                ev_val = best.get("_ev")
                conf = confidence_from_edge(edge, stat_group="corners",
                                           model_prob=mp, value_edge_pct=ve)
                lines.append(
                    f"      Recommended: {side} {pt:g} @ {odds:.2f} "
                    f"({best.get('bookmaker')}, {best.get('market_key')})."
                )
                if mp is not None and ip is not None and ve is not None:
                    lines.append(
                        f"      Model: P({side} {pt:g}) = {mp:.1%} | "
                        f"Fair implied: {ip:.1%} | Value edge: {ve:+.1%}"
                    )
                if ev_val is not None:
                    lines.append(f"      EV: {ev_val:+.3f} per unit ({conf} confidence).")
                else:
                    lines.append(f"      Edge {edge:+.2f} ({conf} confidence).")
            else:
                side, anchor, mp = _best_model_only_line(c_bl, t_var)
                conf = "high" if mp >= 0.65 else ("medium" if mp >= 0.55 else "low")
                lines.append(
                    f"      Recommended: {side} {anchor:g} "
                    f"(model-only, P({side} {anchor:g}) = {mp:.1%}, {conf} confidence)."
                )
                # Adjacent line probabilities
                for adj_offset in [-1.0, 1.0]:
                    adj_line = anchor + adj_offset
                    if adj_line >= 0.5:
                        adj_p_over = over_prob(c_bl, adj_line, t_var)
                        adj_side = "Over" if adj_p_over >= 0.5 else "Under"
                        adj_p = adj_p_over if adj_side == "Over" else (1.0 - adj_p_over)
                        lines.append(
                            f"        Alt: {adj_side} {adj_line:g} — P = {adj_p:.1%}."
                        )

        # Match total corners summary
        if len(_corner_projs) == 2:
            _c_vals = list(_corner_projs.values())
            lines.append(f"    Match total: {_c_vals[0] + _c_vals[1]:.1f} projected corners.")

        # ---- CARDS ----
        lines.append("  CARDS:")
        if ref_mod.source == "profile":
            rn = (ref_mod.referee_name or "Unknown").title()
            ra = ref_mod.avg_cards_per_match
            rs = ref_mod.strictness_ratio
            rns = ref_mod.sample_size
            label = "strict" if rs >= 1.10 else ("lenient" if rs <= 0.90 else "average")
            lines.append(
                f"    Referee: {rn} — {ra:.2f} cards/match ({label}, "
                f"{rs:.2f}x league avg, {rns} matches)."
            )
            ref_cpf = ref_mod.cards_per_foul
            if ref_cpf > 0:
                lines.append(
                    f"    Ref cards/foul: {ref_cpf:.3f} — "
                    f"avg fouls/match: {ref_mod.avg_fouls_per_match:.1f}."
                )
        else:
            lines.append("    Referee: not yet assigned or API-Football key not set.")

        _card_projs = {}
        for team, opp, is_home in [(home, away, True), (away, home, False)]:
            k_bl, k_sea, k_rec = projected_team_cards(team, opp, league, is_home,
                                                       ref_mod=ref_mod)
            # Apply knockout modifier
            if ko_ctx and ko_ctx.is_knockout and ko_ctx.cards_modifier != 1.0 and k_bl is not None:
                k_bl *= ko_ctx.cards_modifier
            # Apply lineup adjustment
            if lineup_ctx and lineup_ctx.is_available and k_bl is not None:
                adj = (lineup_ctx.home_adjustment or {}).get("cards", 1.0) if is_home \
                    else (lineup_ctx.away_adjustment or {}).get("cards", 1.0)
                if abs(adj - 1.0) >= 0.01:
                    k_bl *= adj
            if k_bl is None:
                lines.append(f"    {team}: insufficient data.")
                continue
            _card_projs[team] = k_bl
            lines.append(f"    {team}: projected {k_bl:.2f} cards/match.")
            if k_sea is not None:
                lines.append(f"      Season model: {k_sea:.2f}.")
            if k_rec is not None:
                lines.append(f"      Recent/context anchor: {k_rec:.2f}.")
            t_var = get_blended_variance(team, league, "cards_var")
            options = extract_team_total_line_options(ev, team, "cards")
            best = choose_best_total_line(options, k_bl, t_var) if options else None
            if best:
                side = str(best.get("side") or "").title()
                pt = float(best.get("point"))
                odds = float(best.get("odds"))
                edge = (k_bl - pt) if side.lower() == "over" else (pt - k_bl)
                mp = best.get("_model_prob")
                ip = best.get("_implied_prob")
                ve = best.get("_value_edge")
                ev_val = best.get("_ev")
                conf = confidence_from_edge(edge, stat_group="cards",
                                           model_prob=mp, value_edge_pct=ve)
                lines.append(
                    f"      Recommended: {side} {pt:g} @ {odds:.2f} "
                    f"({best.get('bookmaker')}, {best.get('market_key')})."
                )
                if mp is not None and ip is not None and ve is not None:
                    lines.append(
                        f"      Model: P({side} {pt:g}) = {mp:.1%} | "
                        f"Fair implied: {ip:.1%} | Value edge: {ve:+.1%}"
                    )
                if ev_val is not None:
                    lines.append(f"      EV: {ev_val:+.3f} per unit ({conf} confidence).")
                else:
                    lines.append(f"      Edge {edge:+.2f} ({conf} confidence).")
            else:
                side, anchor, mp = _best_model_only_line(k_bl, t_var)
                conf = "high" if mp >= 0.65 else ("medium" if mp >= 0.55 else "low")
                lines.append(
                    f"      Recommended: {side} {anchor:g} "
                    f"(model-only, P({side} {anchor:g}) = {mp:.1%}, {conf} confidence)."
                )
                # Adjacent line probabilities
                for adj_offset in [-1.0, 1.0]:
                    adj_line = anchor + adj_offset
                    if adj_line >= 0.5:
                        adj_p_over = over_prob(k_bl, adj_line, t_var)
                        adj_side = "Over" if adj_p_over >= 0.5 else "Under"
                        adj_p = adj_p_over if adj_side == "Over" else (1.0 - adj_p_over)
                        lines.append(
                            f"        Alt: {adj_side} {adj_line:g} — P = {adj_p:.1%}."
                        )

        # Match total cards summary
        if len(_card_projs) == 2:
            _k_vals = list(_card_projs.values())
            lines.append(f"    Match total: {_k_vals[0] + _k_vals[1]:.1f} projected cards.")

    lines.append("Method: per-team Poisson/NegBin probability model from local KB.")
    return "\n".join(lines)


def comparison_signals_cards(home: str, away: str, league: str) -> Tuple[Optional[str], List[str], Optional[float], Optional[float]]:
    hm = _profile_meta(home, league)
    am = _profile_meta(away, league)
    hr = _recent_stats(home, league, last_n=6)
    ar = _recent_stats(away, league, last_n=6)

    h_cards = safe_float(hm.get("cards_per_90_team"))
    a_cards = safe_float(am.get("cards_per_90_team"))
    h_fouls = safe_float(hm.get("fouls_per_90_team"))
    a_fouls = safe_float(am.get("fouls_per_90_team"))
    h_agg = safe_float(hm.get("aggression_index_norm"))
    a_agg = safe_float(am.get("aggression_index_norm"))
    h_ctrl = safe_float(hm.get("control_index"))
    a_ctrl = safe_float(am.get("control_index"))
    h_cpf = safe_float(hm.get("cards_per_foul_team"))
    a_cpf = safe_float(am.get("cards_per_foul_team"))
    h_opp_induced = safe_float(hm.get("opp_cards_induced_pm"))
    a_opp_induced = safe_float(am.get("opp_cards_induced_pm"))

    hr_cards = hr.get("cards_avg")
    ar_cards = ar.get("cards_avg")
    hr_fouls = hr.get("fouls_avg")
    ar_fouls = ar.get("fouls_avg")

    # Base deterministic projection from team profile.
    h_proj, a_proj = projected_cards(hm, am)
    if h_proj is None and h_cards is not None:
        h_proj = h_cards
    if a_proj is None and a_cards is not None:
        a_proj = a_cards

    # Matchup adjustment: opponent's card-inducing tendency shifts projections
    w_ma = SCORING_WEIGHTS["kb_quality"]["cards_matchup_adjust_w"]
    if h_opp_induced is not None and a_opp_induced is not None:
        league_avg_induced = (h_opp_induced + a_opp_induced) / 2.0
        if h_proj is not None:
            h_proj *= (1.0 + (a_opp_induced - league_avg_induced) * w_ma)
        if a_proj is not None:
            a_proj *= (1.0 + (h_opp_induced - league_avg_induced) * w_ma)

    # Referee strictness modifier
    ref_mod = get_referee_modifier(
        home, away, league,
        weight=SCORING_WEIGHTS["referee"].get("modifier_weight", 0.25),
    )
    if ref_mod.multiplier != 1.0:
        if h_proj is not None:
            h_proj *= ref_mod.multiplier
        if a_proj is not None:
            a_proj *= ref_mod.multiplier

    # Composite card risk to enrich reasoning (kept deterministic).
    cw = SCORING_WEIGHTS["comparison_cards"]

    def risk(cards, fouls, agg, recent_cards, recent_fouls, opp_control, opp_induced) -> Optional[float]:
        vals = [cards, fouls, agg, recent_cards, recent_fouls, opp_control, opp_induced]
        if all(v is None for v in vals):
            return None
        c = cards if cards is not None else 0.0
        f = fouls if fouls is not None else 0.0
        a = agg if agg is not None else 0.0
        rc = recent_cards if recent_cards is not None else 0.0
        rf = recent_fouls if recent_fouls is not None else 0.0
        oc = opp_control if opp_control is not None else 0.0
        oi = opp_induced if opp_induced is not None else 0.0
        return ((cw["cards"] * c) + (cw["fouls"] * f)
                + (cw["agg"] * a * cw["agg_scale"])
                + (cw["recent_cards"] * rc) + (cw["recent_fouls"] * rf)
                + (cw["opp_control"] * oc) + (cw.get("opp_induced", 0.20) * oi))

    h_risk = risk(h_cards, h_fouls, h_agg, hr_cards, hr_fouls, a_ctrl, a_opp_induced)
    a_risk = risk(a_cards, a_fouls, a_agg, ar_cards, ar_fouls, h_ctrl, h_opp_induced)

    pick = None
    h_score = h_risk if h_risk is not None else h_proj
    a_score = a_risk if a_risk is not None else a_proj
    if h_score is not None and a_score is not None:
        pick = home if h_score >= a_score else away
    elif h_score is not None:
        pick = home
    elif a_score is not None:
        pick = away

    lines: List[str] = []
    if h_proj is not None and a_proj is not None:
        lines.append(f"Profile projection (matchup-adj): {home} {h_proj:.2f} vs {away} {a_proj:.2f} cards.")
    if h_cards is not None and a_cards is not None:
        lines.append(f"Season cards/90: {home} {h_cards:.2f} vs {away} {a_cards:.2f}.")
    if h_fouls is not None and a_fouls is not None:
        lines.append(f"Season fouls/90: {home} {h_fouls:.2f} vs {away} {a_fouls:.2f}.")
    if h_cpf is not None and a_cpf is not None:
        lines.append(f"Cards per foul: {home} {h_cpf:.3f} vs {away} {a_cpf:.3f}.")
    if h_agg is not None and a_agg is not None:
        lines.append(f"Aggression index: {home} {h_agg:.2f} vs {away} {a_agg:.2f}.")
    if h_opp_induced is not None and a_opp_induced is not None:
        lines.append(f"Opp cards induced/match: {home} {h_opp_induced:.2f} vs {away} {a_opp_induced:.2f}.")
    if hr_cards is not None and ar_cards is not None:
        lines.append(f"Recent 6-match cards avg: {home} {hr_cards:.2f} vs {away} {ar_cards:.2f}.")
    if h_risk is not None and a_risk is not None:
        lines.append(f"Composite card risk: {home} {h_risk:.2f} vs {away} {a_risk:.2f}.")
    if ref_mod.source == "profile" and ref_mod.referee_name:
        rw_ref = SCORING_WEIGHTS["referee"]
        if abs(ref_mod.multiplier - 1.0) > rw_ref["evidence_threshold"]:
            direction_word = "strict" if ref_mod.multiplier > 1.0 else "lenient"
            lines.append(
                f"Referee: {ref_mod.referee_name.title()} ({direction_word}, "
                f"{ref_mod.strictness_ratio:.2f}x league avg, {ref_mod.sample_size} matches)."
            )
    return pick, lines[:8], h_score, a_score


def comparison_signals_corners(home: str, away: str, league: str) -> Tuple[Optional[str], List[str], Optional[float], Optional[float]]:
    hm = _profile_meta(home, league)
    am = _profile_meta(away, league)
    hr = _recent_stats(home, league, last_n=6)
    ar = _recent_stats(away, league, last_n=6)

    h_for = safe_float(hm.get("corners_pm"))
    a_for = safe_float(am.get("corners_pm"))
    h_against = safe_float(hm.get("corners_against_pm"))
    a_against = safe_float(am.get("corners_against_pm"))
    h_edge = safe_float(hm.get("corner_edge_pm"))
    a_edge = safe_float(am.get("corner_edge_pm"))
    h_ctrl = safe_float(hm.get("control_index"))
    a_ctrl = safe_float(am.get("control_index"))
    h_poss = safe_float(hm.get("possession"))
    a_poss = safe_float(am.get("possession"))
    h_dom = safe_float(hm.get("dominance_index"))
    a_dom = safe_float(am.get("dominance_index"))
    h_shots = safe_float(hm.get("shots_for_pm"))
    a_shots = safe_float(am.get("shots_for_pm"))
    h_home_corners = safe_float(hm.get("corners_home_pm"))
    a_away_corners = safe_float(am.get("corners_away_pm"))

    hr_for = hr.get("corners_for_avg")
    ar_for = ar.get("corners_for_avg")
    hr_against = hr.get("corners_against_avg")
    ar_against = ar.get("corners_against_avg")

    # projected_corners already uses venue-specific blend
    h_proj, a_proj = projected_corners(hm, am)

    ccw = SCORING_WEIGHTS["comparison_corners"]
    kbw = SCORING_WEIGHTS["kb_quality"]

    def corner_score(proj, season_for, opp_allow, recent_for, control, edge, shots) -> Optional[float]:
        vals = [proj, season_for, opp_allow, recent_for, control, edge]
        if all(v is None for v in vals):
            return None
        p = proj if proj is not None else 0.0
        sf = season_for if season_for is not None else 0.0
        oa = opp_allow if opp_allow is not None else 0.0
        rf = recent_for if recent_for is not None else 0.0
        c = control if control is not None else 0.0
        e = edge if edge is not None else 0.0
        s = ccw.get("shots", 0.08) * (shots if shots is not None else 0.0)
        return ccw["proj"] * p + ccw["season_for"] * sf + ccw["opp_allow"] * oa + ccw["recent_for"] * rf + ccw["control"] * c + ccw["edge"] * e + s

    h_score = corner_score(h_proj, h_for, a_against, hr_for, h_ctrl, h_edge, h_shots)
    a_score = corner_score(a_proj, a_for, h_against, ar_for, a_ctrl, a_edge, a_shots)

    # Style-clash bonus: large possession gap with high dominance gap → both scores rise
    style_clash_bonus = 0.0
    if h_poss is not None and a_poss is not None:
        poss_gap = abs(h_poss - a_poss)
        if h_dom is not None and a_dom is not None:
            dom_gap = abs(h_dom - a_dom)
            style_clash_bonus = poss_gap * dom_gap * kbw["corners_dom_asymmetry_w"]
        if h_shots is not None and a_shots is not None:
            dominant_shots = max(h_shots, a_shots)
            shot_excess = max(0.0, dominant_shots - 12.0)
            style_clash_bonus += poss_gap * shot_excess * kbw["corners_style_clash_w"]
    if h_score is not None:
        h_score += style_clash_bonus / 2.0
    if a_score is not None:
        a_score += style_clash_bonus / 2.0

    pick = None
    if h_score is not None and a_score is not None:
        pick = home if h_score >= a_score else away
    elif h_score is not None:
        pick = home
    elif a_score is not None:
        pick = away

    lines: List[str] = []
    if h_proj is not None and a_proj is not None:
        lines.append(f"Projected corners (venue-adj): {home} {h_proj:.2f} vs {away} {a_proj:.2f}.")
    if h_home_corners is not None and a_away_corners is not None:
        lines.append(f"Venue-specific: {home} home {h_home_corners:.2f} vs {away} away {a_away_corners:.2f}.")
    if h_for is not None and a_for is not None:
        lines.append(f"Season corners for/match: {home} {h_for:.2f} vs {away} {a_for:.2f}.")
    if h_against is not None and a_against is not None:
        lines.append(f"Season corners against/match: {home} {h_against:.2f} vs {away} {a_against:.2f}.")
    if h_shots is not None and a_shots is not None:
        lines.append(f"Shots for/match: {home} {h_shots:.1f} vs {away} {a_shots:.1f}.")
    if h_poss is not None and a_poss is not None:
        lines.append(f"Possession: {home} {h_poss:.1%} vs {away} {a_poss:.1%}.")
    if hr_for is not None and ar_for is not None:
        lines.append(f"Recent 6-match corners for: {home} {hr_for:.2f} vs {away} {ar_for:.2f}.")
    if h_edge is not None and a_edge is not None:
        lines.append(f"Corner edge: {home} {h_edge:+.2f} vs {away} {a_edge:+.2f}.")
    if style_clash_bonus > 0.05:
        lines.append(f"Style-clash bonus: +{style_clash_bonus:.2f} (possession/dominance gap amplifier).")
    if h_score is not None and a_score is not None:
        lines.append(f"Composite corner pressure score: {home} {h_score:.2f} vs {away} {a_score:.2f}.")
    return pick, lines[:8], h_score, a_score


def comparison_signals_sot(home: str, away: str, league: str) -> Tuple[Optional[str], List[str], Optional[float], Optional[float]]:
    hm = _profile_meta(home, league)
    am = _profile_meta(away, league)
    hr = _recent_stats(home, league, last_n=6)
    ar = _recent_stats(away, league, last_n=6)

    h_for = safe_float(hm.get("sot_for_pm"))
    a_for = safe_float(am.get("sot_for_pm"))
    h_against = safe_float(hm.get("sot_against_pm"))
    a_against = safe_float(am.get("sot_against_pm"))
    h_ratio = safe_float(hm.get("_sot_ratio_for"))
    a_ratio = safe_float(am.get("_sot_ratio_for"))

    hr_for = hr.get("sot_for_avg")
    ar_for = ar.get("sot_for_avg")

    h_proj, a_proj = projected_sot(hm, am)

    sw = SCORING_WEIGHTS["comparison_sot"]

    def sot_score(proj, season_for, opp_allow, recent_for, ratio) -> Optional[float]:
        vals = [proj, season_for, opp_allow, recent_for, ratio]
        if all(v is None for v in vals):
            return None
        p = proj if proj is not None else 0.0
        sf = season_for if season_for is not None else 0.0
        oa = opp_allow if opp_allow is not None else 0.0
        rf = recent_for if recent_for is not None else 0.0
        r = ratio if ratio is not None else 0.0
        return sw["proj"] * p + sw["season_for"] * sf + sw["opp_allow"] * oa + sw["recent_for"] * rf + sw["sot_ratio"] * r

    h_score = sot_score(h_proj, h_for, a_against, hr_for, h_ratio)
    a_score = sot_score(a_proj, a_for, h_against, ar_for, a_ratio)

    pick = None
    if h_score is not None and a_score is not None:
        pick = home if h_score >= a_score else away
    elif h_score is not None:
        pick = home
    elif a_score is not None:
        pick = away

    lines: List[str] = []
    if h_proj is not None and a_proj is not None:
        lines.append(f"Projected SoT: {home} {h_proj:.2f} vs {away} {a_proj:.2f}.")
    if h_for is not None and a_for is not None:
        lines.append(f"Season SoT for/match: {home} {h_for:.2f} vs {away} {a_for:.2f}.")
    if h_against is not None and a_against is not None:
        lines.append(f"Season SoT against/match: {home} {h_against:.2f} vs {away} {a_against:.2f}.")
    if hr_for is not None and ar_for is not None:
        lines.append(f"Recent 6-match SoT for: {home} {hr_for:.2f} vs {away} {ar_for:.2f}.")
    if h_ratio is not None and a_ratio is not None:
        lines.append(f"SoT accuracy ratio: {home} {h_ratio:.2f} vs {away} {a_ratio:.2f}.")
    if h_score is not None and a_score is not None:
        lines.append(f"Composite SoT pressure score: {home} {h_score:.2f} vs {away} {a_score:.2f}.")
    return pick, lines[:6], h_score, a_score


def render_comparison_answer(user_q: str, league: str, events: List[Dict]) -> str:
    group = requested_stat_group(user_q)
    if group not in {"corners", "cards", "sot"}:
        return (
            "I can compare corners/cards/shots-on-target for fixtures, but I couldn't detect which stat you want. "
            "Try: 'for all tomorrow EPL games, which team gets more corners?'"
        )

    detailed = wants_deep_explanation(user_q)
    lines: List[str] = []
    title = {"corners": "Corners", "cards": "Cards/Bookings", "sot": "Shots on Target"}[group]
    lines.append(f"{title} comparison by fixture:")

    for ev in events:
        home = str(ev.get("home_team") or "Home")
        away = str(ev.get("away_team") or "Away")

        if group == "corners":
            pick, reasons, h_score, a_score = comparison_signals_corners(home, away, league)
            stat_label = "corners"
            gap_high_key, gap_med_key = "corners_gap_high", "corners_gap_medium"
        elif group == "sot":
            pick, reasons, h_score, a_score = comparison_signals_sot(home, away, league)
            stat_label = "shots on target"
            gap_high_key, gap_med_key = "sot_gap_high", "sot_gap_medium"
        else:
            pick, reasons, h_score, a_score = comparison_signals_cards(home, away, league)
            stat_label = "cards"
            gap_high_key, gap_med_key = "cards_gap_high", "cards_gap_medium"

        if pick is None:
            lines.append(f"- {home} vs {away}: insufficient profile stats for {stat_label}.")
            continue
        lines.append(f"- {home} vs {away}: {pick} likely more {stat_label}.")
        if detailed:
            for r in reasons[:5]:
                lines.append(f"  Reasoning: {r}")
            if h_score is not None and a_score is not None:
                margin = abs(h_score - a_score)
                cfw = SCORING_WEIGHTS["confidence"]
                conf = "high" if margin >= cfw[gap_high_key] else ("medium" if margin >= cfw[gap_med_key] else "low")
                lines.append(f"  Reasoning: Confidence={conf} (score gap {margin:.2f}).")
        elif reasons:
            lines.append(f"  Evidence: {reasons[0]}")
            if len(reasons) > 1:
                lines.append(f"  Evidence: {reasons[1]}")

    method = "Method: deterministic multi-signal projection from local KB (season + recent fixtures), not bookmaker lines."
    lines.append(method if detailed else "Method: profile-based projection from your local KB (not bookmaker line prices).")
    return "\n".join(lines)


# ---- Intent Classification (scored classifier) ----
INTENT_SCHEDULE = "schedule"
INTENT_CAPABILITY = "capability"
INTENT_COMPARISON = "comparison"
INTENT_TOTALS_LINE = "totals_line"
INTENT_PLAYER_PROP = "player_prop"
INTENT_SPREADS_LINE = "spreads_line"
INTENT_BTTS = "btts"
INTENT_TEAM_TOTALS = "team_totals"
INTENT_CORRECT_SCORE = "correct_score"
INTENT_MONEYLINE = "moneyline_standalone"
INTENT_PARLAY = "parlay"
INTENT_UNKNOWN = "unknown"

INTENT_LABELS = {
    INTENT_SCHEDULE: "Schedule & Fixtures",
    INTENT_CAPABILITY: "Market Availability",
    INTENT_COMPARISON: "Stat Comparison",
    INTENT_TOTALS_LINE: "Totals Advice",
    INTENT_BTTS: "BTTS Advice",
    INTENT_TEAM_TOTALS: "Per-Team Totals Lines",
    INTENT_CORRECT_SCORE: "Correct Score Prediction",
    INTENT_SPREADS_LINE: "Handicap Line Advice",
    INTENT_MONEYLINE: "Moneyline Prediction",
    INTENT_PLAYER_PROP: "Player Props",
    INTENT_PARLAY: "Parlay Builder",
    INTENT_UNKNOWN: "General",
}


def classify_intent(user_q: str) -> List[Tuple[str, float]]:
    """Score all applicable intents, return sorted highest-first."""
    scores: List[Tuple[str, float]] = []

    if is_schedule_query(user_q):
        scores.append((INTENT_SCHEDULE, 1.0))
    if is_capability_query(user_q):
        scores.append((INTENT_CAPABILITY, 0.9))

    parlay = is_parlay_query(user_q)
    player_prop = is_player_prop_query(user_q)
    if player_prop:
        scores.append((INTENT_PLAYER_PROP, 0.85))
    if is_comparison_query(user_q):
        scores.append((INTENT_COMPARISON, 0.2 if (parlay or player_prop) else 0.7))
    if is_spreads_line_query(user_q):
        scores.append((INTENT_SPREADS_LINE, 0.15 if (parlay or player_prop) else 0.68))
    if is_btts_query(user_q):
        scores.append((INTENT_BTTS, 0.15 if (parlay or player_prop) else 0.70))
    if is_correct_score_query(user_q):
        scores.append((INTENT_CORRECT_SCORE, 0.15 if (parlay or player_prop) else 0.69))
    if is_moneyline_query(user_q):
        scores.append((INTENT_MONEYLINE, 0.15 if (parlay or player_prop) else 0.71))
    if is_team_totals_query(user_q):
        scores.append((INTENT_TEAM_TOTALS, 0.15 if (parlay or player_prop) else 0.72))
    if is_totals_line_query(user_q):
        scores.append((INTENT_TOTALS_LINE, 0.15 if (parlay or player_prop) else 0.65))
    if parlay and not player_prop:
        scores.append((INTENT_PARLAY, 0.6))

    if not scores:
        return [(INTENT_UNKNOWN, 0.0)]

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def _handle_schedule(user_q: str, events: List[Dict], notes: List[str]) -> str:
    return "Upcoming fixtures:\n" + event_fixtures_text(events)


def _handle_capability(user_q: str, events: List[Dict], notes: List[str]) -> str:
    lines: List[str] = []
    cands = build_candidates(events)
    groups = available_market_groups(cands)
    asked = requested_groups_from_query(user_q)
    if asked:
        missing = [g for g in sorted(asked) if g not in groups]
        have = [g for g in sorted(asked) if g in groups]
        if have:
            lines.append(f"Available for requested window: {', '.join(have)}")
        if missing:
            lines.append(f"Not available from current feed/window: {', '.join(missing)}")
    else:
        lines.append("Available market groups: " + (", ".join(sorted(groups)) if groups else "none"))
    if events:
        lines.append("Fixtures considered:")
        lines.append(event_fixtures_text(events))
    if notes:
        lines.append("Notes:")
        for n in notes:
            lines.append(f"- {n}")
    return "\n".join(lines)


def _handle_comparison(user_q: str, c: ConstraintSpec, events: List[Dict], notes: List[str]) -> str:
    out = render_comparison_answer(user_q, c.league, events)
    if notes:
        out = out + "\nNotes:\n" + "\n".join(f"- {n}" for n in notes)
    return out


def _handle_totals_line(user_q: str, c: ConstraintSpec, events: List[Dict], notes: List[str]) -> str:
    out = render_totals_line_answer(user_q, c.league, events)
    if notes:
        out = out + "\nNotes:\n" + "\n".join(f"- {n}" for n in notes)
    return out


def _handle_spreads_line(user_q: str, c: ConstraintSpec, events: List[Dict], notes: List[str]) -> str:
    out = render_spreads_line_answer(user_q, c.league, events)
    if notes:
        out = out + "\nNotes:\n" + "\n".join(f"- {n}" for n in notes)
    return out


def _handle_btts(user_q: str, c: ConstraintSpec, events: List[Dict], notes: List[str]) -> str:
    out = render_btts_answer(user_q, c.league, events)
    if notes:
        out = out + "\nNotes:\n" + "\n".join(f"- {n}" for n in notes)
    return out


def _handle_correct_score(user_q: str, c: ConstraintSpec, events: List[Dict], notes: List[str]) -> str:
    out = render_correct_score_answer(user_q, c.league, events)
    if notes:
        out = out + "\nNotes:\n" + "\n".join(f"- {n}" for n in notes)
    return out


def _handle_moneyline(user_q: str, c: ConstraintSpec, events: List[Dict], notes: List[str]) -> str:
    out = render_moneyline_answer(user_q, c.league, events)
    if notes:
        out = out + "\nNotes:\n" + "\n".join(f"- {n}" for n in notes)
    return out


def _handle_team_totals(user_q: str, c: ConstraintSpec, events: List[Dict], notes: List[str]) -> str:
    out = render_team_totals_answer(user_q, c.league, events)
    if notes:
        out = out + "\nNotes:\n" + "\n".join(f"- {n}" for n in notes)
    return out


def _handle_player_props(user_q: str, c: ConstraintSpec, events: List[Dict], notes: List[str]) -> str:
    if not events:
        return "No fixtures found for the requested date. Cannot analyse player props without a fixture list."

    # Detect which stats the user is asking about
    requested_stats = _parse_player_prop_stats(user_q)

    # Use projection-backed handler when available
    if _HAS_PLAYER_PROJ:
        return _handle_player_props_projected(user_q, c, events, notes, requested_stats)

    # Fallback to LLM-only handler
    return _handle_player_props_llm(user_q, c, events, notes)


def _parse_player_prop_stats(user_q: str) -> List[str]:
    """Detect which player stats the user is asking about."""
    q = user_q.lower()
    stats = []
    if re.search(r"\b(goal|scor|anytime|first\s*scorer)\b", q):
        stats.append("goals")
    if re.search(r"\b(shot|sot|shots?\s+on\s+target)\b", q):
        stats.append("sot")
    if re.search(r"\b(assists?|chance|creat)", q):
        stats.append("assists")
    if re.search(r"\b(cards?|booked?|booking|yellow|red|caution)\b", q):
        stats.append("cards")
    if re.search(r"\b(fouls?|foul\s+committed|to\s+foul|player\s+foul)", q):
        stats.append("fouls")
    if not stats:
        stats = ["goals"]  # default
    return stats


def _get_recommendable_players(
    team: str, league: str, limit: int = 8,
) -> List[Dict]:
    """Retrieve player profiles filtered to recommendable players (not bench).
    Sorted by start_rate descending.

    For European competitions, falls back to the team's domestic league
    since European comp player profiles are often sparse or missing.
    """
    col = get_collection_handle(create_if_missing=True)
    variants = _kb_team_variants(team)

    # For European comps, try domestic league first (more data)
    search_leagues = [league]
    if league in EUROPEAN_COMPETITIONS:
        domestic = resolve_domestic_league(team)
        if domestic:
            search_leagues = [domestic, league]

    # Determine current season
    prof = get_team_profile_doc(team, league)
    current_season = (prof.get("meta") or {}).get("season")

    profiles = []
    seen_ids = set()
    for search_lg in search_leagues:
        # Try with season filter first, then without (handles season format mismatches)
        for use_season in ([current_season, None] if current_season else [None]):
            for tv in variants:
                filters: List[Dict] = [{"doc_type": "player_profile"}, {"team": tv}]
                if use_season:
                    filters.append({"season": use_season})
                where = build_where(search_lg, extra_filters=filters)
                res = col.get(where=where, include=["documents", "metadatas"], limit=50)
                for i, d, m in zip(res.get("ids", []), res.get("documents", []), res.get("metadatas", [])):
                    if i in seen_ids:
                        continue
                    seen_ids.add(i)
                    meta = m or {}
                    role = meta.get("minutes_risk", "")
                    if role == "bench":
                        continue
                    profiles.append({"id": i, "text": d or "", "meta": meta})
            if profiles:
                break  # found players, stop trying season variants
        if profiles:
            break  # found players in this league

    profiles.sort(
        key=lambda d: float(d.get("meta", {}).get("start_rate") or 0),
        reverse=True,
    )
    return profiles[:limit]


def _parse_player_stats_from_text(text: str) -> Dict:
    """Extract stat fields from player fixture document text.

    Chroma drops integer 0 from metadata, so we parse stats from the
    text which has format: 'G:1 A:0 Shots:3 (On:2) ... YC:1 RC:0'
    """
    stats: Dict = {}
    if not text:
        return stats
    # Goals
    m = re.search(r"\bG:(\d+)", text)
    if m:
        stats["goals"] = int(m.group(1))
    # Assists
    m = re.search(r"\bA:(\d+)", text)
    if m:
        stats["assists"] = int(m.group(1))
    # Shots on target
    m = re.search(r"\(On:(\d+)\)", text)
    if m:
        stats["shots_on"] = int(m.group(1))
    # Yellow + Red cards -> cards_total
    yc = rc = 0
    m = re.search(r"\bYC:(\d+)", text)
    if m:
        yc = int(m.group(1))
    m = re.search(r"\bRC:(\d+)", text)
    if m:
        rc = int(m.group(1))
    stats["cards_total"] = yc + rc
    return stats


def _get_player_recent_fixtures(
    player_id, team: str, league: str, limit: int = 8, min_minutes: int = 30,
) -> List[Dict]:
    """Retrieve recent fixture docs for a specific player.

    Parses stat fields from document text (Chroma drops 0-valued metadata).
    """
    col = get_collection_handle(create_if_missing=True)
    variants = _kb_team_variants(team)

    prof = get_team_profile_doc(team, league)
    current_season = (prof.get("meta") or {}).get("season")

    fixtures = []
    # Query by player_id directly — much more efficient than filtering in Python
    filters: List[Dict] = [
        {"doc_type": "player_fixture"},
        {"player_id": int(player_id) if str(player_id).isdigit() else player_id},
        {"minutes": {"$gte": min_minutes}},
    ]
    if current_season:
        filters.append({"season": current_season})
    where = build_where(league, extra_filters=filters)
    res = col.get(where=where, include=["documents", "metadatas"], limit=limit)
    for i, d, m in zip(res.get("ids", []), res.get("documents", []), res.get("metadatas", [])):
        meta = m or {}
        # Parse stats from text since Chroma drops integer 0 metadata
        text_stats = _parse_player_stats_from_text(d or "")
        enriched_meta = {**meta, **text_stats}
        fixtures.append({"id": i, "text": d or "", "meta": enriched_meta})

    # Fallback: retry without season filter if no results (season format mismatch)
    if not fixtures and current_season:
        filters_no_season = [f for f in filters if "season" not in f]
        where_ns = build_where(league, extra_filters=filters_no_season)
        res = col.get(where=where_ns, include=["documents", "metadatas"], limit=limit)
        for i, d, m in zip(res.get("ids", []), res.get("documents", []), res.get("metadatas", [])):
            meta = m or {}
            text_stats = _parse_player_stats_from_text(d or "")
            enriched_meta = {**meta, **text_stats}
            fixtures.append({"id": i, "text": d or "", "meta": enriched_meta})

    fixtures.sort(key=lambda d: d.get("meta", {}).get("minutes") or 0, reverse=True)
    return fixtures[:limit]


def _handle_player_props_projected(
    user_q: str, c: ConstraintSpec, events: List[Dict],
    notes: List[str], requested_stats: List[str],
) -> str:
    """Projection-backed player prop handler."""
    league = c.league
    all_recommendations: List[PlayerPropRecommendation] = []

    # Pre-check if cards are requested (referee lookup needed)
    needs_referee = "cards" in requested_stats

    for ev in events:
        home_team = ev.get("home_team", "")
        away_team = ev.get("away_team", "")
        if not home_team or not away_team:
            continue

        fixture_label = f"{home_team} vs {away_team}"
        home_team_meta = get_team_profile_meta(home_team, league) or {}
        away_team_meta = get_team_profile_meta(away_team, league) or {}

        # Fetch referee modifier once per fixture (for card projections)
        ref_mod = None
        if needs_referee and _REFEREE_AVAILABLE:
            try:
                rw = SCORING_WEIGHTS.get("referee", {})
                ref_mod = get_referee_modifier(
                    home_team, away_team, league,
                    weight=rw.get("modifier_weight", 0.25),
                )
            except Exception:
                ref_mod = None

        for team, opp_meta, is_home in [
            (home_team, away_team_meta, True),
            (away_team, home_team_meta, False),
        ]:
            team_meta = home_team_meta if is_home else away_team_meta
            players = _get_recommendable_players(team, league, limit=8)

            for player_doc in players:
                player_meta = player_doc.get("meta", {})
                player_id = player_meta.get("player_id")
                player_text = player_doc.get("text", "")
                if not player_id:
                    continue

                recent = _get_player_recent_fixtures(
                    player_id, team, league, limit=8, min_minutes=30,
                )

                for stat in requested_stats:
                    proj = projected_player_stat(
                        player_meta=player_meta,
                        team_meta=team_meta,
                        opponent_meta=opp_meta,
                        stat=stat,
                        league=league,
                        is_home=is_home,
                        recent_fixtures=recent,
                        fixture_label=fixture_label,
                        player_text=player_text,
                        ref_mod=ref_mod if stat == "cards" else None,
                    )
                    if proj is None:
                        continue

                    rec = recommend_player_prop(proj, event=ev)
                    if rec is not None:
                        all_recommendations.append(rec)

    # Sort: highest projection value first (best scorers/assisters on top),
    # then by confidence, then by P(Over 0.5)
    all_recommendations.sort(key=lambda r: (
        r.projection.projection,  # actual expected value — best scorers first
        {"high": 3, "medium": 2, "low": 1}.get(r.projection.confidence, 0),
        r.projection.prob_over_0_5 or 0,
    ), reverse=True)

    # Limit to top 5 per fixture to keep output focused
    fixture_counts: Dict[str, int] = {}
    filtered_recs: List[PlayerPropRecommendation] = []
    for rec in all_recommendations:
        fx = rec.projection.fixture
        cnt = fixture_counts.get(fx, 0)
        if cnt < 5:
            filtered_recs.append(rec)
            fixture_counts[fx] = cnt + 1

    out = render_player_prop_answer(league, filtered_recs)

    if notes:
        out += "\n\nNotes:\n" + "\n".join(f"- {n}" for n in notes)

    return out


def _handle_player_props_llm(user_q: str, c: ConstraintSpec, events: List[Dict], notes: List[str]) -> str:
    """LLM-only fallback for player props (used when player_projections module unavailable)."""
    if client is None:
        return "OpenAI client not configured — player prop analysis requires LLM access."

    fixtures_text = event_fixtures_text(events)
    player_docs = get_player_snippets_for_events(
        events, c.league, per_team_profiles=6, per_team_fixtures=8,
    )
    player_text = "\n\n".join((d.get("text") or "") for d in player_docs[:30])

    profile_docs = get_team_profile_snippets_for_events(events, c.league, per_team=1)
    profile_text = "\n\n".join((d.get("text") or "") for d in profile_docs[:10])

    system_msg = (
        "You are a football betting analyst specialising in player props.\n"
        "The user will ask about goalscorers, shots on target, assists, cards, or other player-level markets.\n"
        "You are provided with:\n"
        "- The fixture list for the requested date.\n"
        "- Player profile docs (season averages: goals/90, assists/90, shots/90, SoT/90, fouls/90, cards/90, form index).\n"
        "- Player fixture docs (recent per-match stats: goals, assists, shots, SoT, cards, minutes, rating).\n"
        "- Team profile docs (team-level context: form, control, dominance, aggression).\n\n"
        "Guidelines:\n"
        "- Base your picks on the statistical data provided. Cite specific numbers.\n"
        "- IMPORTANT: Only recommend players who are regular starters (look at minutes played and appearances).\n"
        "- Do NOT recommend players who mostly come off the bench or have very few appearances.\n"
        "- For goalscorer picks: prioritise goals/90, shots/90, SoT/90, minutes played, and recent form index.\n"
        "- For SoT picks: use shots/90, SoT/90, and recent fixture shot counts.\n"
        "- For card picks: use cards/90, fouls/90, aggression index, and recent yellow card counts.\n"
        "- For assists: use assists/90, passes/90, and chance creation signals.\n"
        "- If data is sparse for a player, say so rather than guessing.\n"
        "- When recommending multiple picks, rank by confidence and explain the reasoning.\n"
        "- Format your answer clearly with fixture headers and bullet points.\n"
    )
    user_msg = (
        f"User question: {user_q}\n\n"
        f"Fixtures on this date:\n{fixtures_text}\n\n"
        f"Team profile context:\n{profile_text or 'None'}\n\n"
        f"Player data:\n{player_text or 'No player data available in KB.'}\n"
    )
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.3,
        )
        out = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        out = f"LLM call failed for player props: {exc}"

    if notes:
        out = out + "\n\nNotes:\n" + "\n".join(f"- {n}" for n in notes)
    out += "\nMethod: LLM analysis backed by player profile and fixture data from local KB."
    return out


def answer_once(user_q: str, default_league: str = "EPL") -> None:
    raw_user_q = user_q
    followup_mode = is_followup_request(user_q)
    c = parse_constraints(user_q, default_league=default_league)
    parlay_intent = is_parlay_query(user_q)
    force_totals_only = wants_totals_only_parlay(user_q)
    totals_stat_group = requested_stat_group(user_q) or "goals"
    if force_totals_only:
        c.requested_markets = {"totals"}
        # Reset include/exclude to avoid contradictions with generic only_* detectors
        # from parse_constraints(), then set clean group-specific constraints.
        if totals_stat_group == "goals":
            c.hard_include_groups = {"totals"}
            c.hard_exclude_groups = {"moneyline", "spreads", "corners", "cards", "sot"}
        elif totals_stat_group == "corners":
            # Corner totals market keys (e.g. "alternate_totals_corners") are classified
            # as "corners" group, NOT "totals". So hard_include must be "corners" only.
            c.hard_include_groups = {"corners"}
            c.hard_exclude_groups = {"moneyline", "spreads", "cards", "totals", "sot"}
            c.require_total_corner_keys = True
        elif totals_stat_group == "cards":
            # Card totals market keys are classified as "cards" group.
            c.hard_include_groups = {"cards"}
            c.hard_exclude_groups = {"moneyline", "spreads", "corners", "totals", "sot"}
        elif totals_stat_group == "sot":
            c.hard_include_groups = {"sot"}
            c.hard_exclude_groups = {"moneyline", "spreads", "corners", "cards", "totals"}
    # Parse target date BEFORE fetching events so we query the right day.
    explicit_date = parse_explicit_date(user_q)
    # Also check time_window for day-name keywords (today, tomorrow, saturday, etc.)
    target_fetch_date = explicit_date
    if target_fetch_date is None and c.time_window not in ("upcoming", ""):
        from datetime import timedelta as _td
        _today = date.today()
        if c.time_window == "today":
            target_fetch_date = _today
        elif c.time_window == "tomorrow":
            target_fetch_date = _today + _td(days=1)

    if c.leagues:
        events_raw, notes = fetch_events_multi(c.leagues, c.requested_markets)
    else:
        events_raw, notes = fetch_events(c.league, c.requested_markets,
                                          target_date=target_fetch_date)
        for ev in events_raw:
            ev["_league"] = c.league

    # If no events for the target date, try a 7-day lookahead
    if not events_raw and target_fetch_date is None:
        from datetime import timedelta as _td
        _today = date.today()
        all_lookahead: List[Dict] = []
        all_lookahead_notes: List[str] = []
        for offset in range(1, 8):
            d = _today + _td(days=offset)
            if c.leagues:
                day_events: List[Dict] = []
                for lg in c.leagues:
                    lg_events, lg_notes = fetch_events(lg, c.requested_markets, target_date=d)
                    for ev in lg_events:
                        ev["_league"] = lg
                    day_events.extend(lg_events)
                    all_lookahead_notes.extend(lg_notes)
                all_lookahead.extend(day_events)
            else:
                day_events, day_notes = fetch_events(c.league, c.requested_markets,
                                                      target_date=d)
                for ev in day_events:
                    ev["_league"] = c.league
                all_lookahead.extend(day_events)
                all_lookahead_notes.extend(day_notes)
        if all_lookahead:
            events_raw = all_lookahead
            notes = all_lookahead_notes

    if not events_raw:
        print("Could not fetch odds events.")
        for n in notes:
            print("-", n)
        return
    if explicit_date is not None:
        events = filter_events_by_exact_date(events_raw, explicit_date)
        notes.append(f"Applied explicit date filter: {explicit_date.isoformat()}.")
    else:
        events = filter_events_by_window(events_raw, c.time_window)

    events = filter_events_by_mentioned_teams(user_q, events)
    mentioned_teams = extract_mentioned_event_teams(user_q, events_raw)
    if is_singular_fixture_request(user_q) and len(mentioned_teams) == 1 and events:
        team = next(iter(mentioned_teams))
        team_events = [ev for ev in events if (ev.get("home_team") == team) or (ev.get("away_team") == team)]
        if team_events:
            team_events.sort(key=event_sort_key)
            events = [team_events[0]]
            notes.append(
                f"Applied singular fixture lock: {events[0].get('home_team')} vs {events[0].get('away_team')}."
            )

    has_explicit_team_mention = query_mentions_any_event_team(user_q, events_raw)
    if followup_mode and memory.get("last_selected_event_ids") and not has_explicit_team_mention:
        keep = {str(x) for x in (memory.get("last_selected_event_ids") or [])}
        filtered = [ev for ev in events if str(ev.get("id") or "") in keep]
        if filtered:
            events = filtered
            notes.append("Applied follow-up memory: kept fixtures from previous recommendation.")

    # Event-level enrichment for groups the user explicitly asked about.
    asked_groups = requested_groups_from_query(user_q)
    needed_groups = set(c.hard_include_groups) | set(c.required_group_counts.keys()) | set(asked_groups)

    # For parlay flows, widen candidates with corners/cards/sot only when
    # those groups are not explicitly excluded by constraints.
    if parlay_intent and not force_totals_only:
        parlay_groups = {"corners", "cards", "sot", "btts"}
        if c.hard_include_groups:
            parlay_groups &= c.hard_include_groups
        parlay_groups -= c.hard_exclude_groups
        if parlay_groups:
            needed_groups.update(parlay_groups)
            notes.append(
                "Enriching all parlay events with "
                + "/".join(sorted(parlay_groups))
                + " candidates."
            )

    # Team totals intent always needs corners + cards enrichment
    if is_team_totals_query(user_q):
        needed_groups.update({"corners", "cards"})

    if needed_groups:
        if c.leagues:
            events_by_lg: Dict[str, List[Dict]] = {}
            for ev in events:
                events_by_lg.setdefault(ev.get("_league", c.league), []).append(ev)
            enriched_all: List[Dict] = []
            for lg, lg_evs in events_by_lg.items():
                enriched, enotes = enrich_events_for_groups(lg_evs, lg, needed_groups)
                enriched_all.extend(enriched)
                notes.extend(enotes)
            events = enriched_all
        else:
            events, enrich_notes = enrich_events_for_groups(events, c.league, needed_groups)
            notes.extend(enrich_notes)

    if parlay_intent and force_totals_only:
        if c.leagues:
            events_by_lg2: Dict[str, List[Dict]] = {}
            for ev in events:
                events_by_lg2.setdefault(ev.get("_league", c.league), []).append(ev)
            depth_all: List[Dict] = []
            for lg, lg_evs in events_by_lg2.items():
                depth_evs, depth_notes = enrich_events_for_totals_depth(lg_evs, lg, stat_group=totals_stat_group)
                depth_all.extend(depth_evs)
                notes.extend(depth_notes)
            events = depth_all
        else:
            events, depth_notes = enrich_events_for_totals_depth(events, c.league, stat_group=totals_stat_group)
        notes.extend(depth_notes)
        notes.append(f"Expanded full-game totals ladder for parlay optimization ({totals_stat_group}).")
    # ---- Intent-based routing via scored classifier ----
    intent_scores = classify_intent(user_q)
    top_intent = intent_scores[0][0]

    if top_intent == INTENT_SCHEDULE:
        out = _handle_schedule(user_q, events, notes)
        print(out)
        add_chat_turn(raw_user_q, out)
        return

    if top_intent == INTENT_CAPABILITY:
        out = _handle_capability(user_q, events, notes)
        print(out)
        add_chat_turn(raw_user_q, out)
        return

    if top_intent == INTENT_COMPARISON:
        out = _handle_comparison(user_q, c, events, notes)
        print(out)
        add_chat_turn(raw_user_q, out)
        return

    if top_intent == INTENT_MONEYLINE:
        out = _handle_moneyline(user_q, c, events, notes)
        print(out)
        add_chat_turn(raw_user_q, out)
        if _HAS_TRACKER:
            _log_predictions(out, c.league, events)
        memory["last_intent"] = INTENT_MONEYLINE
        return

    if top_intent == INTENT_SPREADS_LINE:
        out = _handle_spreads_line(user_q, c, events, notes)
        print(out)
        add_chat_turn(raw_user_q, out)
        if _HAS_TRACKER:
            _log_predictions(out, c.league, events)
        return

    if top_intent == INTENT_TOTALS_LINE:
        out = _handle_totals_line(user_q, c, events, notes)
        print(out)
        add_chat_turn(raw_user_q, out)
        if _HAS_TRACKER:
            _log_predictions(out, c.league, events)
        return

    if top_intent == INTENT_BTTS:
        out = _handle_btts(user_q, c, events, notes)
        print(out)
        add_chat_turn(raw_user_q, out)
        if _HAS_TRACKER:
            _log_predictions(out, c.league, events)
        return

    if top_intent == INTENT_CORRECT_SCORE:
        out = _handle_correct_score(user_q, c, events, notes)
        print(out)
        add_chat_turn(raw_user_q, out)
        if _HAS_TRACKER:
            _log_predictions(out, c.league, events)
        memory["last_intent"] = INTENT_CORRECT_SCORE
        return

    if top_intent == INTENT_TEAM_TOTALS:
        out = _handle_team_totals(user_q, c, events, notes)
        print(out)
        add_chat_turn(raw_user_q, out)
        if _HAS_TRACKER:
            _log_predictions(out, c.league, events)
        return

    if top_intent == INTENT_PLAYER_PROP:
        out = _handle_player_props(user_q, c, events, notes)
        print(out)
        add_chat_turn(raw_user_q, out)
        memory["last_intent"] = INTENT_PLAYER_PROP
        return

    if top_intent == INTENT_UNKNOWN:
        out = "I can help with fixture questions and parlays.\nTry: 'for all tomorrow EPL games, which team gets more corners?'"
        print(out)
        add_chat_turn(raw_user_q, out)
        return

    if not events:
        out = "No fixtures found for requested window."
        print(out)
        add_chat_turn(raw_user_q, out)
        return

    candidates = build_candidates(events)
    if not candidates:
        out = "No odds candidates found in provider response."
        print(out)
        add_chat_turn(raw_user_q, out)
        return

    leg_count = c.leg_count if c.leg_count is not None else 3
    if c.per_match_mode:
        leg_count = len(events)

    # Richer follow-up: carry over constraints from memory when applicable
    if followup_mode and memory.get("last_constraint_spec"):
        prev = memory["last_constraint_spec"]
        q_lower = user_q.lower()
        # "same markets/types" → inherit market group constraints
        if re.search(r"\b(same|keep|previous)\b.*\b(market|type|kind)\b", q_lower):
            if prev.get("hard_include_groups") and not c.hard_include_groups:
                c.hard_include_groups = set(prev["hard_include_groups"])
                notes.append("Carried over market group constraints from previous recommendation.")
            if prev.get("hard_exclude_groups") and not c.hard_exclude_groups:
                c.hard_exclude_groups = set(prev["hard_exclude_groups"])
            if prev.get("soft_prefer_groups") and not c.soft_prefer_groups:
                c.soft_prefer_groups = set(prev["soft_prefer_groups"])
        # "safer" / "cap odds" → inherit unique-events and per-match from memory
        if prev.get("per_match_mode") and c.per_match_mode is False:
            if re.search(r"\b(same|keep)\b.*\b(fixture|match|game)\b", q_lower):
                c.per_match_mode = True
                c.require_unique_events = True
        if c.leg_count is None and memory.get("last_leg_count"):
            leg_count = int(memory.get("last_leg_count") or leg_count)
        if re.search(r"\badd\b|\banother\b|\bone more\b", q_lower):
            leg_count = min(leg_count + 1, max(1, len(candidates)))

    # Keep realistic count under available unique-event capacity when required.
    if c.require_unique_events:
        unique_events = len({x.event_id for x in candidates})
        if unique_events > 0:
            leg_count = min(leg_count, unique_events)

    # Feasibility pre-check before expensive solver
    feasible, feasibility_warnings = check_feasibility(candidates, leg_count, c)
    notes.extend(feasibility_warnings)
    if not feasible:
        print("Could not construct a parlay with current constraints.")
        for n in notes:
            print("-", n)
        add_chat_turn(raw_user_q, "Could not construct a parlay with current constraints.\n" + "\n".join(f"- {n}" for n in notes))
        return
    if c.require_unique_events:
        unique_events = len({x.event_id for x in candidates})
        if unique_events < leg_count and unique_events > 0:
            leg_count = unique_events

    selected, sel_notes = select_parlay(candidates, leg_count, c)
    notes.extend(sel_notes)
    if force_totals_only:
        notes.append("Applied totals-only parlay routing from user query.")
    if not selected:
        print("Could not construct a parlay with current constraints.")
        for n in notes:
            print("-", n)
        return

    # Optional LLM validation pass (gated by RAG_LLM_VALIDATION env var)
    league_label = ", ".join(c.leagues) if c.leagues else c.league
    validation_passed, validation_concern = llm_validate_selection(selected, league_label)
    if not validation_passed and validation_concern:
        notes.append(f"LLM validation concern: {validation_concern}")

    if c.leagues:
        llm_evidence, llm_why, llm_note = get_llm_reasoning_multi(user_q, selected, events)
    else:
        llm_evidence, llm_why, llm_note = get_llm_reasoning(user_q, selected, events, c.league)
    if llm_note:
        notes.append(llm_note)

    out = render_parlay(
        user_q,
        c,
        events,
        selected,
        notes,
        llm_evidence=llm_evidence,
        llm_why=llm_why,
    )
    print(out)
    # Parlays are NOT logged to the track record — they are user-requested
    # compositions optimized for odds targets, not standalone model opinions.
    # Only standalone market workflows (goals, corners, cards, SoT, BTTS,
    # moneyline, spreads) are tracked for accuracy.

    memory["last_selected_event_ids"] = [x.event_id for x in selected]
    memory["last_selected_fixtures"] = [x.fixture for x in selected]
    memory["last_leg_count"] = len(selected)
    memory["last_intent"] = top_intent
    memory["last_league"] = c.league
    memory["last_leagues"] = c.leagues or []
    memory["last_constraint_spec"] = {
        "hard_include_groups": list(c.hard_include_groups),
        "hard_exclude_groups": list(c.hard_exclude_groups),
        "soft_prefer_groups": list(c.soft_prefer_groups),
        "required_group_counts": dict(c.required_group_counts),
        "target_multiplier": c.target_multiplier,
        "target_mode": c.target_mode,
        "per_match_mode": c.per_match_mode,
        "require_unique_events": c.require_unique_events,
        "forbid_spread_keys": c.forbid_spread_keys,
        "require_total_corner_keys": c.require_total_corner_keys,
    }
    memory["last_selected_legs"] = [
        {"fixture": x.fixture, "market_key": x.market_key, "outcome": x.outcome,
         "odds": x.odds, "point": x.point, "market_group": market_group_from_key(x.market_key)}
        for x in selected
    ]
    memory["last_market_groups_used"] = list({market_group_from_key(x.market_key) for x in selected})
    add_chat_turn(raw_user_q, out)


# ---------------------------------------------------------------------------
# Calibration diagnostic
# ---------------------------------------------------------------------------

def check_calibration(league: str = "EPL", season: str = "2025/26"):
    """Print calibration stats: for each confidence bucket, what's the actual hit rate?

    This diagnostic function would:
    1. Load historical fixture outcomes from Output/{league}_feature_engineering/
    2. Re-run projections for each completed fixture (projected_total_goals, corners, cards, SoT)
    3. Bucket predictions by confidence level (high/medium/low from confidence_from_edge)
    4. Compare projected lines vs actual outcomes to compute hit rates per bucket
    5. Print a calibration table: bucket | n_predictions | hit_rate | avg_edge | avg_model_prob

    Expected output format:
        Confidence | Count | Hit Rate | Avg Edge | Avg Model P
        high       |    42 |   71.4%  |  +12.3%  |   0.68
        medium     |   118 |   58.5%  |   +5.1%  |   0.59
        low        |   203 |   49.8%  |   +1.2%  |   0.53

    A well-calibrated system should show hit rates that increase with confidence level.
    If high-confidence picks hit at the same rate as low-confidence, the confidence
    thresholds in confidence_from_edge() need recalibration.

    The actual backtest calibration is implemented in backtest.py — this stub exists
    as a reminder of the integration point and expected interface.

    Usage: python rag_cli_v2.py --calibrate EPL
    """
    print(f"[calibration] Placeholder for {league} {season} — use backtest.py for full calibration analysis.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Odds-first betting assistant (RAG v2, lightweight).")
    ap.add_argument("--league", type=str, default="EPL", help="Default league if query does not specify.")
    ap.add_argument("--once", type=str, default=None, help="Run one query and exit.")
    args = ap.parse_args()

    if args.once:
        answer_once(args.once, default_league=args.league)
        return

    print(f"RAG v2 ready. Default league={args.league}. Type 'exit' to quit.")
    while True:
        try:
            q = input("\nClient: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        answer_once(q, default_league=args.league)


if __name__ == "__main__":
    main()
