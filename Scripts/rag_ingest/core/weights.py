"""
core/weights.py — Centralized scoring weights and league/competition constants.

Extracted verbatim from rag_cli_v2.py.  Do NOT modify values here without
updating the original file (until the migration is complete).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# League / Competition constants
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Scoring Weights (centralized for regression testing + tuning)
# ---------------------------------------------------------------------------

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
        "home_advantage": 0.0,  # ~0.15 goals boost for home team (split: +0.15 home, -0.15 away)
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
        "corners_direct": {
            "season": 0.65,
            "recent": 0.35,
        },
        "corners_output": {
            "match_anchor_weight": 0.55,
            "direct_stabilizer_weight": 0.45,
        },
        "corners_share": {
            "season_blend": 0.65,
            "recent_blend": 0.35,
            "shrink": 0.06,
            "floor": 0.28,
            "ceiling": 0.72,
            # Season share features
            "season_own": 0.35,
            "season_opp_allow": 0.40,
            "season_control": 0.10,
            "season_dominance": 0.08,
            "season_possession": 0.07,
            # Recent share features
            "recent_own": 0.40,
            "recent_opp_allow": 0.35,
            "recent_shots": 0.05,
            "recent_sot": 0.05,
            "recent_control": 0.08,
            "recent_possession": 0.07,
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
        "positive_ev_weight": 2.2,
        "min_model_prob": 0.56,
        "min_value_edge": 0.015,
        "min_ev": 0.015,
        "min_goal_edge": 0.18,
        "short_price_extra_model_prob": 0.02,
        "short_price_extra_ev": 0.01,
    },
    "moneyline_selection": {
        # Winner prediction and value-side identification are separate.
        "min_value_edge": 0.03,
        "min_ev": 0.05,
        "min_model_prob_home_away": 0.40,
        "min_model_prob_draw": 0.32,
        "longshot_odds_threshold": 4.50,
        "longshot_min_model_prob": 0.20,
        "longshot_extra_edge": 0.05,
        "longshot_extra_ev": 0.10,
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
        "elo_signal_weight": 0.15,
        "elo_scale": 400.0,
        "min_elo_diff": 20,
    },
    "cards_total_model": {
        # Yellow-card base model: match total yellows is the primary anchor
        "yellows_base_weight": 0.82,      # weight on yellow-card season projection
        "reds_add_weight": 0.18,          # weight on red-card additive adjustment
        "reds_base_rate": 0.18,           # league-average reds per match (both teams)
        "reds_cap": 0.60,                 # max reds contribution to total
        # Recent induced-card component
        "induced_weight": 0.12,           # weight on recent induced-cards signal
        # Lineup card-risk component
        "lineup_risk_weight": 0.08,       # weight on lineup card-risk adjustment
        "lineup_risk_cap": 0.25,          # max absolute adjustment from lineup risk
    },
    "cards_yellow_red_split": {
        # Season yellow/red split estimation
        "default_yellow_ratio": 0.92,     # typical yellows / total cards
        "default_red_ratio": 0.08,        # typical reds / total cards
        "min_yellow_ratio": 0.80,         # floor
        "max_yellow_ratio": 0.98,         # ceiling
    },
    "cards_share_model": {
        # Team share of match-total cards (extends existing cards_share)
        "match_anchor_weight": 0.55,      # weight on match-total-derived share
        "direct_stabilizer_weight": 0.45, # weight on direct team-card stabilizer
        "yellows_share_weight": 0.70,     # weight on yellow-card share signal
        "reds_share_weight": 0.30,        # weight on red-card share signal
    },
    "cards_confidence": {
        # Confidence gating for cards recommendations
        "min_fixtures_season": 5,         # min season fixtures for season anchor
        "min_fixtures_recent": 3,         # min recent fixtures for recent anchor
        "referee_high_confidence": 0.70,  # referee confidence threshold for "high"
        "referee_medium_confidence": 0.35, # threshold for "medium"
        "low_confidence_suppress": True,  # suppress "Recommended" when low confidence
        "uncertainty_base": 0.8,          # base uncertainty (cards per match)
        "uncertainty_no_referee": 0.3,    # additional uncertainty when no referee
        "uncertainty_no_lineup": 0.1,     # additional uncertainty when no lineup
        "uncertainty_low_sample": 0.2,    # additional uncertainty when <10 fixtures
    },
    "league_context": {
        "short_rest_penalty": 0.03,        # 3% reduction per short-rest team (<=3 days)
        "congestion_threshold": 4,         # fixtures in 14-day window to trigger penalty
        "congestion_penalty": 0.02,        # 2% reduction per congested team
        "rest_adjustment_floor": 0.90,     # max 10% total rest-based reduction
        "regime_shift_weight": 0.08,       # how much regime shift influences projection
        "regime_shift_cap": 0.06,          # max 6% regime-based adjustment
        "adjustment_floor": 0.85,          # absolute floor for combined adjustment
        "adjustment_ceiling": 1.15,        # absolute ceiling for combined adjustment
    },
    "market_blend": {
        "model_weight": 0.70,
        "market_weight": 0.30,
    },
    "recency": {
        "alpha": 0.85,  # exponential decay per match (most recent=1.0, next=0.85, then 0.72...)
        "trend_weight": 0.0,  # disabled: slope signal too noisy with 6-game windows
    },
    "european": {
        "domestic_weight": 0.80,       # weight for domestic league profile in blended projections
        "euro_weight": 0.20,           # weight for European competition profile
        "min_euro_fixtures": 3,        # minimum European fixtures before blending kicks in
        "non_top5_confidence_penalty": 0.15,  # confidence penalty for teams without domestic profiles
    },
}
