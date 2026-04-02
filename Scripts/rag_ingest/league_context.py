"""
league_context.py — Deterministic pre-match context aggregation for domestic leagues.

Collects context from existing modules and data:
- Rest days / short-rest penalty (from fixture dates)
- Congestion proxy (fixtures in recent window)
- Stake context from stake_context.py
- Regime shift: last-5 vs season delta in key stats
- Lineup context summary (when available)

All adjustments are bounded and capped via SCORING_WEIGHTS["league_context"].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("league_context")

# ---------------------------------------------------------------------------
# Imports with fallbacks
# ---------------------------------------------------------------------------

try:
    from core.weights import SCORING_WEIGHTS
except ImportError:
    try:
        from Scripts.rag_ingest.core.weights import SCORING_WEIGHTS
    except ImportError:
        SCORING_WEIGHTS = {}

try:
    from core.team_resolution import (
        _profile_meta,
        _recent_stats,
        get_recent_team_fixture_rows,
    )
except ImportError:
    try:
        from Scripts.rag_ingest.core.team_resolution import (
            _profile_meta,
            _recent_stats,
            get_recent_team_fixture_rows,
        )
    except ImportError:
        def _profile_meta(*a, **kw): return {}
        def _recent_stats(*a, **kw): return {}
        def get_recent_team_fixture_rows(*a, **kw): return []

try:
    from stake_context import get_stake_context, StakeContext
except ImportError:
    try:
        from Scripts.rag_ingest.stake_context import get_stake_context, StakeContext
    except ImportError:
        get_stake_context = None
        StakeContext = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LeagueContext:
    """Pre-match context for a domestic league fixture."""
    # Rest / congestion
    home_rest_days: Optional[int] = None
    away_rest_days: Optional[int] = None
    home_short_rest: bool = False       # <=3 days since last match
    away_short_rest: bool = False
    home_congestion: int = 0            # fixtures in last 14 days
    away_congestion: int = 0

    # Regime shift: recent form divergence from season baseline
    home_regime_shift: Dict[str, float] = field(default_factory=dict)
    away_regime_shift: Dict[str, float] = field(default_factory=dict)

    # Stake context (if available)
    stake_modifier: Dict[str, float] = field(default_factory=lambda: {
        "goals": 1.0, "corners": 1.0, "cards": 1.0, "sot": 1.0,
    })
    stake_description: str = ""

    # Bounded adjustments (applied to projections)
    adjustments: Dict[str, float] = field(default_factory=lambda: {
        "goals": 1.0, "corners": 1.0, "cards": 1.0, "sot": 1.0,
    })

    # Evidence lines for rendering
    evidence: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    source: str = "none"


# ---------------------------------------------------------------------------
# Rest days computation from fixture rows
# ---------------------------------------------------------------------------

REST_DAYS_CAP = 14
SHORT_REST_THRESHOLD = 3
CONGESTION_WINDOW_DAYS = 14


def _compute_rest_days(team: str, league: str, target_date: Optional[str] = None) -> Tuple[Optional[int], int]:
    """Compute rest days and congestion for a team from recent fixture rows.

    Returns (rest_days, congestion_count).
    rest_days: days since last match, or None if no fixture date data.
    congestion_count: number of fixtures in the last CONGESTION_WINDOW_DAYS days.
    """
    rows = get_recent_team_fixture_rows(team, league, limit=8)
    if not rows:
        return None, 0

    dates = []
    for r in rows:
        meta = r.get("meta") or {}
        fd = meta.get("fixture_date") or ""
        if fd and len(fd) >= 10:
            try:
                dates.append(datetime.strptime(fd[:10], "%Y-%m-%d"))
            except ValueError:
                continue

    if not dates:
        return None, 0

    dates.sort(reverse=True)  # most recent first

    if target_date:
        try:
            ref = datetime.strptime(target_date[:10], "%Y-%m-%d")
        except ValueError:
            ref = datetime.now()
    else:
        ref = datetime.now()

    rest_days = min(REST_DAYS_CAP, max(1, (ref - dates[0]).days))
    congestion = sum(1 for d in dates if (ref - d).days <= CONGESTION_WINDOW_DAYS)

    return rest_days, congestion


# ---------------------------------------------------------------------------
# Regime shift: last-5 vs season delta
# ---------------------------------------------------------------------------

REGIME_STATS = ["corners_for_avg", "cards_avg", "sot_for_avg", "xg_for_avg",
                "control_avg", "possession_avg"]

REGIME_STAT_TO_GROUP = {
    "corners_for_avg": "corners",
    "cards_avg": "cards",
    "sot_for_avg": "sot",
    "xg_for_avg": "goals",
    "control_avg": "control",
    "possession_avg": "possession",
}


def _safe_float(val, default=None):
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _compute_regime_shift(team: str, league: str) -> Dict[str, float]:
    """Compute regime shift: (recent_5 - season) / season for key stats.

    Positive = team trending up. Negative = trending down.
    """
    recent = _recent_stats(team, league, last_n=5)
    profile = _profile_meta(team, league)
    if not recent or not profile:
        return {}

    season_map = {
        "corners_for_avg": "corners_pm",
        "cards_avg": "cards_per_90_team",
        "sot_for_avg": "sot_for_pm",
        "xg_for_avg": "goals_for_pm",
        "control_avg": "control_index",
        "possession_avg": "possession",
    }

    shift = {}
    for stat_key in REGIME_STATS:
        recent_val = _safe_float(recent.get(stat_key))
        season_key = season_map.get(stat_key, "")
        season_val = _safe_float(profile.get(season_key))
        if recent_val is not None and season_val is not None and season_val > 0:
            delta = (recent_val - season_val) / season_val
            shift[REGIME_STAT_TO_GROUP.get(stat_key, stat_key)] = round(delta, 4)

    return shift


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_league_context(
    home: str,
    away: str,
    league: str,
    target_date: Optional[str] = None,
) -> LeagueContext:
    """Build pre-match league context for a domestic fixture.

    Aggregates rest days, congestion, stake context, and regime shift.
    Returns bounded adjustments that can be applied to projections.
    """
    ctx = LeagueContext()
    lcw = SCORING_WEIGHTS.get("league_context", {})
    evidence = []
    warnings = []

    # --- Rest days ---
    h_rest, h_cong = _compute_rest_days(home, league, target_date)
    a_rest, a_cong = _compute_rest_days(away, league, target_date)
    ctx.home_rest_days = h_rest
    ctx.away_rest_days = a_rest
    ctx.home_congestion = h_cong
    ctx.away_congestion = a_cong

    if h_rest is not None and h_rest <= SHORT_REST_THRESHOLD:
        ctx.home_short_rest = True
    if a_rest is not None and a_rest <= SHORT_REST_THRESHOLD:
        ctx.away_short_rest = True

    # Short-rest penalty: reduce expected output slightly
    short_rest_penalty = lcw.get("short_rest_penalty", 0.03)  # 3% reduction per short-rest team
    rest_adj = 1.0
    if ctx.home_short_rest:
        rest_adj -= short_rest_penalty
        evidence.append(f"{home}: short rest ({h_rest}d)")
    if ctx.away_short_rest:
        rest_adj -= short_rest_penalty
        evidence.append(f"{away}: short rest ({a_rest}d)")

    # Congestion penalty: if both teams have 4+ fixtures in 14 days
    congestion_threshold = lcw.get("congestion_threshold", 4)
    congestion_penalty = lcw.get("congestion_penalty", 0.02)
    if h_cong >= congestion_threshold:
        rest_adj -= congestion_penalty
        evidence.append(f"{home}: {h_cong} fixtures in {CONGESTION_WINDOW_DAYS}d")
    if a_cong >= congestion_threshold:
        rest_adj -= congestion_penalty
        evidence.append(f"{away}: {a_cong} fixtures in {CONGESTION_WINDOW_DAYS}d")

    # Cap rest adjustment
    rest_floor = lcw.get("rest_adjustment_floor", 0.90)
    rest_adj = max(rest_floor, rest_adj)

    # --- Stake context ---
    if get_stake_context is not None:
        try:
            stake = get_stake_context(home, away, league)
            if stake and stake.source != "none":
                ctx.stake_modifier = {
                    "goals": stake.goals_modifier,
                    "corners": stake.corners_modifier,
                    "cards": stake.cards_modifier,
                    "sot": stake.sot_modifier,
                }
                ctx.stake_description = stake.description
                if stake.description:
                    evidence.append(f"Stake: {stake.description}")
        except Exception as exc:
            log.debug("Stake context failed: %s", exc)

    # --- Regime shift ---
    ctx.home_regime_shift = _compute_regime_shift(home, league)
    ctx.away_regime_shift = _compute_regime_shift(away, league)

    # Compute regime adjustment per stat: average of home + away deltas, bounded
    regime_weight = lcw.get("regime_shift_weight", 0.08)
    regime_cap = lcw.get("regime_shift_cap", 0.06)

    for stat in ["goals", "corners", "cards", "sot"]:
        h_delta = ctx.home_regime_shift.get(stat, 0.0)
        a_delta = ctx.away_regime_shift.get(stat, 0.0)
        avg_delta = (h_delta + a_delta) / 2.0
        regime_adj = max(-regime_cap, min(regime_cap, avg_delta * regime_weight))

        # Combine: rest * stake * (1 + regime)
        stake_mod = ctx.stake_modifier.get(stat, 1.0)
        combined = rest_adj * stake_mod * (1.0 + regime_adj)

        # Final bounds
        adj_floor = lcw.get("adjustment_floor", 0.85)
        adj_ceiling = lcw.get("adjustment_ceiling", 1.15)
        ctx.adjustments[stat] = round(max(adj_floor, min(adj_ceiling, combined)), 4)

        # Evidence for significant regime shifts
        if abs(avg_delta) >= 0.15:
            direction = "up" if avg_delta > 0 else "down"
            evidence.append(f"{stat} regime: {direction} {abs(avg_delta):.0%}")

    ctx.evidence = evidence
    ctx.warnings = warnings
    ctx.source = "computed"

    return ctx


def league_context_evidence_line(ctx: LeagueContext) -> Optional[str]:
    """Generate a human-readable evidence line from league context."""
    if not ctx.evidence or ctx.source == "none":
        return None
    return "Context: " + " | ".join(ctx.evidence[:4]) + "."
