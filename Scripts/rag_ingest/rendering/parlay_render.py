"""
rendering/parlay_render.py — render_parlay, leg_evidence, leg_label.

Extracted from rag_cli_v2.py without modification.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Cross-module imports with fallbacks
# ---------------------------------------------------------------------------

try:
    from core.weights import SCORING_WEIGHTS
except ImportError:
    try:
        from Scripts.rag_ingest.core.weights import SCORING_WEIGHTS
    except ImportError:
        try:
            from rag_cli_v2 import SCORING_WEIGHTS
        except ImportError:
            SCORING_WEIGHTS = {
                "referee": {"evidence_threshold": 0.02},
                "prob": {"prob_quality_weight": 1.5},
            }

try:
    from core.projections import (
        projected_corners, projected_total_sot, projected_total_cards,
        projected_total_corners, projected_total_goals,
        projected_btts_prob, projected_goal_difference, projected_moneyline_probs,
        projected_cards,
    )
except ImportError:
    try:
        from Scripts.rag_ingest.core.projections import (
            projected_corners, projected_total_sot, projected_total_cards,
            projected_total_corners, projected_total_goals,
            projected_btts_prob, projected_goal_difference, projected_moneyline_probs,
            projected_cards,
        )
    except ImportError:
        try:
            from rag_cli_v2 import (
                projected_corners, projected_total_sot, projected_total_cards,
                projected_total_corners, projected_total_goals,
                projected_btts_prob, projected_goal_difference, projected_moneyline_probs,
                projected_cards,
            )
        except ImportError:
            def projected_corners(*a, **kw): return (None, None)
            def projected_total_sot(*a, **kw): return (None, None, None)
            def projected_total_cards(*a, **kw): return (None, None, None, None)
            def projected_total_corners(*a, **kw): return (None, None, None)
            def projected_total_goals(*a, **kw): return (None, None, None)
            def projected_btts_prob(*a, **kw): return (None, None, None, None, None)
            def projected_goal_difference(*a, **kw): return (None, None, None)
            def projected_moneyline_probs(*a, **kw): return (None, None, None, None, None)
            def projected_cards(*a, **kw): return (None, None)

try:
    from core.team_resolution import _profile_meta, _recent_stats, get_team_recent_variance
except ImportError:
    try:
        from Scripts.rag_ingest.core.team_resolution import _profile_meta, _recent_stats, get_team_recent_variance
    except ImportError:
        try:
            from rag_cli_v2 import _profile_meta, _recent_stats, get_team_recent_variance
        except ImportError:
            def _profile_meta(team: str, league: str) -> dict:
                return {}
            def _recent_stats(team: str, league: str, last_n: int = 6) -> dict:
                return {}
            def get_team_recent_variance(team: str, league: str, last_n: int = 8) -> dict:
                return {}

try:
    from core.line_selection import confidence_from_edge
except ImportError:
    try:
        from Scripts.rag_ingest.core.line_selection import confidence_from_edge
    except ImportError:
        try:
            from rag_cli_v2 import confidence_from_edge
        except ImportError:
            def confidence_from_edge(*a, **kw): return "low"

try:
    from prob_models import over_prob, under_prob, implied_prob, value_edge
except ImportError:
    try:
        from Scripts.rag_ingest.prob_models import over_prob, under_prob, implied_prob, value_edge
    except ImportError:
        def over_prob(*a, **kw): return 0.5
        def under_prob(*a, **kw): return 0.5
        def implied_prob(odds): return 1.0 / odds if odds > 0 else 0.0
        def value_edge(mp, ip): return mp - ip

try:
    from ml_edge import ml_predict_total, get_elo_edge
    _HAS_ML = True
except ImportError:
    try:
        from Scripts.rag_ingest.ml_edge import ml_predict_total, get_elo_edge
        _HAS_ML = True
    except ImportError:
        _HAS_ML = False
        def ml_predict_total(*a, **kw): return None
        def get_elo_edge(*a, **kw): return None

try:
    from heuristics import top_markets
except ImportError:
    try:
        from Scripts.rag_ingest.heuristics import top_markets
    except ImportError:
        def top_markets(*a, **kw): return []

# Import types and helpers that exist in rag_cli_v2 or core modules
try:
    from rag_cli_v2 import (
        CandidateLeg, ConstraintSpec, market_group_from_key, _leg_league,
        MIN_LEG_ODDS, build_candidates, available_market_groups,
        combined_odds, summarize_constraints, heuristic_groups_for_event,
        _numeric, _leg_standalone_confidence,
    )
except ImportError:
    try:
        from core.parlay import CandidateLeg, market_group_from_key, _leg_league, MIN_LEG_ODDS, build_candidates, available_market_groups, combined_odds, heuristic_groups_for_event
        from rag_cli_v2 import ConstraintSpec, summarize_constraints, _numeric, _leg_standalone_confidence
    except ImportError:
        # Minimal stubs — these will be resolved at runtime when rag_cli_v2.py
        # is the actual entry point
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class CandidateLeg:
            event_id: str = ""
            fixture: str = ""
            home_team: str = ""
            away_team: str = ""
            market_key: str = ""
            outcome: str = ""
            odds: float = 1.0
            point: Optional[float] = None
            bookmaker: Optional[str] = None
            league: str = ""

        @dataclass
        class ConstraintSpec:
            requested_markets: set = None
            hard_include_groups: set = None
            hard_exclude_groups: set = None
            soft_prefer_groups: set = None
            required_group_counts: dict = None
            forbid_spread_keys: bool = False
            require_total_corner_keys: bool = False
            target_multiplier: Optional[float] = None
            target_mode: str = "none"
            leg_count: Optional[int] = None
            per_match_mode: bool = False
            require_unique_events: bool = True
            time_window: str = "upcoming"
            league: str = "EPL"
            leagues: list = None
            target_min: Optional[float] = None
            target_max: Optional[float] = None

        MIN_LEG_ODDS = 1.10

        def market_group_from_key(market_key: str) -> str:
            k = (market_key or "").lower()
            if "corner" in k: return "corners"
            if "card" in k or "booking" in k: return "cards"
            if ("shot" in k and "target" in k) or k == "sot": return "sot"
            if "btts" in k or "both_teams" in k: return "btts"
            if "h2h" in k or "moneyline" in k or "1x2" in k: return "moneyline"
            if "spread" in k or "handicap" in k: return "spreads"
            if "total" in k: return "totals"
            return k

        def _leg_league(leg, fallback: str) -> str:
            return leg.league if leg.league else fallback

        def build_candidates(events): return []
        def available_market_groups(candidates): return set()
        def combined_odds(legs):
            p = 1.0
            for leg in legs:
                p *= leg.odds
            return p
        def summarize_constraints(c, available_groups, selected): return ([], [])
        def heuristic_groups_for_event(hm, am): return set()
        def _numeric(value):
            try: return float(value)
            except Exception: return None
        def _leg_standalone_confidence(leg, league): return {"confidence": None, "model_prob": None, "side_agrees": True}


# ---------------------------------------------------------------------------
# Exported functions
# ---------------------------------------------------------------------------


def leg_label(leg: CandidateLeg) -> str:
    g = market_group_from_key(leg.market_key)
    if leg.point is None:
        return f"{leg.outcome} ({leg.market_key}/{g})"
    if g == "spreads":
        return f"{leg.outcome} {leg.point:+g} ({leg.market_key}/{g})"
    return f"{leg.outcome} {leg.point:g} ({leg.market_key}/{g})"


def leg_evidence(leg, league):
    from core.quote_assessment import quote_evidence
    return quote_evidence(leg, league)


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
        conf_label = sc.get("confidence") or "\u2014"
        model_p = sc.get("model_prob")
        conf_str = f" ({conf_label} confidence" + (f", {'price-comparison probability' if sc.get('probability_basis') == 'asian_equivalent_non_push' else 'outcome probability'} {model_p:.0%}" if model_p else "") + ")"
        leg_confidence_cache.append(conf_label if conf_label != "\u2014" else None)

        lines.append(f"- Leg {i}: {lg_badge}{leg.fixture} | {leg_label(leg)} @ {leg.odds:.2f}{bm}{conf_str}")

        # Cross-validation warning
        if not sc.get("side_agrees", True) and sc.get("warning"):
            lines.append(f"  \u26a0 {sc['warning']}")
            warnings.append(f"Leg {i}: {sc['warning']}")
        elif not sc.get("side_agrees", True) and sc.get("standalone_side"):
            g = market_group_from_key(leg.market_key)
            lines.append(f"  \u26a0 Standalone {g} model favors {sc['standalone_side']}")
            warnings.append(f"Leg {i}: standalone {g} model favors {sc['standalone_side']}")

        ev_lines = (llm_evidence or {}).get(i) or leg_evidence(leg, league)
        for ev in ev_lines[:3]:
            lines.append(f"  Evidence: {ev}")

    lines.append(f"- Individual price product (combined quote unverified): {combo:.2f}x")

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
        lines.append(f"- \u26a0 Cross-check warnings: {len(warnings)} leg(s) disagree with standalone projections")

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
