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

        # Use detail helper for richer cards evidence
        try:
            from core.projections import projected_total_cards_detail as _ptcd_render
        except ImportError:
            _ptcd_render = None
        _detail = None
        if _ptcd_render:
            try:
                _detail = _ptcd_render(leg.home_team, leg.away_team, league)
            except Exception:
                pass
        if _detail:
            cards_total = _detail.get("total_cards")
            cards_ref_mod = _detail.get("referee_modifier")
        else:
            cards_total, _, _, cards_ref_mod = projected_total_cards(
                leg.home_team, leg.away_team, league
            )
        if cards_total is not None:
            lines.append(f"Projected combined cards: {cards_total:.2f}.")
            if _detail:
                t_y = _detail.get("total_yellows")
                t_r = _detail.get("total_reds")
                if t_y is not None and t_r is not None:
                    lines.append(f"Split: ~{t_y:.2f} yellows + ~{t_r:.2f} reds.")
            if leg.point is not None:
                direction = leg.outcome.lower()
                if "over" in direction:
                    lines.append(f"Line fit: projected {cards_total:.2f} vs over {leg.point:g}.")
                elif "under" in direction:
                    lines.append(f"Line fit: projected {cards_total:.2f} vs under {leg.point:g}.")
            if _detail and _detail.get("confidence_bucket") == "low":
                lines.append("Cards confidence: low — higher uncertainty.")

        # Referee evidence for cards
        if cards_ref_mod.source == "profile" and cards_ref_mod.referee_name:
            rw = SCORING_WEIGHTS["referee"]
            if abs(cards_ref_mod.multiplier - 1.0) > rw["evidence_threshold"]:
                direction_word = "stricter" if cards_ref_mod.multiplier > 1.0 else "more lenient"
                lines.append(
                    f"Referee: {cards_ref_mod.referee_name.title()} is {direction_word} than avg "
                    f"({cards_ref_mod.strictness_ratio:.2f}x, {cards_ref_mod.sample_size} matches, "
                    f"{cards_ref_mod.confidence:.0%} confidence)."
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
            direction = leg.outcome.lower()
            is_over = "over" in direction
            h_var = get_team_recent_variance(leg.home_team, league)
            a_var = get_team_recent_variance(leg.away_team, league)
            var_key = {"corners": "corners_for_var", "cards": "cards_var",
                       "sot": "sot_for_var", "totals": "goals_var"}[g]
            h_v, a_v = h_var.get(var_key), a_var.get(var_key)
            comb_var = (h_v + a_v) if (h_v is not None and a_v is not None) else None
            model_p = over_prob(proj_total, leg.point, comb_var) if is_over else under_prob(proj_total, leg.point, comb_var)
            implied_p_val = implied_prob(leg.odds)
            ve = value_edge(model_p, implied_p_val)
            side_label = "Over" if is_over else "Under"
            lines.append(f"P({side_label} {leg.point:g}) = {model_p:.1%} | Books: {implied_p_val:.1%} | Value: {ve:+.1%}")

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
    # prioritize multi-signal support -- allow extra lines for probability + ML
    return lines[:6]


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
        conf_str = f" ({conf_label} confidence" + (f", model {model_p:.0%}" if model_p else "") + ")"
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
