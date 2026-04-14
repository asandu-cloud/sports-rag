"""
rendering/totals.py — render_totals_line_answer and render_goal_intervals.

Extracted from rag_cli_v2.py without modification.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Cross-module imports with fallbacks
# ---------------------------------------------------------------------------

try:
    from core.weights import SCORING_WEIGHTS, EUROPEAN_COMPETITIONS
except ImportError:
    try:
        from Scripts.rag_ingest.core.weights import SCORING_WEIGHTS, EUROPEAN_COMPETITIONS
    except ImportError:
        try:
            from rag_cli_v2 import SCORING_WEIGHTS, EUROPEAN_COMPETITIONS
        except ImportError:
            SCORING_WEIGHTS = {}
            EUROPEAN_COMPETITIONS = {"UCL", "UEL", "UECL"}

try:
    from core.projections import (
        projected_total_goals, projected_total_corners, projected_total_cards,
        projected_total_sot, projected_total_cards_detail,
    )
except ImportError:
    try:
        from Scripts.rag_ingest.core.projections import (
            projected_total_goals, projected_total_corners, projected_total_cards,
            projected_total_sot, projected_total_cards_detail,
        )
    except ImportError:
        try:
            from rag_cli_v2 import (
                projected_total_goals, projected_total_corners, projected_total_cards,
                projected_total_sot, projected_total_cards_detail,
            )
        except ImportError:
            def projected_total_goals(*a, **kw): return (None, None, None)
            def projected_total_corners(*a, **kw): return (None, None, None)
            def projected_total_cards(*a, **kw): return (None, None, None, None)
            def projected_total_sot(*a, **kw): return (None, None, None)
            def projected_total_cards_detail(*a, **kw): return {
                "total_cards": None, "total_yellows": None, "total_reds": None,
                "season_total": None, "recent_total": None, "referee_modifier": None,
                "lineup_card_context": None, "uncertainty": 999.0,
                "confidence_bucket": "low", "warnings": [],
            }

try:
    from core.projections import compute_stat_confidence
except ImportError:
    try:
        from Scripts.rag_ingest.core.projections import compute_stat_confidence
    except ImportError:
        def compute_stat_confidence(*a, **kw): return "medium"

try:
    from core.odds_extraction import extract_total_line_options, extract_goal_interval_odds
except ImportError:
    try:
        from Scripts.rag_ingest.core.odds_extraction import extract_total_line_options, extract_goal_interval_odds
    except ImportError:
        try:
            from rag_cli_v2 import extract_total_line_options, extract_goal_interval_odds
        except ImportError:
            def extract_total_line_options(*a, **kw): return []
            def extract_goal_interval_odds(*a, **kw): return {}

try:
    from core.line_selection import (
        choose_best_total_line, confidence_from_edge, _best_model_only_line,
        select_best_total_recommendation,
    )
except ImportError:
    try:
        from Scripts.rag_ingest.core.line_selection import (
            choose_best_total_line, confidence_from_edge, _best_model_only_line,
            select_best_total_recommendation,
        )
    except ImportError:
        try:
            from rag_cli_v2 import choose_best_total_line, confidence_from_edge, _best_model_only_line
            def select_best_total_recommendation(options, projection_total, combined_var=None):
                best = choose_best_total_line(options, projection_total, combined_var)
                return {
                    "best_value": best,
                    "bet_recommendation": best,
                    "recommended": best,
                    "no_bet_reason": None if best else "No totals line available.",
                    "all_lines": options,
                }
        except ImportError:
            def choose_best_total_line(*a, **kw): return None
            def confidence_from_edge(*a, **kw): return "low"
            def _best_model_only_line(*a, **kw): return ("Over", 2.5, 0.50)
            def select_best_total_recommendation(*a, **kw):
                return {"best_value": None, "bet_recommendation": None, "recommended": None,
                        "no_bet_reason": "Totals selector unavailable.", "all_lines": []}

try:
    from core.team_resolution import get_blended_variance, requested_stat_group
except ImportError:
    try:
        from Scripts.rag_ingest.core.team_resolution import get_blended_variance, requested_stat_group
    except ImportError:
        try:
            from rag_cli_v2 import get_blended_variance, requested_stat_group
        except ImportError:
            def get_blended_variance(*a, **kw): return None
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

try:
    from rendering.evidence import _euro_data_source_note
except ImportError:
    try:
        from Scripts.rag_ingest.rendering.evidence import _euro_data_source_note
    except ImportError:
        try:
            from rag_cli_v2 import _euro_data_source_note
        except ImportError:
            def _euro_data_source_note(team: str, league: str) -> Optional[str]:
                return None

try:
    from prob_models import interval_prob, implied_prob, value_edge
except ImportError:
    try:
        from Scripts.rag_ingest.prob_models import interval_prob, implied_prob, value_edge
    except ImportError:
        def interval_prob(*a, **kw): return 0.0
        def implied_prob(odds): return 1.0 / odds if odds > 0 else 0.0
        def value_edge(mp, ip): return mp - ip

try:
    from knockout_context import get_knockout_context, knockout_evidence_line
    _HAS_KNOCKOUT = True
except ImportError:
    try:
        from Scripts.rag_ingest.knockout_context import get_knockout_context, knockout_evidence_line
        _HAS_KNOCKOUT = True
    except ImportError:
        _HAS_KNOCKOUT = False
        def get_knockout_context(*a, **kw):
            from dataclasses import dataclass
            @dataclass
            class _KO:
                is_knockout: bool = False
                goals_modifier: float = 1.0
                corners_modifier: float = 1.0
                cards_modifier: float = 1.0
                sot_modifier: float = 1.0
                source: str = "none"
            return _KO()
        def knockout_evidence_line(ctx) -> None:
            return None

try:
    from league_context import get_league_context, league_context_evidence_line
except ImportError:
    try:
        from Scripts.rag_ingest.league_context import get_league_context, league_context_evidence_line
    except ImportError:
        def get_league_context(*args, **kwargs):
            return None
        def league_context_evidence_line(ctx):
            return None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Exported functions
# ---------------------------------------------------------------------------


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
        fixture_date = str(ev.get("commence_time") or ev.get("fixture_date") or "")
        league_ctx = None
        try:
            league_ctx = get_league_context(home, away, league, target_date=fixture_date)
        except Exception:
            league_ctx = None

        # Knockout round context for European competitions
        ko_ctx = None
        if league in EUROPEAN_COMPETITIONS:
            try:
                ko_date = ev.get("commence_time", "")
                ko_ctx = get_knockout_context(home, away, league, ko_date)
            except Exception:
                ko_ctx = None

        _cards_ref_mod = None
        _cards_detail = None
        if group == "goals":
            proj_total, season_total, recent_total = projected_total_goals(
                home, away, league, knockout_ctx=ko_ctx,
                league_ctx=league_ctx, fixture_date=fixture_date,
            )
        elif group == "corners":
            proj_total, season_total, recent_total = projected_total_corners(
                home, away, league, knockout_ctx=ko_ctx,
                league_ctx=league_ctx, fixture_date=fixture_date,
            )
        elif group == "sot":
            proj_total, season_total, recent_total = projected_total_sot(
                home, away, league, knockout_ctx=ko_ctx,
                league_ctx=league_ctx, fixture_date=fixture_date,
            )
        else:
            _cards_detail = projected_total_cards_detail(
                home, away, league, knockout_ctx=ko_ctx,
                league_ctx=league_ctx, fixture_date=fixture_date,
            )
            proj_total = _cards_detail.get("total_cards")
            season_total = _cards_detail.get("season_total")
            recent_total = _cards_detail.get("recent_total")
            _cards_ref_mod = _cards_detail.get("referee_modifier")

        if proj_total is None:
            lines.append(f"- {fixture}: insufficient profile data to project totals.")
            continue

        # Compute general confidence bucket for all stats
        _stat_confidence = "medium"
        if group == "cards" and _cards_detail:
            _stat_confidence = _cards_detail.get("confidence_bucket", "medium")
        else:
            try:
                _stat_confidence = compute_stat_confidence(group, home, away, league)
            except Exception:
                pass

        # Knockout context evidence
        if ko_ctx and ko_ctx.is_knockout:
            ko_line = knockout_evidence_line(ko_ctx)
            if ko_line:
                lines.append(f"  {ko_line}")
        ctx_line = league_context_evidence_line(league_ctx) if league_ctx is not None else None
        if ctx_line:
            lines.append(f"  {ctx_line}")

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
        total_result = select_best_total_recommendation(options, proj_total, combined_var)
        best = total_result.get("bet_recommendation")
        lines.append(f"- {fixture}: projected total {proj_total:.2f}.")
        if _stat_confidence == "low" and group != "cards":  # cards already has its own warning
            lines.append(f"  Note: low confidence — limited data for one or both teams.")

        if season_total is not None:
            lines.append(f"  Season model: {season_total:.2f}.")
        if recent_total is not None:
            lines.append(f"  Recent 6-match: {recent_total:.2f}.")

        # Referee profile and card detail for cards totals
        if group == "cards" and _cards_detail:
            # Yellow/red split
            t_yellows = _cards_detail.get("total_yellows")
            t_reds = _cards_detail.get("total_reds")
            if t_yellows is not None and t_reds is not None:
                lines.append(f"  Yellow/red split: ~{t_yellows:.2f} yellows + ~{t_reds:.2f} reds.")

            if _cards_ref_mod and _cards_ref_mod.source == "profile":
                ref_name = (_cards_ref_mod.referee_name or "Unknown").title()
                ref_avg = _cards_ref_mod.avg_cards_per_match
                ref_strict = _cards_ref_mod.strictness_ratio
                ref_n = _cards_ref_mod.sample_size
                ref_conf = _cards_ref_mod.confidence
                if ref_strict >= 1.10:
                    label = "strict"
                elif ref_strict <= 0.90:
                    label = "lenient"
                else:
                    label = "average"
                conf_label = "high" if ref_conf >= 0.70 else ("medium" if ref_conf >= 0.35 else "low")
                lines.append(
                    f"  Referee: {ref_name} — {ref_avg:.2f} cards/match ({label}, "
                    f"{ref_strict:.2f}x league avg, {ref_n} matches, {conf_label} confidence)."
                )
                ref_cpf = _cards_ref_mod.cards_per_foul
                if ref_cpf > 0:
                    lines.append(
                        f"  Ref cards/foul: {ref_cpf:.3f} — "
                        f"avg fouls/match: {_cards_ref_mod.avg_fouls_per_match:.1f}."
                    )
            else:
                lines.append("  Referee: not yet assigned or API-Football key not set.")

            # Lineup card risk
            lcc = _cards_detail.get("lineup_card_context")
            if lcc:
                lines.append(
                    f"  Lineup card risk: {lcc.get('combined_risk_90', 0):.2f} combined cards/90 "
                    f"(high-risk starters: {lcc.get('home_high_risk', 0)} home, {lcc.get('away_high_risk', 0)} away)."
                )

            # Uncertainty and warnings
            uncertainty = _cards_detail.get("uncertainty", 0)
            conf_bucket = _cards_detail.get("confidence_bucket", "low")
            if conf_bucket == "low":
                lines.append(f"  ⚠ Low confidence (uncertainty: {uncertainty:.2f}). Treat as estimate only.")
            card_warnings = _cards_detail.get("warnings", [])
            for w in card_warnings:
                lines.append(f"  Note: {w}")

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
            # Suppress "Recommended" label for low-confidence projections
            rec_label = "Recommended"
            if _stat_confidence == "low":
                rec_label = "Estimate"
            lines.append(
                f"  {rec_label}: {side} {point:g} @ {odds:.2f} ({best.get('bookmaker')}, {best.get('market_key')})."
            )
            if model_p is not None and implied_p is not None and val_edge is not None:
                lines.append(
                    f"  Model: P({side} {point:g}) = {model_p:.1%} | Fair implied: {implied_p:.1%} | Value edge: {val_edge:+.1%}"
                )
            if ev is not None:
                lines.append(f"  Expected value: {ev:+.3f} per unit staked ({conf} confidence).")
            else:
                lines.append(f"  Model edge {edge:+.2f} ({conf} confidence).")
        elif options:
            reason = total_result.get("no_bet_reason") or "No totals line clears the betting thresholds."
            lines.append(f"  Verdict: No totals bet.")
            lines.append(f"  Reason: {reason}")
        else:
            side, anchor, mp = _best_model_only_line(proj_total, combined_var)
            lines.append(
                f"  Recommended: {side} {anchor:g} "
                f"(model-only, P({side} {anchor:g}) = {mp:.1%})."
            )

    lines.append("Method: Poisson probability model from local KB + available full-game market lines.")
    return "\n".join(lines)


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

            # Only recommend from bands with <=3 goals (0-1 or 2-3)
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
