"""
rendering/spreads.py — render_spreads_line_answer.

Extracted from rag_cli_v2.py without modification.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Cross-module imports with fallbacks
# ---------------------------------------------------------------------------

try:
    from core.weights import EUROPEAN_COMPETITIONS
except ImportError:
    try:
        from Scripts.rag_ingest.core.weights import EUROPEAN_COMPETITIONS
    except ImportError:
        try:
            from rag_cli_v2 import EUROPEAN_COMPETITIONS
        except ImportError:
            EUROPEAN_COMPETITIONS = {"UCL", "UEL", "UECL"}

try:
    from core.projections import projected_goal_difference
except ImportError:
    try:
        from Scripts.rag_ingest.core.projections import projected_goal_difference
    except ImportError:
        try:
            from rag_cli_v2 import projected_goal_difference
        except ImportError:
            def projected_goal_difference(*a, **kw): return (None, None, None)

try:
    from core.odds_extraction import extract_spread_line_options
except ImportError:
    try:
        from Scripts.rag_ingest.core.odds_extraction import extract_spread_line_options
    except ImportError:
        try:
            from rag_cli_v2 import extract_spread_line_options
        except ImportError:
            def extract_spread_line_options(*a, **kw): return []

try:
    from core.line_selection import (
        choose_best_spread_line, confidence_from_edge, _best_model_only_spread,
        select_best_spread_recommendation,
    )
except ImportError:
    try:
        from Scripts.rag_ingest.core.line_selection import (
            choose_best_spread_line, confidence_from_edge, _best_model_only_spread,
            select_best_spread_recommendation,
        )
    except ImportError:
        try:
            from rag_cli_v2 import (
                choose_best_spread_line, confidence_from_edge, _best_model_only_spread,
                select_best_spread_recommendation,
            )
        except ImportError:
            def choose_best_spread_line(*a, **kw): return None
            def confidence_from_edge(*a, **kw): return "low"
            def _best_model_only_spread(*a, **kw): return ("Home", "+0.5", 0.50)
            def select_best_spread_recommendation(*a, **kw):
                return {"bet_recommendation": None, "best_value": None, "no_bet_reason": "Spread selector unavailable."}

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


# ---------------------------------------------------------------------------
# Exported function
# ---------------------------------------------------------------------------


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
        spread_result = select_best_spread_recommendation(
            options, proj_diff, home, away_team=away, league=league,
        )
        best = spread_result.get("bet_recommendation")
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
            if best.get("_push_prob"):
                lines.append(f"  Push probability: {best['_push_prob']:.1%}.")
            if ev is not None:
                lines.append(f"  Expected value: {ev:+.3f} per unit staked ({conf} confidence).")
            else:
                lines.append(f"  Model edge {edge:+.2f} ({conf} confidence).")
        else:
            best_value = spread_result.get("best_value")
            if best_value:
                bv_team = best_value["team"]
                bv_point = best_value["point"]
                bv_odds = best_value["odds"]
                lines.append(
                    f"  Best value line: {bv_team} {bv_point:+g} @ {bv_odds:.2f} ({best_value['bookmaker']}, {best_value['market_key']})."
                )
                if best_value.get("_model_prob") is not None and best_value.get("_value_edge") is not None:
                    lines.append(
                        f"  Pricing check: {best_value['_model_prob']:.1%} equivalent cover | Value edge: {best_value['_value_edge']:+.1%} | EV: {best_value.get('_ev', 0.0):+.3f}"
                    )
            lines.append(f"  Verdict: No spread bet — {spread_result.get('no_bet_reason') or 'No spread line clears the betting thresholds.'}")
            sp_team, sp_hc, sp_cp = _best_model_only_spread(proj_diff, home, away, league=league)
            lines.append(
                f"  Model lean: {sp_team} {sp_hc} "
                f"(model-only, P(cover) = {sp_cp:.1%})."
            )

    lines.append("Method: scoreline matrix + Asian handicap settlement model from local KB projections and available spread lines.")
    return "\n".join(lines)
