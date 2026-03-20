"""
rendering/btts.py — render_btts_answer.

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
    from core.projections import projected_btts_prob
except ImportError:
    try:
        from Scripts.rag_ingest.core.projections import projected_btts_prob
    except ImportError:
        try:
            from rag_cli_v2 import projected_btts_prob
        except ImportError:
            def projected_btts_prob(*a, **kw): return (None, None, None, None, None)

try:
    from core.odds_extraction import extract_btts_odds
except ImportError:
    try:
        from Scripts.rag_ingest.core.odds_extraction import extract_btts_odds
    except ImportError:
        try:
            from rag_cli_v2 import extract_btts_odds
        except ImportError:
            def extract_btts_odds(*a, **kw): return []

try:
    from core.line_selection import choose_best_btts_side, confidence_from_edge
except ImportError:
    try:
        from Scripts.rag_ingest.core.line_selection import choose_best_btts_side, confidence_from_edge
    except ImportError:
        try:
            from rag_cli_v2 import choose_best_btts_side, confidence_from_edge
        except ImportError:
            def choose_best_btts_side(*a, **kw): return None
            def confidence_from_edge(*a, **kw): return "low"

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
