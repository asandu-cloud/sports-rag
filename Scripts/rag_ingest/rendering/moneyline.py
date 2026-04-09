"""
rendering/moneyline.py — render_moneyline_answer.

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
    from core.projections import projected_moneyline_probs
except ImportError:
    try:
        from Scripts.rag_ingest.core.projections import projected_moneyline_probs
    except ImportError:
        try:
            from rag_cli_v2 import projected_moneyline_probs
        except ImportError:
            def projected_moneyline_probs(*a, **kw): return (None, None, None, None, None)

try:
    from core.odds_extraction import extract_moneyline_odds
except ImportError:
    try:
        from Scripts.rag_ingest.core.odds_extraction import extract_moneyline_odds
    except ImportError:
        try:
            from rag_cli_v2 import extract_moneyline_odds
        except ImportError:
            def extract_moneyline_odds(*a, **kw): return []

try:
    from core.line_selection import choose_best_moneyline_side, confidence_from_edge
except ImportError:
    try:
        from Scripts.rag_ingest.core.line_selection import choose_best_moneyline_side, confidence_from_edge
    except ImportError:
        try:
            from rag_cli_v2 import choose_best_moneyline_side, confidence_from_edge
        except ImportError:
            def choose_best_moneyline_side(*a, **kw): return {"recommended": {}, "all_sides": []}
            def confidence_from_edge(*a, **kw): return "low"

try:
    from core.team_resolution import _profile_meta, _numeric
except ImportError:
    try:
        from Scripts.rag_ingest.core.team_resolution import _profile_meta, _numeric
    except ImportError:
        try:
            from rag_cli_v2 import _profile_meta, _numeric
        except ImportError:
            def _profile_meta(team: str, league: str) -> dict:
                return {}
            def _numeric(value) -> Optional[float]:
                try:
                    return float(value)
                except Exception:
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


# ---------------------------------------------------------------------------
# Exported function
# ---------------------------------------------------------------------------


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
        lines.append(f"  Model: {home} {p_home:.1%} | Draw {p_draw:.1%} | {away} {p_away:.1%}")
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
        rec = result.get("bet_recommendation")
        most_likely = result.get("most_likely") or {"team": home, "model_prob": p_home}
        best_value = result.get("best_value")

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

            lines.append(
                f"  Most likely winner: {most_likely['team']} ({most_likely['model_prob']:.1%})"
            )
            if best_value and best_value.get("best_odds") is not None:
                ve = best_value.get("value_edge")
                ev_val = best_value.get("ev")
                ve_str = f" | Edge: {ve:+.1%}" if ve is not None else ""
                ev_str = f" | EV: {ev_val:+.3f}" if ev_val is not None else ""
                lines.append(
                    f"  Best value side: {best_value['team']} @ {best_value['best_odds']:.2f} "
                    f"({best_value['bookmaker']}){ve_str}{ev_str}"
                )

            if rec:
                conf = confidence_from_edge(
                    abs(rec["value_edge"] or 0), stat_group="goals",
                    model_prob=rec["model_prob"],
                    value_edge_pct=rec["value_edge"],
                )
                lines.append(
                    f"  Verdict: Recommended: {rec['team']} @ {rec['best_odds']:.2f} "
                    f"({conf} confidence)"
                )
            else:
                lines.append("  Verdict: No moneyline bet.")
                reason = result.get("no_bet_reason")
                if reason:
                    lines.append(f"  Reason: {reason}")
        else:
            favored = max(
                [("home", p_home, home), ("draw", p_draw, "Draw"), ("away", p_away, away)],
                key=lambda x: x[1],
            )
            lines.append(f"  Most likely winner: {favored[2]} ({favored[1]:.1%})")
            lines.append("  Verdict: No moneyline odds in feed — model-only lean.")

    lines.append("\nMethod: Poisson/NegBin scoreline matrix from local KB → 3-way outcome probabilities.")
    return "\n".join(lines)
