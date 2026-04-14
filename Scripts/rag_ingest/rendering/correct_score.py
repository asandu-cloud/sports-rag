"""
rendering/correct_score.py — render_correct_score_answer.

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
    from core.projections import projected_correct_score_probs
except ImportError:
    try:
        from Scripts.rag_ingest.core.projections import projected_correct_score_probs
    except ImportError:
        try:
            from rag_cli_v2 import projected_correct_score_probs
        except ImportError:
            def projected_correct_score_probs(*a, **kw): return (None, None, None)

try:
    from core.odds_extraction import extract_correct_score_odds
except ImportError:
    try:
        from Scripts.rag_ingest.core.odds_extraction import extract_correct_score_odds
    except ImportError:
        try:
            from rag_cli_v2 import extract_correct_score_odds
        except ImportError:
            def extract_correct_score_odds(*a, **kw): return {}

try:
    from core.line_selection import choose_best_correct_scores
except ImportError:
    try:
        from Scripts.rag_ingest.core.line_selection import choose_best_correct_scores
    except ImportError:
        try:
            from rag_cli_v2 import choose_best_correct_scores
        except ImportError:
            def choose_best_correct_scores(*a, **kw): return []

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

    lines.append("\nMethod: Dixon-Coles scoreline matrix from local KB projections.")
    return "\n".join(lines)
