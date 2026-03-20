"""
rendering/evidence.py — European data source note helper.

Extracted from rag_cli_v2.py without modification.
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Cross-module imports with fallbacks
# ---------------------------------------------------------------------------

try:
    from core.weights import SCORING_WEIGHTS, EUROPEAN_COMPETITIONS
except ImportError:
    try:
        from Scripts.rag_ingest.core.weights import SCORING_WEIGHTS, EUROPEAN_COMPETITIONS
    except ImportError:
        # Fallback: import from monolith
        try:
            from rag_cli_v2 import SCORING_WEIGHTS, EUROPEAN_COMPETITIONS
        except ImportError:
            SCORING_WEIGHTS = {
                "european": {
                    "domestic_weight": 0.80,
                    "euro_weight": 0.20,
                    "min_euro_fixtures": 3,
                    "non_top5_confidence_penalty": 0.15,
                },
            }
            EUROPEAN_COMPETITIONS = {"UCL", "UEL", "UECL"}

try:
    from core.team_resolution import resolve_domestic_league, get_team_profile_meta, _numeric
except ImportError:
    try:
        from Scripts.rag_ingest.core.team_resolution import resolve_domestic_league, get_team_profile_meta, _numeric
    except ImportError:
        try:
            from rag_cli_v2 import resolve_domestic_league, get_team_profile_meta, _numeric
        except ImportError:
            def resolve_domestic_league(team_name: str) -> Optional[str]:
                return None

            def get_team_profile_meta(team_name: str, league: str) -> dict:
                return {}

            def _numeric(value) -> Optional[float]:
                try:
                    return float(value)
                except Exception:
                    return None


# ---------------------------------------------------------------------------
# Exported function
# ---------------------------------------------------------------------------


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
