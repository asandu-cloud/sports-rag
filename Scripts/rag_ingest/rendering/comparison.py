"""
rendering/comparison.py — render_comparison_answer and comparison signal helpers.

Extracted from rag_cli_v2.py without modification.
"""

from __future__ import annotations

import re
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
                "comparison_cards": {
                    "cards": 0.55, "fouls": 0.08, "agg": 0.35, "agg_scale": 2.0,
                    "recent_cards": 0.25, "recent_fouls": 0.03, "opp_control": 0.25, "opp_induced": 0.20,
                },
                "comparison_corners": {
                    "proj": 0.50, "season_for": 0.25, "opp_allow": 0.15, "recent_for": 0.15,
                    "control": 0.25, "edge": 0.12, "shots": 0.08,
                },
                "comparison_sot": {
                    "proj": 0.45, "season_for": 0.25, "opp_allow": 0.15, "recent_for": 0.25, "sot_ratio": 0.20,
                },
                "kb_quality": {"cards_matchup_adjust_w": 0.15, "corners_dom_asymmetry_w": 0.20, "corners_style_clash_w": 0.35},
                "confidence": {"corners_gap_high": 0.60, "corners_gap_medium": 0.25,
                               "cards_gap_high": 0.45, "cards_gap_medium": 0.20,
                               "sot_gap_high": 0.50, "sot_gap_medium": 0.20,
                               "edge_high": 0.75, "edge_medium": 0.35},
                "referee": {"modifier_weight": 0.25, "evidence_threshold": 0.02},
            }

try:
    from core.team_resolution import _profile_meta, _recent_stats, safe_float
except ImportError:
    try:
        from Scripts.rag_ingest.core.team_resolution import _profile_meta, _recent_stats, safe_float
    except ImportError:
        try:
            from rag_cli_v2 import _profile_meta, _recent_stats, safe_float
        except ImportError:
            def _profile_meta(team: str, league: str) -> dict:
                return {}
            def _recent_stats(team: str, league: str, last_n: int = 6) -> dict:
                return {}
            def safe_float(x) -> Optional[float]:
                try:
                    return float(x)
                except Exception:
                    return None

try:
    from core.projections import projected_cards, projected_corners, projected_sot
except ImportError:
    try:
        from Scripts.rag_ingest.core.projections import projected_cards, projected_corners, projected_sot
    except ImportError:
        try:
            from rag_cli_v2 import projected_cards, projected_corners, projected_sot
        except ImportError:
            def projected_cards(*a, **kw): return (None, None)
            def projected_corners(*a, **kw): return (None, None)
            def projected_sot(*a, **kw): return (None, None)

try:
    from referee_data import get_referee_modifier
    _REFEREE_AVAILABLE = True
except ImportError:
    try:
        from Scripts.rag_ingest.referee_data import get_referee_modifier
        _REFEREE_AVAILABLE = True
    except ImportError:
        _REFEREE_AVAILABLE = False
        def get_referee_modifier(*args, **kwargs):
            from dataclasses import dataclass
            @dataclass
            class _RM:
                multiplier: float = 1.0
                referee_name: str = None
                sample_size: int = 0
                confidence: float = 0.0
                strictness_ratio: float = 1.0
                avg_cards_per_match: float = 0.0
                avg_fouls_per_match: float = 0.0
                cards_per_foul: float = 0.0
                source: str = "unavailable"
            return _RM()

try:
    from rivalry_context import get_rivalry_context, rivalry_context_evidence_line
except ImportError:
    try:
        from Scripts.rag_ingest.rivalry_context import get_rivalry_context, rivalry_context_evidence_line
    except ImportError:
        def get_rivalry_context(*args, **kwargs): return None
        def rivalry_context_evidence_line(*args, **kwargs): return None


# ---------------------------------------------------------------------------
# Helpers (also from rag_cli_v2.py)
# ---------------------------------------------------------------------------


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


def wants_deep_explanation(user_q: str) -> bool:
    q = user_q.lower()
    return bool(
        re.search(r"\b(why|reason|reasoning|explain|explanation)\b", q)
        or re.search(r"\b(solid|detailed|deep|thorough)\b", q)
    )


# ---------------------------------------------------------------------------
# Comparison signal functions
# ---------------------------------------------------------------------------


def comparison_signals_cards(home: str, away: str, league: str) -> Tuple[Optional[str], List[str], Optional[float], Optional[float]]:
    hm = _profile_meta(home, league)
    am = _profile_meta(away, league)
    hr = _recent_stats(home, league, last_n=6)
    ar = _recent_stats(away, league, last_n=6)

    h_cards = safe_float(hm.get("cards_per_90_team"))
    a_cards = safe_float(am.get("cards_per_90_team"))
    h_fouls = safe_float(hm.get("fouls_per_90_team"))
    a_fouls = safe_float(am.get("fouls_per_90_team"))
    h_agg = safe_float(hm.get("aggression_index_norm"))
    a_agg = safe_float(am.get("aggression_index_norm"))
    h_ctrl = safe_float(hm.get("control_index"))
    a_ctrl = safe_float(am.get("control_index"))
    h_cpf = safe_float(hm.get("cards_per_foul_team"))
    a_cpf = safe_float(am.get("cards_per_foul_team"))
    h_opp_induced = safe_float(hm.get("opp_cards_induced_pm"))
    a_opp_induced = safe_float(am.get("opp_cards_induced_pm"))

    hr_cards = hr.get("cards_avg")
    ar_cards = ar.get("cards_avg")
    hr_fouls = hr.get("fouls_avg")
    ar_fouls = ar.get("fouls_avg")

    # Base deterministic projection from team profile.
    h_proj, a_proj = projected_cards(hm, am)
    if h_proj is None and h_cards is not None:
        h_proj = h_cards
    if a_proj is None and a_cards is not None:
        a_proj = a_cards

    # Matchup adjustment: opponent's card-inducing tendency shifts projections
    w_ma = SCORING_WEIGHTS["kb_quality"]["cards_matchup_adjust_w"]
    if h_opp_induced is not None and a_opp_induced is not None:
        league_avg_induced = (h_opp_induced + a_opp_induced) / 2.0
        if h_proj is not None:
            h_proj *= (1.0 + (a_opp_induced - league_avg_induced) * w_ma)
        if a_proj is not None:
            a_proj *= (1.0 + (h_opp_induced - league_avg_induced) * w_ma)

    # Referee strictness modifier
    ref_mod = get_referee_modifier(
        home, away, league,
        weight=SCORING_WEIGHTS["referee"].get("modifier_weight", 0.25),
    )
    if ref_mod.multiplier != 1.0:
        if h_proj is not None:
            h_proj *= ref_mod.multiplier
        if a_proj is not None:
            a_proj *= ref_mod.multiplier

    # Composite card risk to enrich reasoning (kept deterministic).
    cw = SCORING_WEIGHTS["comparison_cards"]

    def risk(cards, fouls, agg, recent_cards, recent_fouls, opp_control, opp_induced) -> Optional[float]:
        vals = [cards, fouls, agg, recent_cards, recent_fouls, opp_control, opp_induced]
        if all(v is None for v in vals):
            return None
        c = cards if cards is not None else 0.0
        f = fouls if fouls is not None else 0.0
        a = agg if agg is not None else 0.0
        rc = recent_cards if recent_cards is not None else 0.0
        rf = recent_fouls if recent_fouls is not None else 0.0
        oc = opp_control if opp_control is not None else 0.0
        oi = opp_induced if opp_induced is not None else 0.0
        return ((cw["cards"] * c) + (cw["fouls"] * f)
                + (cw["agg"] * a * cw["agg_scale"])
                + (cw["recent_cards"] * rc) + (cw["recent_fouls"] * rf)
                + (cw["opp_control"] * oc) + (cw.get("opp_induced", 0.20) * oi))

    h_risk = risk(h_cards, h_fouls, h_agg, hr_cards, hr_fouls, a_ctrl, a_opp_induced)
    a_risk = risk(a_cards, a_fouls, a_agg, ar_cards, ar_fouls, h_ctrl, h_opp_induced)

    pick = None
    h_score = h_risk if h_risk is not None else h_proj
    a_score = a_risk if a_risk is not None else a_proj
    if h_score is not None and a_score is not None:
        pick = home if h_score >= a_score else away
    elif h_score is not None:
        pick = home
    elif a_score is not None:
        pick = away

    lines: List[str] = []
    if h_proj is not None and a_proj is not None:
        lines.append(f"Profile projection (matchup-adj): {home} {h_proj:.2f} vs {away} {a_proj:.2f} cards.")
    if h_cards is not None and a_cards is not None:
        lines.append(f"Season cards/90: {home} {h_cards:.2f} vs {away} {a_cards:.2f}.")
    if h_fouls is not None and a_fouls is not None:
        lines.append(f"Season fouls/90: {home} {h_fouls:.2f} vs {away} {a_fouls:.2f}.")
    if h_cpf is not None and a_cpf is not None:
        lines.append(f"Cards per foul: {home} {h_cpf:.3f} vs {away} {a_cpf:.3f}.")
    if h_agg is not None and a_agg is not None:
        lines.append(f"Aggression index: {home} {h_agg:.2f} vs {away} {a_agg:.2f}.")
    if h_opp_induced is not None and a_opp_induced is not None:
        lines.append(f"Opp cards induced/match: {home} {h_opp_induced:.2f} vs {away} {a_opp_induced:.2f}.")
    if hr_cards is not None and ar_cards is not None:
        lines.append(f"Recent 6-match cards avg: {home} {hr_cards:.2f} vs {away} {ar_cards:.2f}.")
    if h_risk is not None and a_risk is not None:
        lines.append(f"Composite card risk: {home} {h_risk:.2f} vs {away} {a_risk:.2f}.")
    if ref_mod.source == "profile" and ref_mod.referee_name:
        rw_ref = SCORING_WEIGHTS["referee"]
        if abs(ref_mod.multiplier - 1.0) > rw_ref["evidence_threshold"]:
            direction_word = "strict" if ref_mod.multiplier > 1.0 else "lenient"
            lines.append(
                f"Referee: {ref_mod.referee_name.title()} ({direction_word}, "
                f"{ref_mod.strictness_ratio:.2f}x league avg, {ref_mod.sample_size} matches)."
            )
    return pick, lines[:8], h_score, a_score


def comparison_signals_corners(home: str, away: str, league: str) -> Tuple[Optional[str], List[str], Optional[float], Optional[float]]:
    hm = _profile_meta(home, league)
    am = _profile_meta(away, league)
    hr = _recent_stats(home, league, last_n=6)
    ar = _recent_stats(away, league, last_n=6)

    h_for = safe_float(hm.get("corners_pm"))
    a_for = safe_float(am.get("corners_pm"))
    h_against = safe_float(hm.get("corners_against_pm"))
    a_against = safe_float(am.get("corners_against_pm"))
    h_edge = safe_float(hm.get("corner_edge_pm"))
    a_edge = safe_float(am.get("corner_edge_pm"))
    h_ctrl = safe_float(hm.get("control_index"))
    a_ctrl = safe_float(am.get("control_index"))
    h_poss = safe_float(hm.get("possession"))
    a_poss = safe_float(am.get("possession"))
    h_dom = safe_float(hm.get("dominance_index"))
    a_dom = safe_float(am.get("dominance_index"))
    h_shots = safe_float(hm.get("shots_for_pm"))
    a_shots = safe_float(am.get("shots_for_pm"))
    h_home_corners = safe_float(hm.get("corners_home_pm"))
    a_away_corners = safe_float(am.get("corners_away_pm"))

    hr_for = hr.get("corners_for_avg")
    ar_for = ar.get("corners_for_avg")
    hr_against = hr.get("corners_against_avg")
    ar_against = ar.get("corners_against_avg")

    # projected_corners already uses venue-specific blend
    h_proj, a_proj = projected_corners(hm, am)

    ccw = SCORING_WEIGHTS["comparison_corners"]
    kbw = SCORING_WEIGHTS["kb_quality"]

    def corner_score(proj, season_for, opp_allow, recent_for, control, edge, shots) -> Optional[float]:
        vals = [proj, season_for, opp_allow, recent_for, control, edge]
        if all(v is None for v in vals):
            return None
        p = proj if proj is not None else 0.0
        sf = season_for if season_for is not None else 0.0
        oa = opp_allow if opp_allow is not None else 0.0
        rf = recent_for if recent_for is not None else 0.0
        c = control if control is not None else 0.0
        e = edge if edge is not None else 0.0
        s = ccw.get("shots", 0.08) * (shots if shots is not None else 0.0)
        return ccw["proj"] * p + ccw["season_for"] * sf + ccw["opp_allow"] * oa + ccw["recent_for"] * rf + ccw["control"] * c + ccw["edge"] * e + s

    h_score = corner_score(h_proj, h_for, a_against, hr_for, h_ctrl, h_edge, h_shots)
    a_score = corner_score(a_proj, a_for, h_against, ar_for, a_ctrl, a_edge, a_shots)

    # Style-clash bonus: large possession gap with high dominance gap -> both scores rise
    style_clash_bonus = 0.0
    if h_poss is not None and a_poss is not None:
        poss_gap = abs(h_poss - a_poss)
        if h_dom is not None and a_dom is not None:
            dom_gap = abs(h_dom - a_dom)
            style_clash_bonus = poss_gap * dom_gap * kbw["corners_dom_asymmetry_w"]
        if h_shots is not None and a_shots is not None:
            dominant_shots = max(h_shots, a_shots)
            shot_excess = max(0.0, dominant_shots - 12.0)
            style_clash_bonus += poss_gap * shot_excess * kbw["corners_style_clash_w"]
    if h_score is not None:
        h_score += style_clash_bonus / 2.0
    if a_score is not None:
        a_score += style_clash_bonus / 2.0

    pick = None
    if h_score is not None and a_score is not None:
        pick = home if h_score >= a_score else away
    elif h_score is not None:
        pick = home
    elif a_score is not None:
        pick = away

    lines: List[str] = []
    if h_proj is not None and a_proj is not None:
        lines.append(f"Projected corners (venue-adj): {home} {h_proj:.2f} vs {away} {a_proj:.2f}.")
    if h_home_corners is not None and a_away_corners is not None:
        lines.append(f"Venue-specific: {home} home {h_home_corners:.2f} vs {away} away {a_away_corners:.2f}.")
    if h_for is not None and a_for is not None:
        lines.append(f"Season corners for/match: {home} {h_for:.2f} vs {away} {a_for:.2f}.")
    if h_against is not None and a_against is not None:
        lines.append(f"Season corners against/match: {home} {h_against:.2f} vs {away} {a_against:.2f}.")
    if h_shots is not None and a_shots is not None:
        lines.append(f"Shots for/match: {home} {h_shots:.1f} vs {away} {a_shots:.1f}.")
    if h_poss is not None and a_poss is not None:
        lines.append(f"Possession: {home} {h_poss:.1%} vs {away} {a_poss:.1%}.")
    if hr_for is not None and ar_for is not None:
        lines.append(f"Recent 6-match corners for: {home} {hr_for:.2f} vs {away} {ar_for:.2f}.")
    if h_edge is not None and a_edge is not None:
        lines.append(f"Corner edge: {home} {h_edge:+.2f} vs {away} {a_edge:+.2f}.")
    if style_clash_bonus > 0.05:
        lines.append(f"Style-clash bonus: +{style_clash_bonus:.2f} (possession/dominance gap amplifier).")
    if h_score is not None and a_score is not None:
        lines.append(f"Composite corner pressure score: {home} {h_score:.2f} vs {away} {a_score:.2f}.")
    return pick, lines[:8], h_score, a_score


def comparison_signals_sot(home: str, away: str, league: str) -> Tuple[Optional[str], List[str], Optional[float], Optional[float]]:
    hm = _profile_meta(home, league)
    am = _profile_meta(away, league)
    hr = _recent_stats(home, league, last_n=6)
    ar = _recent_stats(away, league, last_n=6)

    h_for = safe_float(hm.get("sot_for_pm"))
    a_for = safe_float(am.get("sot_for_pm"))
    h_against = safe_float(hm.get("sot_against_pm"))
    a_against = safe_float(am.get("sot_against_pm"))
    h_ratio = safe_float(hm.get("_sot_ratio_for"))
    a_ratio = safe_float(am.get("_sot_ratio_for"))

    hr_for = hr.get("sot_for_avg")
    ar_for = ar.get("sot_for_avg")

    h_proj, a_proj = projected_sot(hm, am)

    sw = SCORING_WEIGHTS["comparison_sot"]

    def sot_score(proj, season_for, opp_allow, recent_for, ratio) -> Optional[float]:
        vals = [proj, season_for, opp_allow, recent_for, ratio]
        if all(v is None for v in vals):
            return None
        p = proj if proj is not None else 0.0
        sf = season_for if season_for is not None else 0.0
        oa = opp_allow if opp_allow is not None else 0.0
        rf = recent_for if recent_for is not None else 0.0
        r = ratio if ratio is not None else 0.0
        return sw["proj"] * p + sw["season_for"] * sf + sw["opp_allow"] * oa + sw["recent_for"] * rf + sw["sot_ratio"] * r

    h_score = sot_score(h_proj, h_for, a_against, hr_for, h_ratio)
    a_score = sot_score(a_proj, a_for, h_against, ar_for, a_ratio)

    pick = None
    if h_score is not None and a_score is not None:
        pick = home if h_score >= a_score else away
    elif h_score is not None:
        pick = home
    elif a_score is not None:
        pick = away

    lines: List[str] = []
    if h_proj is not None and a_proj is not None:
        lines.append(f"Projected SoT: {home} {h_proj:.2f} vs {away} {a_proj:.2f}.")
    if h_for is not None and a_for is not None:
        lines.append(f"Season SoT for/match: {home} {h_for:.2f} vs {away} {a_for:.2f}.")
    if h_against is not None and a_against is not None:
        lines.append(f"Season SoT against/match: {home} {h_against:.2f} vs {away} {a_against:.2f}.")
    if hr_for is not None and ar_for is not None:
        lines.append(f"Recent 6-match SoT for: {home} {hr_for:.2f} vs {away} {ar_for:.2f}.")
    if h_ratio is not None and a_ratio is not None:
        lines.append(f"SoT accuracy ratio: {home} {h_ratio:.2f} vs {away} {a_ratio:.2f}.")
    if h_score is not None and a_score is not None:
        lines.append(f"Composite SoT pressure score: {home} {h_score:.2f} vs {away} {a_score:.2f}.")
    return pick, lines[:6], h_score, a_score


# ---------------------------------------------------------------------------
# Main comparison render
# ---------------------------------------------------------------------------


def render_comparison_answer(user_q: str, league: str, events: List[Dict]) -> str:
    group = requested_stat_group(user_q)
    if group not in {"corners", "cards", "sot"}:
        return (
            "I can compare corners/cards/shots-on-target for fixtures, but I couldn't detect which stat you want. "
            "Try: 'for all tomorrow EPL games, which team gets more corners?'"
        )

    detailed = wants_deep_explanation(user_q)
    lines: List[str] = []
    title = {"corners": "Corners", "cards": "Cards/Bookings", "sot": "Shots on Target"}[group]
    lines.append(f"{title} comparison by fixture:")

    for ev in events:
        home = str(ev.get("home_team") or "Home")
        away = str(ev.get("away_team") or "Away")
        rivalry_line = None
        try:
            rivalry_line = rivalry_context_evidence_line(get_rivalry_context(home, away, league))
        except Exception:
            rivalry_line = None

        if group == "corners":
            pick, reasons, h_score, a_score = comparison_signals_corners(home, away, league)
            stat_label = "corners"
            gap_high_key, gap_med_key = "corners_gap_high", "corners_gap_medium"
        elif group == "sot":
            pick, reasons, h_score, a_score = comparison_signals_sot(home, away, league)
            stat_label = "shots on target"
            gap_high_key, gap_med_key = "sot_gap_high", "sot_gap_medium"
        else:
            pick, reasons, h_score, a_score = comparison_signals_cards(home, away, league)
            stat_label = "cards"
            gap_high_key, gap_med_key = "cards_gap_high", "cards_gap_medium"

        if pick is None:
            lines.append(f"- {home} vs {away}: insufficient profile stats for {stat_label}.")
            continue
        lines.append(f"- {home} vs {away}: {pick} likely more {stat_label}.")
        if detailed:
            if rivalry_line:
                lines.append(f"  Reasoning: {rivalry_line}")
            for r in reasons[:5]:
                lines.append(f"  Reasoning: {r}")
            if h_score is not None and a_score is not None:
                margin = abs(h_score - a_score)
                cfw = SCORING_WEIGHTS["confidence"]
                conf = "high" if margin >= cfw[gap_high_key] else ("medium" if margin >= cfw[gap_med_key] else "low")
                lines.append(f"  Reasoning: Confidence={conf} (score gap {margin:.2f}).")
        else:
            if rivalry_line:
                lines.append(f"  Evidence: {rivalry_line}")
        if not detailed and reasons:
            lines.append(f"  Evidence: {reasons[0]}")
            if len(reasons) > 1:
                lines.append(f"  Evidence: {reasons[1]}")

    method = "Method: deterministic multi-signal projection from local KB (season + recent fixtures), not bookmaker lines."
    lines.append(method if detailed else "Method: profile-based projection from your local KB (not bookmaker line prices).")
    return "\n".join(lines)
