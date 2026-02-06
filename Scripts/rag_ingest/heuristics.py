"""
Lightweight heuristic scorer for matchup-aware markets.
Inputs are dictionaries (typically Chroma metadata from team_profile docs).
Produces a scored slate of realistic markets with stat-based rationales.
"""

from __future__ import annotations
from typing import Dict, List, Tuple


def _f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _s(x, default="n/a"):
    return x if isinstance(x, str) and x else default


def _pace(meta: Dict) -> float:
    # Use shots_for_pm + shots_against_pm proxy if present, else goals proxies
    sf = _f(meta.get("shots_for_pm"))
    sa = _f(meta.get("shots_against_pm"))
    if sf == 0 and sa == 0:
        return _f(meta.get("goals_for_pm")) * 2  # rough
    return sf + sa


def _card_signal(meta: Dict) -> float:
    """
    Cards per 90 if present; else yellow+red per match proxy.
    """
    c90 = _f(meta.get("cards_per_90_team"))
    if c90 > 0:
        return c90
    # fallback to cards_pm if present in profile docs
    return _f(meta.get("cards_pm"))


def market_scores(home: Dict, away: Dict) -> List[Dict]:
    """
    Returns list of scored candidate markets with rationales and support counts.
    Keys used (if present): goals_for_pm, corners_pm, corners_against_pm,
    control_index, aggression_index_norm, form_index_team,
    cards_per_90_team, fouls_per_90_team, possession, archetype.
    """
    h_g = _f(home.get("goals_for_pm"))
    a_g = _f(away.get("goals_for_pm"))
    h_ctrl = _f(home.get("control_index"))
    a_ctrl = _f(away.get("control_index"))
    h_form = _f(home.get("form_index_team"))
    a_form = _f(away.get("form_index_team"))
    h_corner = _f(home.get("corners_pm"))
    a_corner = _f(away.get("corners_pm"))
    h_c_against = _f(home.get("corners_against_pm"))
    a_c_against = _f(away.get("corners_against_pm"))
    h_agg = _f(home.get("aggression_index_norm"))
    a_agg = _f(away.get("aggression_index_norm"))
    h_cards = _card_signal(home)
    a_cards = _card_signal(away)
    h_fouls = _f(home.get("fouls_per_90_team"))
    a_fouls = _f(away.get("fouls_per_90_team"))
    h_poss = _f(home.get("possession"))
    a_poss = _f(away.get("possession"))

    h_arche = _s(home.get("archetype"))
    a_arche = _s(away.get("archetype"))

    scores: List[Dict] = []

    def support_count(*vals) -> int:
        """How many non-zero/non-None signals back this market."""
        return sum(1 for v in vals if v is not None and _f(v) != 0)

    # Goals / BTTS leaning
    ctrl_diff = h_ctrl - a_ctrl
    form_diff = h_form - a_form

    goals_pace = (h_g + a_g) * 0.35 + (h_ctrl + a_ctrl) * 0.25 + (h_form + a_form) * 0.2 + (_pace(home) + _pace(away)) * 0.2
    over_score = goals_pace + max(ctrl_diff, 0) * 0.1 + max(form_diff, 0) * 0.1
    scores.append({
        "market": "Over 2.5 Goals / BTTS lean",
        "score": over_score,
        "rationale": f"GF_pm {h_g:.2f}/{a_g:.2f}, control {h_ctrl:.2f}/{a_ctrl:.2f}, form {h_form:.2f}/{a_form:.2f}, pace {_pace(home)+_pace(away):.2f}, ctrl_diff {ctrl_diff:.2f}, form_diff {form_diff:.2f}",
        "support": support_count(h_g, a_g, h_ctrl, a_ctrl, h_form, a_form)
    })

    # Low-scoring lean
    under_score = (2 - over_score) + max(-ctrl_diff, 0) * 0.1 + max(-form_diff, 0) * 0.1
    scores.append({
        "market": "Under 2.5 Goals lean",
        "score": under_score,
        "rationale": f"Lower combined attack pace; GF_pm {h_g:.2f}+{a_g:.2f}, control {h_ctrl:.2f}+{a_ctrl:.2f}, ctrl_diff {ctrl_diff:.2f}, form_diff {form_diff:.2f}",
        "support": support_count(h_g, a_g, h_ctrl, a_ctrl, h_form, a_form)
    })

    # Corners advantage (home & away)
    home_corner_edge = h_corner - a_c_against
    away_corner_edge = a_corner - h_c_against
    scores.append({
        "market": "Home corners advantage",
        "score": home_corner_edge,
        "rationale": f"Home corners_pm {h_corner:.2f} vs away concede {a_c_against:.2f}",
        "support": support_count(h_corner, a_c_against)
    })
    scores.append({
        "market": "Away corners advantage",
        "score": away_corner_edge,
        "rationale": f"Away corners_pm {a_corner:.2f} vs home concede {h_c_against:.2f}",
        "support": support_count(a_corner, h_c_against)
    })

    # Cards / fouls
    card_pressure = (h_agg + a_agg) * 0.30 + (h_cards + a_cards) * 0.50 + (h_fouls + a_fouls) * 0.20
    scores.append({
        "market": "Over total cards",
        "score": card_pressure,
        "rationale": f"Agg {h_agg:.2f}+{a_agg:.2f}, cards/90 {h_cards:.2f}+{a_cards:.2f}, fouls/90 {h_fouls:.2f}+{a_fouls:.2f}",
        "support": support_count(h_agg, a_agg, h_cards, a_cards, h_fouls, a_fouls)
    })
    scores.append({
        "market": "Home over cards",
        "score": h_cards + h_agg,
        "rationale": f"Home cards signal {h_cards:.2f}, aggression {h_agg:.2f}",
        "support": support_count(h_cards, h_agg)
    })
    scores.append({
        "market": "Away over cards",
        "score": a_cards + a_agg,
        "rationale": f"Away cards signal {a_cards:.2f}, aggression {a_agg:.2f}",
        "support": support_count(a_cards, a_agg)
    })

    # Archetype fit (style clash/edge)
    arche_note = f"{h_arche} vs {a_arche}"
    if "possession" in h_arche and "transition" in a_arche:
        edge = 1.0
        reason = "Possession side vs transition side; risk of counters + cards."
    elif "aggressive" in h_arche or "aggressive" in a_arche:
        edge = 0.8
        reason = "Aggressive archetype involved; elevates fouls/cards."
    else:
        edge = 0.4
        reason = "Archetype clash moderate."
    scores.append({
        "market": "Style clash insight",
        "score": edge,
        "rationale": f"{arche_note} → {reason}",
        "support": support_count(edge)
    })

    # Small penalty against single H2H recency: none applied directly here;
    # scorer already ignores single fixtures. LLM prompt enforces no single-H2H reasoning.

    # Sort high to low
    scores = sorted(scores, key=lambda x: x["score"], reverse=True)
    return scores


def top_markets(home: Dict, away: Dict, n: int = 3) -> List[Dict]:
    scored = market_scores(home, away)

    # Filter out weakly-supported picks (need at least 2 signals if possible)
    supported = [s for s in scored if s.get("support", 0) >= 2] or scored

    # Ensure variety: goal, cards, corners, then best remaining
    goal_pick = next((s for s in supported if "Goals" in s["market"] or "BTTS" in s["market"]), None)
    cards_pick = next((s for s in supported if "cards" in s["market"].lower()), None)
    corners_pick = next((s for s in supported if "corners" in s["market"].lower()), None)

    picks: List[Dict] = []
    for p in [goal_pick, cards_pick, corners_pick]:
        if p and p not in picks:
            picks.append(p)

    for s in scored:
        if len(picks) >= n:
            break
        if s not in picks:
            picks.append(s)
    return picks[:n]
