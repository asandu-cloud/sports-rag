"""
Lightweight heuristic scorer for matchup-aware markets.
Inputs are dictionaries (typically Chroma metadata from team_profile docs).
We avoid heavy dependencies and keep numbers interpretable for LLM prompts.
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


def market_scores(home: Dict, away: Dict) -> List[Dict]:
    """
    Returns list of scored candidate markets with rationales.
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
    h_cards = _f(home.get("cards_per_90_team"))
    a_cards = _f(away.get("cards_per_90_team"))
    h_fouls = _f(home.get("fouls_per_90_team"))
    a_fouls = _f(away.get("fouls_per_90_team"))
    h_poss = _f(home.get("possession"))
    a_poss = _f(away.get("possession"))

    h_arche = _s(home.get("archetype"))
    a_arche = _s(away.get("archetype"))

    scores: List[Dict] = []

    # Goals / BTTS leaning
    goals_pace = (h_g + a_g) * 0.6 + (h_ctrl + a_ctrl) * 0.4
    goals_form = (h_form + a_form) / 2
    over_score = goals_pace + goals_form
    scores.append({
        "market": "Over 2.5 Goals / BTTS lean",
        "score": over_score,
        "rationale": f"GF_pm {h_g:.2f}/{a_g:.2f}, control {h_ctrl:.2f}/{a_ctrl:.2f}, form {h_form:.2f}/{a_form:.2f}"
    })

    # Low-scoring lean
    under_score = 2 - over_score  # invert
    scores.append({
        "market": "Under 2.5 Goals lean",
        "score": under_score,
        "rationale": f"Lower combined attack pace; GF_pm {h_g:.2f}+{a_g:.2f}, control {h_ctrl:.2f}+{a_ctrl:.2f}"
    })

    # Corners advantage (home)
    home_corner_edge = h_corner - a_c_against
    scores.append({
        "market": "Home corners advantage",
        "score": home_corner_edge,
        "rationale": f"Home corners_pm {h_corner:.2f} vs away concede {a_c_against:.2f}"
    })

    # Cards / fouls
    card_pressure = (h_agg + a_agg) * 0.4 + (h_cards + a_cards) * 0.4 + (h_fouls + a_fouls) * 0.2
    scores.append({
        "market": "Over total cards",
        "score": card_pressure,
        "rationale": f"Agg {h_agg:.2f}+{a_agg:.2f}, cards/90 {h_cards:.2f}+{a_cards:.2f}, fouls/90 {h_fouls:.2f}+{a_fouls:.2f}"
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
        "rationale": f"{arche_note} → {reason}"
    })

    # Sort high to low
    scores = sorted(scores, key=lambda x: x["score"], reverse=True)
    return scores


def top_markets(home: Dict, away: Dict, n: int = 3) -> List[Dict]:
    return market_scores(home, away)[:n]
