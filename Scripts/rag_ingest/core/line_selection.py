"""
core/line_selection.py — Line selection and confidence functions extracted from rag_cli_v2.py.

Functions that take projection outputs + bookmaker odds and pick the best
line/side/score to recommend.  Also includes confidence assignment logic.

Dependencies:
    - core.weights     (SCORING_WEIGHTS)
    - prob_models      (over_prob, under_prob, implied_prob, value_edge,
                         expected_value, remove_vig_two_way)
"""
from __future__ import annotations

from math import erf, sqrt
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Internal imports — try core.* first, fall back to flat for backward compat
# ---------------------------------------------------------------------------
try:
    from core.weights import SCORING_WEIGHTS
except ImportError:
    try:
        from weights import SCORING_WEIGHTS
    except ImportError:
        from rag_cli_v2 import SCORING_WEIGHTS  # type: ignore[import]

try:
    from prob_models import (
        over_prob, under_prob, implied_prob, remove_vig_two_way,
        value_edge, expected_value,
    )
except ImportError:
    from Scripts.rag_ingest.prob_models import (  # type: ignore[import]
        over_prob, under_prob, implied_prob, remove_vig_two_way,
        value_edge, expected_value,
    )


# ===================================================================
#  Small helpers
# ===================================================================

def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


# ===================================================================
#  Totals line selection
# ===================================================================

def choose_best_total_line(options: List[Dict], projection_total: float,
                           combined_var: Optional[float] = None) -> Optional[Dict]:
    if not options:
        return None
    tw = SCORING_WEIGHTS["totals_line"]
    pw = SCORING_WEIGHTS["prob"]

    filtered: List[Dict] = []
    for opt in options:
        odds = safe_float(opt.get("odds"))
        if odds is None or odds < tw["min_odds_hard"]:
            continue
        filtered.append(opt)
    if not filtered:
        return None

    primary = [o for o in filtered if "alternate" not in str(o.get("market_key") or "").lower()]
    options_use = primary if primary else filtered

    # Pre-compute vig-free implied probabilities for over/under pairs
    _pair_map: Dict[Tuple, Dict[str, float]] = {}
    for _o in options_use:
        _bm = str(_o.get("bookmaker") or "")
        _pt = safe_float(_o.get("point"))
        _sd = str(_o.get("side") or "").lower()
        _od = safe_float(_o.get("odds"))
        if _pt is not None and _od is not None:
            _pair_map.setdefault((_bm, _pt), {})[_sd] = _od
    _fair_implied: Dict[Tuple, float] = {}
    for (_bm, _pt), _sides in _pair_map.items():
        _ov = _sides.get("over")
        _un = _sides.get("under")
        if _ov is not None and _un is not None:
            _f_ov, _f_un = remove_vig_two_way(_ov, _un)
            _fair_implied[(_bm, _pt, "over")] = _f_ov
            _fair_implied[(_bm, _pt, "under")] = _f_un

    best = None
    best_score = None
    for opt in options_use:
        side = str(opt.get("side") or "")
        point = safe_float(opt.get("point"))
        odds = safe_float(opt.get("odds"))
        if point is None or odds is None:
            continue
        edge = projection_total - point if side == "over" else point - projection_total

        dist = abs(projection_total - point)
        proximity_penalty = dist * tw["proximity_penalty"]
        edge_term = edge * tw["edge_weight"]

        if odds < tw["min_odds_preferred"]:
            odds_term = -tw["short_price_penalty"] * (tw["min_odds_preferred"] - odds)
        elif tw["ideal_odds_low"] <= odds <= tw["ideal_odds_high"]:
            odds_term = tw["ideal_range_bonus"] * (odds - 1.0)
        else:
            odds_term = tw["outside_ideal_bonus"] * (odds - 1.0)

        far_penalty = 0.0
        if dist >= tw["far_threshold"]:
            far_penalty = (dist - tw["far_threshold"]) * tw["far_penalty"]

        # --- Probability-aware scoring (vig-adjusted) ---
        model_p = (over_prob(projection_total, point, combined_var) if side == "over"
                   else under_prob(projection_total, point, combined_var))
        fair_key = (str(opt.get("bookmaker") or ""), point, side.lower())
        implied_p = _fair_implied.get(fair_key) or implied_prob(odds)
        val_edge = value_edge(model_p, implied_p)
        ev = expected_value(model_p, odds)

        prob_term = val_edge * pw["value_edge_weight"]
        ev_penalty = abs(ev) * pw["negative_ev_penalty"] if ev < 0 else 0.0
        # Reward higher absolute model probability — prevents thin-edge recs
        model_prob_bonus = (model_p - pw["min_model_prob"]) * pw["model_prob_weight"]

        score = (edge_term - proximity_penalty + odds_term - far_penalty
                 + prob_term - ev_penalty + model_prob_bonus)

        # Attach probability data for rendering
        opt["_model_prob"] = model_p
        opt["_implied_prob"] = implied_p
        opt["_value_edge"] = val_edge
        opt["_ev"] = ev

        if best is None or score > float(best_score):
            best = opt
            best_score = score
    return best


# ===================================================================
#  Model-only fallbacks
# ===================================================================

def _best_model_only_line(projection: float,
                          variance: Optional[float] = None) -> Tuple[str, float, float]:
    """Pick the strongest (side, line, model_prob) for a model-only recommendation.

    Evaluates half-integer lines (X.5, no push) within +/-1.0 of the projection
    and returns the side+line combination with the highest model probability.
    """
    best_side, best_line, best_prob = "Over", round(projection * 2) / 2, 0.5

    base = int(projection)
    candidates = [base - 0.5, base + 0.5, base + 1.5]
    if base >= 2:
        candidates.append(base - 1.5)

    for line in candidates:
        if line < 0.5 or abs(line - projection) > 1.0:
            continue
        p_over = over_prob(projection, line, variance)
        p_under = 1.0 - p_over
        side = "Over" if p_over >= p_under else "Under"
        prob = max(p_over, p_under)
        if prob > best_prob:
            best_side, best_line, best_prob = side, line, prob

    return best_side, best_line, best_prob


def _best_model_only_spread(proj_diff: float, home: str, away: str) -> Tuple[str, str, float]:
    """Pick the strongest (team, handicap_display, cover_prob) for model-only spread.

    Evaluates half-integer handicaps within +/-1.0 of |proj_diff|.
    """
    margin_std = SCORING_WEIGHTS["prob"]["margin_std_default"]
    best_team = home if proj_diff >= 0 else away
    best_hc = "-0.5" if proj_diff >= 0 else "+0.5"
    best_prob = 0.5

    for handicap_abs in [0.5, 1.5, 2.5]:
        if abs(handicap_abs - abs(proj_diff)) > 1.0:
            continue
        # Home giving -handicap
        p_home = 1.0 - _normal_cdf((handicap_abs - proj_diff) / margin_std)
        # Away receiving +handicap
        p_away = 1.0 - _normal_cdf((proj_diff - handicap_abs) / margin_std)

        if p_home > best_prob:
            best_team, best_hc, best_prob = home, f"{-handicap_abs:+g}", p_home
        if p_away > best_prob:
            best_team, best_hc, best_prob = away, f"+{handicap_abs:g}", p_away

    return best_team, best_hc, best_prob


# ===================================================================
#  BTTS side selection
# ===================================================================

def choose_best_btts_side(options: List[Dict], p_btts_yes: float) -> Optional[Dict]:
    """Score BTTS Yes/No options and return the best one with probability data attached.

    BTTS is a directional bet — the model's conviction (probability) must drive the
    recommendation, not just value edge. A 72% BTTS Yes probability should almost always
    recommend Yes, regardless of whether No has a marginally better edge. Value edge is
    used as a tiebreaker and to flag when the model-preferred side is overpriced.
    """
    if not options:
        return None
    pw = SCORING_WEIGHTS["prob"]

    # Build vig-free implied probs from Yes/No pairs per bookmaker
    _pair_map: Dict[str, Dict[str, float]] = {}
    for o in options:
        bm = str(o.get("bookmaker") or "")
        side = str(o.get("side") or "").lower().strip()
        odds = safe_float(o.get("odds"))
        if odds is not None and side:
            _pair_map.setdefault(bm, {})[side] = odds
    _fair_implied: Dict[Tuple[str, str], float] = {}
    for bm, sides in _pair_map.items():
        yes_odds = sides.get("yes")
        no_odds = sides.get("no")
        if yes_odds is not None and no_odds is not None:
            f_yes, f_no = remove_vig_two_way(yes_odds, no_odds)
            _fair_implied[(bm, "yes")] = f_yes
            _fair_implied[(bm, "no")] = f_no

    # Step 1: Determine the model's preferred side based on probability
    model_preferred_side = "yes" if p_btts_yes >= 0.50 else "no"
    model_conviction = max(p_btts_yes, 1.0 - p_btts_yes)

    # Step 2: Score all options, but heavily weight model conviction
    best = None
    best_score = None
    for opt in options:
        side = str(opt.get("side") or "").lower().strip()
        odds = safe_float(opt.get("odds"))
        if not side or odds is None or odds <= 1.0:
            continue

        model_p = p_btts_yes if side == "yes" else (1.0 - p_btts_yes)
        bm = str(opt.get("bookmaker") or "")
        implied_p = _fair_implied.get((bm, side)) or implied_prob(odds)
        val_edge = value_edge(model_p, implied_p)
        ev = expected_value(model_p, odds)

        # Score: model probability is the PRIMARY driver (weight 4.0)
        # Value edge is secondary (weight 1.0)
        # Penalize going against model conviction heavily
        conviction_score = (model_p - 0.50) * pw.get("model_prob_weight", 4.0)
        edge_bonus = val_edge * pw.get("value_edge_weight", 2.0) if val_edge > 0 else 0.0
        ev_penalty = abs(ev) * pw["negative_ev_penalty"] if ev < 0 else 0.0

        # Strong penalty for recommending opposite of model conviction
        against_model_penalty = 0.0
        if side != model_preferred_side and model_conviction >= 0.55:
            against_model_penalty = (model_conviction - 0.50) * 6.0

        score = conviction_score + edge_bonus - ev_penalty - against_model_penalty

        opt["_model_prob"] = model_p
        opt["_implied_prob"] = implied_p
        opt["_value_edge"] = val_edge
        opt["_ev"] = ev

        if best is None or score > best_score:
            best = opt
            best_score = score
    return best


# ===================================================================
#  Correct score selection
# ===================================================================

def choose_best_correct_scores(
    model_probs: Dict[Tuple[int, int], float],
    market_odds: Dict[Tuple[int, int], List[Dict]],
    top_n: int = 5,
) -> List[Dict]:
    """Rank correct score bets by model probability + value edge.  Returns up to *top_n*."""
    candidates: List[Dict] = []
    sorted_scores = sorted(model_probs.items(), key=lambda x: x[1], reverse=True)

    for score, model_p in sorted_scores[:15]:
        entry: Dict = {
            "score": score,
            "score_str": f"{score[0]}-{score[1]}",
            "model_prob": model_p,
            "best_odds": None,
            "bookmaker": None,
            "implied_prob": None,
            "value_edge": None,
            "ev": None,
        }
        if score in market_odds and market_odds[score]:
            best = max(market_odds[score], key=lambda x: x["odds"])
            entry["best_odds"] = best["odds"]
            entry["bookmaker"] = best["bookmaker"]
            entry["implied_prob"] = implied_prob(best["odds"])
            entry["value_edge"] = value_edge(model_p, entry["implied_prob"])
            entry["ev"] = expected_value(model_p, best["odds"])
        candidates.append(entry)

    def _rank(c):
        base = c["model_prob"] * 10.0
        if c["value_edge"] is not None:
            base += max(c["value_edge"], 0) * 5.0
        if c["ev"] is not None and c["ev"] > 0:
            base += c["ev"] * 2.0
        return -base

    candidates.sort(key=_rank)
    return candidates[:top_n]


# ===================================================================
#  Moneyline side selection
# ===================================================================

def choose_best_moneyline_side(
    p_home: float, p_draw: float, p_away: float,
    market_odds: List[Dict],
    home: str, away: str,
) -> Dict:
    """Score home/draw/away and pick the best value moneyline bet."""
    model_probs = {"home": p_home, "draw": p_draw, "away": p_away}
    team_labels = {"home": home, "draw": "Draw", "away": away}

    # Group odds by side, pick best (highest) per side
    best_per_side: Dict[str, Dict] = {}
    for entry in market_odds:
        s = entry["side"]
        if s not in best_per_side or entry["odds"] > best_per_side[s]["odds"]:
            best_per_side[s] = entry

    # Remove vig (3-way proportional)
    fair_probs: Dict[str, float] = {}
    if all(s in best_per_side for s in ("home", "draw", "away")):
        raw = {s: 1.0 / best_per_side[s]["odds"] for s in ("home", "draw", "away")}
        total = sum(raw.values())
        fair_probs = {s: raw[s] / total for s in ("home", "draw", "away")}

    candidates: List[Dict] = []
    for side in ("home", "draw", "away"):
        entry: Dict = {
            "side": side,
            "team": team_labels[side],
            "model_prob": model_probs[side],
            "best_odds": None,
            "bookmaker": None,
            "implied_prob": None,
            "fair_implied": None,
            "value_edge": None,
            "ev": None,
        }
        if side in best_per_side:
            bps = best_per_side[side]
            entry["best_odds"] = bps["odds"]
            entry["bookmaker"] = bps["bookmaker"]
            entry["implied_prob"] = implied_prob(bps["odds"])
            if side in fair_probs:
                entry["fair_implied"] = fair_probs[side]
                entry["value_edge"] = model_probs[side] - fair_probs[side]
            else:
                entry["value_edge"] = value_edge(model_probs[side], entry["implied_prob"])
            entry["ev"] = expected_value(model_probs[side], bps["odds"])
        candidates.append(entry)

    # Pick best: model probability is the primary driver, value edge is secondary.
    # A Draw at +2% edge should NOT beat a Home Win at 55% probability.
    # Score = model_prob * 3.0 + value_edge_bonus - negative_ev_penalty
    best = None
    best_score = -999.0
    for c in candidates:
        model_p = c["model_prob"]
        ve = c["value_edge"]
        ev_val = c["ev"]

        # Model conviction is the dominant signal
        conviction = (model_p - 0.33) * 3.0  # 3-way: baseline is 33%, not 50%

        # Value edge bonus (only when positive)
        edge_bonus = (ve * 2.0) if ve is not None and ve > 0 else 0.0

        # Penalty for negative EV
        ev_penalty = abs(ev_val) * 2.0 if ev_val is not None and ev_val < 0 else 0.0

        # Penalty for recommending Draw (draws are inherently harder to predict
        # and rarely the right recommendation unless model is very confident)
        draw_penalty = 0.0
        if c["side"] == "draw" and model_p < 0.32:
            draw_penalty = 0.5

        score = conviction + edge_bonus - ev_penalty - draw_penalty

        if score > best_score:
            best_score = score
            best = c

    return {"recommended": best, "all_sides": candidates}


# ===================================================================
#  Spread / Handicap line selection
# ===================================================================

def choose_best_spread_line(options: List[Dict], projected_diff: float, home_team: str) -> Optional[Dict]:
    """Score available spread lines and return the best one."""
    if not options:
        return None
    sw = SCORING_WEIGHTS["spreads_line"]
    pw = SCORING_WEIGHTS["prob"]
    margin_std = pw["margin_std_default"]

    filtered = [o for o in options if (o.get("odds") or 0) >= sw["min_odds_hard"]]
    if not filtered:
        return None

    primary = [o for o in filtered if "alternate" not in str(o.get("market_key") or "").lower()]
    options_use = primary if primary else filtered

    # Pre-compute vig-free implied probabilities for spread pairs.
    # Each option has (team, signed_point, odds). The matching pair for
    # "TeamA at point X" is "TeamB at point -X" from the same bookmaker.
    # Build a lookup: (bookmaker, team, signed_point) -> odds
    _sp_lookup: Dict[Tuple, float] = {}
    for _o in options_use:
        _bm = str(_o.get("bookmaker") or "")
        _tm = str(_o.get("team") or "").lower().strip()
        _pt = _o.get("point") or 0
        _od = _o.get("odds")
        if _od is not None and _tm:
            _sp_lookup[(_bm, _tm, _pt)] = float(_od)

    # Build fair implied probabilities by pairing each option with its
    # counterpart: same bookmaker, same abs(point), opposite sign, different team.
    # Key: (bookmaker, signed_point, team) -> fair_implied_probability
    _sp_fair_signed: Dict[Tuple, float] = {}
    _sp_done: set = set()
    for _o in options_use:
        _bm = str(_o.get("bookmaker") or "")
        _tm = str(_o.get("team") or "").lower().strip()
        _pt = _o.get("point") or 0
        _od = float(_o.get("odds") or 0)

        key_self = (_bm, _pt, _tm)
        if key_self in _sp_done:
            continue

        opp_entry = None
        for (_bm2, _tm2, _pt2), _od2 in _sp_lookup.items():
            if (_bm2 == _bm
                    and _tm2 != _tm
                    and abs(abs(_pt2) - abs(_pt)) < 0.001
                    and ((_pt < 0 and _pt2 > 0) or (_pt > 0 and _pt2 < 0))):
                opp_entry = (_tm2, _pt2, _od2)
                break

        if opp_entry:
            _tm2, _pt2, _od2 = opp_entry
            _fa, _fb = remove_vig_two_way(_od, _od2)
            _sp_fair_signed[key_self] = _fa
            _sp_fair_signed[(_bm, _pt2, _tm2)] = _fb
            _sp_done.add(key_self)
            _sp_done.add((_bm, _pt2, _tm2))

    # Wrapper to look up fair implied by signed point
    def _get_sp_fair(bookmaker: str, point: float, team: str) -> Optional[float]:
        return _sp_fair_signed.get((bookmaker, point, team.lower().strip()))

    _sp_fair = {}  # legacy compat: not used in lookup below

    best, best_score = None, None
    for opt in options_use:
        team = str(opt.get("team") or "")
        point = opt["point"]
        odds = opt["odds"]

        is_home = team.lower().strip() == home_team.lower().strip()
        sign = 1.0 if is_home else -1.0
        edge = sign * projected_diff + point

        edge_term = edge * sw["edge_weight"]
        dist = abs(point)
        proximity_term = dist * sw["proximity_penalty"]

        if odds < sw["min_odds_preferred"]:
            odds_term = -sw["short_price_penalty"] * (sw["min_odds_preferred"] - odds)
        elif sw["ideal_odds_low"] <= odds <= sw["ideal_odds_high"]:
            odds_term = sw["ideal_range_bonus"] * (odds - 1.0)
        else:
            odds_term = sw["outside_ideal_bonus"] * (odds - 1.0)

        # --- Spread cover probability (normal approximation of goal difference, vig-adjusted) ---
        margin_mean = sign * projected_diff
        cover_threshold = -point
        model_p = 1.0 - _normal_cdf((cover_threshold - margin_mean) / margin_std)
        implied_p = _get_sp_fair(str(opt.get("bookmaker") or ""), point, team) or implied_prob(odds)
        val_edge = value_edge(model_p, implied_p)
        ev = expected_value(model_p, odds)

        prob_term = val_edge * pw["value_edge_weight"]
        ev_penalty = abs(ev) * pw["negative_ev_penalty"] if ev < 0 else 0.0
        # Reward higher absolute model probability — prevents thin-edge recs
        model_prob_bonus = (model_p - pw["min_model_prob"]) * pw["model_prob_weight"]

        score = (edge_term - proximity_term + odds_term + prob_term
                 - ev_penalty + model_prob_bonus)

        opt["_model_prob"] = model_p
        opt["_implied_prob"] = implied_p
        opt["_value_edge"] = val_edge
        opt["_ev"] = ev

        if best is None or score > best_score:
            best = opt
            best_score = score
    return best


# ===================================================================
#  Confidence assignment
# ===================================================================

def confidence_from_edge(edge: float, stat_group: Optional[str] = None,
                         model_prob: Optional[float] = None,
                         value_edge_pct: Optional[float] = None,
                         is_european: bool = False,
                         has_domestic_profile: bool = True) -> str:
    # Probability-calibrated confidence when available
    if model_prob is not None and value_edge_pct is not None:
        if model_prob >= 0.65 and value_edge_pct >= 0.08:
            conf = "high"
        elif model_prob >= 0.55 and value_edge_pct >= 0.03:
            conf = "medium"
        else:
            conf = "low"
    else:
        # Fallback: existing threshold logic
        cw = SCORING_WEIGHTS["confidence"]
        if stat_group == "cards":
            high, medium = cw.get("cards_gap_high", cw["edge_high"]), cw.get("cards_gap_medium", cw["edge_medium"])
        elif stat_group == "corners":
            high, medium = cw.get("corners_gap_high", cw["edge_high"]), cw.get("corners_gap_medium", cw["edge_medium"])
        elif stat_group == "sot":
            high, medium = cw.get("sot_gap_high", cw["edge_high"]), cw.get("sot_gap_medium", cw["edge_medium"])
        else:
            high, medium = cw["edge_high"], cw["edge_medium"]
        if edge >= high:
            conf = "high"
        elif edge >= medium:
            conf = "medium"
        else:
            conf = "low"
    # European competition penalty for non-top-5 teams (no domestic profile)
    if is_european and not has_domestic_profile:
        if conf == "high":
            conf = "medium"
        elif conf == "medium":
            conf = "low"
    return conf
