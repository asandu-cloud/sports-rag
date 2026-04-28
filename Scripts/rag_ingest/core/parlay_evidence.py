"""
Shared deterministic evidence helpers for parlay selection and rendering.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from .parlay import market_group_from_key, heuristic_groups_for_event
except ImportError:
    try:
        from core.parlay import market_group_from_key, heuristic_groups_for_event
    except ImportError:
        try:
            from Scripts.rag_ingest.core.parlay import market_group_from_key, heuristic_groups_for_event
        except ImportError:
            from rag_cli_v2 import market_group_from_key, heuristic_groups_for_event  # type: ignore[import]

try:
    from .projections import (
        projected_btts_prob,
        projected_cards,
        projected_corners,
        projected_goal_difference,
        projected_moneyline_probs,
        projected_total_cards,
        projected_total_cards_detail,
        projected_total_corners,
        projected_total_goals,
        projected_total_sot,
        projected_team_corners,
        projected_team_cards,
        projected_team_goals,
        compute_margin_std,
    )
except ImportError:
    try:
        from core.projections import (
            projected_btts_prob,
            projected_cards,
            projected_corners,
            projected_goal_difference,
            projected_moneyline_probs,
            projected_total_cards,
            projected_total_corners,
            projected_total_goals,
            projected_total_sot,
            projected_team_corners,
            projected_team_cards,
            projected_team_goals,
            compute_margin_std,
        )
    except ImportError:
        try:
            from Scripts.rag_ingest.core.projections import (
                projected_btts_prob,
                projected_cards,
                projected_corners,
                projected_goal_difference,
                projected_moneyline_probs,
                projected_total_cards,
                projected_total_corners,
                projected_total_goals,
                projected_total_sot,
                projected_team_corners,
                projected_team_cards,
                projected_team_goals,
                compute_margin_std,
            )
        except ImportError:
            from rag_cli_v2 import (  # type: ignore[import]
                projected_btts_prob,
                projected_cards,
                projected_corners,
                projected_goal_difference,
                projected_moneyline_probs,
                projected_total_cards,
                projected_total_corners,
                projected_total_goals,
                projected_total_sot,
                projected_team_corners,
                projected_team_cards,
                projected_team_goals,
            )
            compute_margin_std = None  # type: ignore[assignment]

try:
    from .team_resolution import _profile_meta, _recent_stats, _numeric, get_team_recent_variance
except ImportError:
    try:
        from core.team_resolution import _profile_meta, _recent_stats, _numeric, get_team_recent_variance
    except ImportError:
        try:
            from Scripts.rag_ingest.core.team_resolution import _profile_meta, _recent_stats, _numeric, get_team_recent_variance
        except ImportError:
            from rag_cli_v2 import _profile_meta, _recent_stats, _numeric, get_team_recent_variance  # type: ignore[import]

try:
    from .line_selection import confidence_from_edge
except ImportError:
    try:
        from core.line_selection import confidence_from_edge
    except ImportError:
        try:
            from Scripts.rag_ingest.core.line_selection import confidence_from_edge
        except ImportError:
            from rag_cli_v2 import confidence_from_edge  # type: ignore[import]

try:
    from .weights import SCORING_WEIGHTS
except ImportError:
    try:
        from core.weights import SCORING_WEIGHTS
    except ImportError:
        try:
            from Scripts.rag_ingest.core.weights import SCORING_WEIGHTS
        except ImportError:
            from rag_cli_v2 import SCORING_WEIGHTS  # type: ignore[import]

try:
    from prob_models import implied_prob, over_prob, under_prob, value_edge
except ImportError:
    from Scripts.rag_ingest.prob_models import implied_prob, over_prob, under_prob, value_edge  # type: ignore[import]

try:
    from ml_edge import get_elo_edge, ml_predict_total
    _HAS_ML = True
except ImportError:
    try:
        from Scripts.rag_ingest.ml_edge import get_elo_edge, ml_predict_total
        _HAS_ML = True
    except ImportError:
        _HAS_ML = False

        def get_elo_edge(*args, **kwargs):  # type: ignore[override]
            return None

        def ml_predict_total(*args, **kwargs):  # type: ignore[override]
            return None


def _is_team_total_market(market_key: str) -> bool:
    """True for per-team total markets (team_totals_home_corners, etc.)."""
    return "team_total" in (market_key or "").lower()


def _team_total_side(leg) -> str:
    """Return 'home' or 'away' for a team-total leg based on its market_key."""
    k = (getattr(leg, "market_key", "") or "").lower()
    if "home" in k:
        return "home"
    return "away"


def leg_standalone_confidence(leg: Any, league: str) -> Dict[str, Any]:
    """Return deterministic confidence and warning data for a parlay leg."""
    g = market_group_from_key(getattr(leg, "market_key", ""))
    result: Dict[str, Any] = {
        "confidence": None,
        "model_prob": None,
        "projected": None,
        "side_agrees": True,
        "standalone_side": None,
        "warning": None,
    }

    try:
        if g in {"corners", "totals", "cards", "sot"} and getattr(leg, "point", None) is not None:
            mk = getattr(leg, "market_key", "") or ""
            home = getattr(leg, "home_team", "")
            away = getattr(leg, "away_team", "")
            if _is_team_total_market(mk) and g in ("corners", "cards", "totals"):
                side = _team_total_side(leg)
                is_home = side == "home"
                team = home if is_home else away
                opp = away if is_home else home
                if g == "corners":
                    proj_result = projected_team_corners(team, opp, league, is_home)
                elif g == "cards":
                    proj_result = projected_team_cards(team, opp, league, is_home)
                else:
                    proj_result = projected_team_goals(team, opp, league, is_home)
            else:
                proj_result = {
                    "corners": projected_total_corners(home, away, league),
                    "totals": projected_total_goals(home, away, league),
                    "cards": projected_total_cards(home, away, league),
                    "sot": projected_total_sot(home, away, league),
                }[g]
            proj_total = proj_result[0] if proj_result else None
            if proj_total is not None:
                direction = str(getattr(leg, "outcome", "")).lower()
                is_over = "over" in direction
                variance_key = {
                    "corners": "corners_for_var",
                    "totals": "goals_var",
                    "cards": "cards_var",
                    "sot": "sot_for_var",
                }[g]
                h_var = get_team_recent_variance(getattr(leg, "home_team", ""), league)
                a_var = get_team_recent_variance(getattr(leg, "away_team", ""), league)
                hv = h_var.get(variance_key)
                av = a_var.get(variance_key)
                combined_var = (hv + av) if (hv is not None and av is not None) else None
                model_p = over_prob(proj_total, getattr(leg, "point"), combined_var) if is_over else under_prob(proj_total, getattr(leg, "point"), combined_var)
                implied_p_val = implied_prob(float(getattr(leg, "odds") or 0.0))
                edge = value_edge(model_p, implied_p_val)
                standalone_side = "Over" if proj_total > float(getattr(leg, "point")) else "Under"
                stat_group = {"totals": "goals"}.get(g, g)
                result.update(
                    {
                        "projected": proj_total,
                        "model_prob": model_p,
                        "standalone_side": standalone_side,
                        "side_agrees": (is_over and standalone_side == "Over") or (not is_over and standalone_side == "Under"),
                        "confidence": confidence_from_edge(abs(edge), stat_group=stat_group, model_prob=model_p, value_edge_pct=edge),
                    }
                )
        elif g == "btts":
            btts_result = projected_btts_prob(getattr(leg, "home_team", ""), getattr(leg, "away_team", ""), league)
            p_btts = btts_result[0] if btts_result else None
            if p_btts is not None:
                direction = str(getattr(leg, "outcome", "")).lower().strip()
                model_p = p_btts if direction == "yes" else (1.0 - p_btts)
                implied_p_val = implied_prob(float(getattr(leg, "odds") or 0.0))
                edge = value_edge(model_p, implied_p_val)
                standalone_side = "Yes" if p_btts >= 0.50 else "No"
                result.update(
                    {
                        "projected": p_btts,
                        "model_prob": model_p,
                        "standalone_side": standalone_side,
                        "side_agrees": direction == standalone_side.lower(),
                        "confidence": confidence_from_edge(abs(edge), stat_group="goals", model_prob=model_p, value_edge_pct=edge),
                    }
                )
                if not result["side_agrees"] and abs(p_btts - 0.50) > 0.10:
                    result["warning"] = f"Model favors BTTS {standalone_side} ({p_btts:.0%})"
        elif g == "moneyline":
            ml_result = projected_moneyline_probs(getattr(leg, "home_team", ""), getattr(leg, "away_team", ""), league)
            if ml_result and ml_result[0] is not None:
                p_home, p_draw, p_away = ml_result[0], ml_result[1], ml_result[2]
                direction = str(getattr(leg, "outcome", "")).lower().strip()
                home_lower = str(getattr(leg, "home_team", "")).lower()
                if direction.startswith(home_lower) or "home" in direction:
                    model_p = p_home
                elif "draw" in direction:
                    model_p = p_draw
                else:
                    model_p = p_away
                implied_p_val = implied_prob(float(getattr(leg, "odds") or 0.0))
                edge = value_edge(model_p, implied_p_val)
                best_side = max([("Home", p_home), ("Draw", p_draw), ("Away", p_away)], key=lambda x: x[1])
                result.update(
                    {
                        "model_prob": model_p,
                        "standalone_side": best_side[0],
                        "confidence": confidence_from_edge(abs(edge), stat_group="goals", model_prob=model_p, value_edge_pct=edge),
                    }
                )
        elif g == "spreads" and getattr(leg, "point", None) is not None:
            _home = getattr(leg, "home_team", "")
            _away = getattr(leg, "away_team", "")
            proj_diff, _, _ = projected_goal_difference(_home, _away, league)
            if proj_diff is not None:
                from math import erf, sqrt

                if compute_margin_std is not None and _home and _away and league:
                    margin_std = compute_margin_std(_home, _away, league)
                else:
                    margin_std = SCORING_WEIGHTS["prob"].get("margin_std_default", 1.25)
                cover_z = (proj_diff + float(getattr(leg, "point"))) / margin_std
                model_p = 0.5 * (1.0 + erf(cover_z / sqrt(2.0)))
                if str(getattr(leg, "outcome", "")).lower().startswith(str(getattr(leg, "away_team", "")).lower()):
                    model_p = 1.0 - model_p
                implied_p_val = implied_prob(float(getattr(leg, "odds") or 0.0))
                edge = value_edge(model_p, implied_p_val)
                result.update(
                    {
                        "projected": proj_diff,
                        "model_prob": model_p,
                        "confidence": confidence_from_edge(abs(edge), stat_group="goals", model_prob=model_p, value_edge_pct=edge),
                    }
                )
    except Exception:
        return result

    return result


def leg_evidence(leg: Any, league: str) -> List[str]:
    """Return deterministic evidence lines for a leg."""
    hm = _profile_meta(getattr(leg, "home_team", ""), league)
    am = _profile_meta(getattr(leg, "away_team", ""), league)
    hr = _recent_stats(getattr(leg, "home_team", ""), league, last_n=6)
    ar = _recent_stats(getattr(leg, "away_team", ""), league, last_n=6)
    heur_groups = heuristic_groups_for_event(hm, am)
    lines: List[str] = []

    def v(meta: Dict[str, Any], key: str) -> Optional[float]:
        return _numeric(meta.get(key))

    g = market_group_from_key(getattr(leg, "market_key", ""))
    if g == "corners":
        mk = getattr(leg, "market_key", "") or ""
        h_for, h_against = v(hm, "corners_pm"), v(hm, "corners_against_pm")
        a_for, a_against = v(am, "corners_pm"), v(am, "corners_against_pm")
        h_proj, a_proj = projected_corners(hm, am)
        if _is_team_total_market(mk):
            # Per-team corner evidence
            side = _team_total_side(leg)
            is_home = side == "home"
            team = getattr(leg, "home_team", "") if is_home else getattr(leg, "away_team", "")
            opp = getattr(leg, "away_team", "") if is_home else getattr(leg, "home_team", "")
            tc_proj, _, _ = projected_team_corners(team, opp, league, is_home)
            if tc_proj is not None:
                lines.append(f"Projected {team} corners: {tc_proj:.2f}.")
                if getattr(leg, "point", None) is not None:
                    direction = str(getattr(leg, "outcome", "")).lower()
                    if "over" in direction:
                        lines.append(f"Line fit: projected {tc_proj:.2f} vs over {getattr(leg, 'point'):g}.")
                    elif "under" in direction:
                        lines.append(f"Line fit: projected {tc_proj:.2f} vs under {getattr(leg, 'point'):g}.")
        else:
            if h_for is not None and a_for is not None:
                lines.append(f"Season corners for: {getattr(leg, 'home_team', '')} {h_for:.2f}, {getattr(leg, 'away_team', '')} {a_for:.2f}.")
            if h_proj is not None and a_proj is not None:
                total_proj = h_proj + a_proj
                lines.append(f"Projected combined corners: {total_proj:.2f}.")
                if getattr(leg, "point", None) is not None:
                    direction = str(getattr(leg, "outcome", "")).lower()
                    if "over" in direction:
                        lines.append(f"Line fit: projected {total_proj:.2f} vs over {getattr(leg, 'point'):g}.")
                    elif "under" in direction:
                        lines.append(f"Line fit: projected {total_proj:.2f} vs under {getattr(leg, 'point'):g}.")
        h_recent = hr.get("corners_for_avg")
        a_recent = ar.get("corners_for_avg")
        if h_recent is not None and a_recent is not None:
            lines.append(f"Recent 6-match corners for: {getattr(leg, 'home_team', '')} {h_recent:.2f}, {getattr(leg, 'away_team', '')} {a_recent:.2f}.")
        if h_against is not None and a_against is not None:
            lines.append(f"Corners against: {getattr(leg, 'home_team', '')} {h_against:.2f}, {getattr(leg, 'away_team', '')} {a_against:.2f}.")
    elif g == "totals":
        mk = getattr(leg, "market_key", "") or ""
        if _is_team_total_market(mk):
            side = _team_total_side(leg)
            is_home = side == "home"
            team = getattr(leg, "home_team", "") if is_home else getattr(leg, "away_team", "")
            opp = getattr(leg, "away_team", "") if is_home else getattr(leg, "home_team", "")
            tg_proj, _, _ = projected_team_goals(team, opp, league, is_home)
            if tg_proj is not None:
                lines.append(f"Projected {team} goals: {tg_proj:.2f}.")
                if getattr(leg, "point", None) is not None:
                    direction = str(getattr(leg, "outcome", "")).lower()
                    if "over" in direction:
                        lines.append(f"Line fit: projected {tg_proj:.2f} vs over {getattr(leg, 'point'):g}.")
                    elif "under" in direction:
                        lines.append(f"Line fit: projected {tg_proj:.2f} vs under {getattr(leg, 'point'):g}.")
        else:
            goals_proj, _, _ = projected_total_goals(getattr(leg, "home_team", ""), getattr(leg, "away_team", ""), league)
            if goals_proj is not None:
                lines.append(f"Projected combined goals: {goals_proj:.2f}.")
                if getattr(leg, "point", None) is not None:
                    direction = str(getattr(leg, "outcome", "")).lower()
                    if "over" in direction:
                        lines.append(f"Line fit: projected {goals_proj:.2f} vs over {getattr(leg, 'point'):g}.")
                    elif "under" in direction:
                        lines.append(f"Line fit: projected {goals_proj:.2f} vs under {getattr(leg, 'point'):g}.")
            h_gf, a_gf = v(hm, "goals_for_pm"), v(am, "goals_for_pm")
            if h_gf is not None and a_gf is not None:
                lines.append(f"Season goal baseline: {getattr(leg, 'home_team', '')} {h_gf:.2f}, {getattr(leg, 'away_team', '')} {a_gf:.2f}.")
        h_sot, a_sot = hr.get("sot_for_avg"), ar.get("sot_for_avg")
        if h_sot is not None and a_sot is not None:
            lines.append(f"Recent 6-match SoT for: {getattr(leg, 'home_team', '')} {h_sot:.2f}, {getattr(leg, 'away_team', '')} {a_sot:.2f}.")
    elif g == "sot":
        sot_total, _, _ = projected_total_sot(getattr(leg, "home_team", ""), getattr(leg, "away_team", ""), league)
        if sot_total is not None:
            lines.append(f"Projected combined SoT: {sot_total:.2f}.")
            if getattr(leg, "point", None) is not None:
                direction = str(getattr(leg, "outcome", "")).lower()
                if "over" in direction:
                    lines.append(f"Line fit: projected {sot_total:.2f} vs over {getattr(leg, 'point'):g}.")
                elif "under" in direction:
                    lines.append(f"Line fit: projected {sot_total:.2f} vs under {getattr(leg, 'point'):g}.")
        h_recent_sot, a_recent_sot = hr.get("sot_for_avg"), ar.get("sot_for_avg")
        if h_recent_sot is not None and a_recent_sot is not None:
            lines.append(f"Recent 6-match SoT: {getattr(leg, 'home_team', '')} {h_recent_sot:.2f}, {getattr(leg, 'away_team', '')} {a_recent_sot:.2f}.")
    elif g == "cards":
        # Use detail helper for richer evidence
        try:
            detail = projected_total_cards_detail(getattr(leg, "home_team", ""), getattr(leg, "away_team", ""), league)
        except Exception:
            detail = None
        if detail is None:
            cards_total, _, _, cards_ref_mod = projected_total_cards(getattr(leg, "home_team", ""), getattr(leg, "away_team", ""), league)
        else:
            cards_total = detail.get("total_cards")
            cards_ref_mod = detail.get("referee_modifier")
        if cards_total is not None:
            lines.append(f"Projected combined cards: {cards_total:.2f}.")
            if getattr(leg, "point", None) is not None:
                direction = str(getattr(leg, "outcome", "")).lower()
                if "over" in direction:
                    lines.append(f"Line fit: projected {cards_total:.2f} vs over {getattr(leg, 'point'):g}.")
                elif "under" in direction:
                    lines.append(f"Line fit: projected {cards_total:.2f} vs under {getattr(leg, 'point'):g}.")
            # Yellow/red split from detail
            if detail:
                t_y = detail.get("total_yellows")
                t_r = detail.get("total_reds")
                if t_y is not None and t_r is not None:
                    lines.append(f"Split: ~{t_y:.2f} yellows + ~{t_r:.2f} reds.")
                conf = detail.get("confidence_bucket", "low")
                if conf == "low":
                    lines.append("Cards confidence: low — higher uncertainty in projection.")
        h_cards, a_cards = v(hm, "cards_per_90_team"), v(am, "cards_per_90_team")
        if h_cards is not None and a_cards is not None:
            lines.append(f"Cards/90: {getattr(leg, 'home_team', '')} {h_cards:.2f}, {getattr(leg, 'away_team', '')} {a_cards:.2f}.")
        if getattr(cards_ref_mod, "source", "") == "profile" and getattr(cards_ref_mod, "referee_name", None):
            rw = SCORING_WEIGHTS["referee"]
            if abs(float(getattr(cards_ref_mod, "multiplier", 1.0)) - 1.0) > rw["evidence_threshold"]:
                direction_word = "stricter" if getattr(cards_ref_mod, "multiplier", 1.0) > 1.0 else "more lenient"
                lines.append(
                    f"Referee: {str(getattr(cards_ref_mod, 'referee_name', '')).title()} is {direction_word} than average "
                    f"({float(getattr(cards_ref_mod, 'strictness_ratio', 1.0)):.2f}x, "
                    f"{getattr(cards_ref_mod, 'confidence', 0):.0%} confidence)."
                )
        h_recent_cards, a_recent_cards = hr.get("cards_avg"), ar.get("cards_avg")
        if h_recent_cards is not None and a_recent_cards is not None:
            lines.append(f"Recent 6-match cards: {getattr(leg, 'home_team', '')} {h_recent_cards:.2f}, {getattr(leg, 'away_team', '')} {a_recent_cards:.2f}.")
    elif g == "btts":
        btts_result = projected_btts_prob(getattr(leg, "home_team", ""), getattr(leg, "away_team", ""), league)
        p_btts, h_proj, a_proj, _, _ = btts_result if btts_result else (None, None, None, None, None)
        if p_btts is not None:
            lines.append(f"P(BTTS Yes): {p_btts:.1%}.")
        if h_proj is not None and a_proj is not None:
            lines.append(f"Goal projections: {getattr(leg, 'home_team', '')} {h_proj:.2f}, {getattr(leg, 'away_team', '')} {a_proj:.2f}.")
        h_ga, a_ga = v(hm, "goals_against_pm"), v(am, "goals_against_pm")
        if h_ga is not None and a_ga is not None:
            lines.append(f"Goals conceded: {getattr(leg, 'home_team', '')} {h_ga:.2f}, {getattr(leg, 'away_team', '')} {a_ga:.2f}.")
    elif g == "moneyline":
        ml_result = projected_moneyline_probs(getattr(leg, "home_team", ""), getattr(leg, "away_team", ""), league)
        if ml_result and ml_result[0] is not None:
            lines.append(
                f"Moneyline model: home {ml_result[0]:.1%}, draw {ml_result[1]:.1%}, away {ml_result[2]:.1%}."
            )
        h_form, a_form = v(hm, "form_index_team"), v(am, "form_index_team")
        if h_form is not None and a_form is not None:
            lines.append(f"Form index: {getattr(leg, 'home_team', '')} {h_form:.2f} vs {getattr(leg, 'away_team', '')} {a_form:.2f}.")
    elif g == "spreads":
        proj_diff, _, _ = projected_goal_difference(getattr(leg, "home_team", ""), getattr(leg, "away_team", ""), league)
        if proj_diff is not None:
            lines.append(f"Projected goal difference (home-away): {proj_diff:+.2f}.")
        h_dom, a_dom = v(hm, "dominance_index"), v(am, "dominance_index")
        if h_dom is not None and a_dom is not None:
            lines.append(f"Dominance index: {getattr(leg, 'home_team', '')} {h_dom:.2f} vs {getattr(leg, 'away_team', '')} {a_dom:.2f}.")

    if getattr(leg, "point", None) is not None and g in {"corners", "cards", "sot", "totals"}:
        sc = leg_standalone_confidence(leg, league)
        model_p = sc.get("model_prob")
        if model_p is not None:
            implied_p_val = implied_prob(float(getattr(leg, "odds") or 0.0))
            edge = value_edge(model_p, implied_p_val)
            side = "Over" if "over" in str(getattr(leg, "outcome", "")).lower() else "Under"
            lines.append(f"P({side} {getattr(leg, 'point'):g}) = {model_p:.1%} | Books {implied_p_val:.1%} | Value {edge:+.1%}.")

    if _HAS_ML and g in {"corners", "cards", "sot", "totals"}:
        stat_map = {"corners": "corners", "cards": "cards", "sot": "sot", "totals": "goals"}
        stat_label = {"corners": "corners", "cards": "cards", "sot": "SoT", "totals": "goals"}
        ml_proj = ml_predict_total(getattr(leg, "home_team", ""), getattr(leg, "away_team", ""), league, stat_map[g], home_meta=hm, away_meta=am)
        elo_diff = get_elo_edge(getattr(leg, "home_team", ""), getattr(leg, "away_team", ""), stat_map[g])
        if ml_proj is not None:
            lines.append(f"ML model projection: {ml_proj:.1f} {stat_label[g]}.")
        if elo_diff is not None and abs(elo_diff) >= 20:
            leader = "home" if elo_diff > 0 else "away"
            lines.append(f"Elo: {leader} +{abs(elo_diff):.0f} {stat_label[g]} rating edge.")

    if g in heur_groups:
        lines.append(f"Heuristic slate also flags {g} for this fixture.")

    if not lines:
        lines.append("Team-profile stats unavailable for this fixture in the local knowledge base.")
    return lines[:6]
