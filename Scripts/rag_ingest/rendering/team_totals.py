"""
rendering/team_totals.py — render_team_totals_answer.

Extracted from rag_cli_v2.py without modification.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Cross-module imports with fallbacks
# ---------------------------------------------------------------------------

try:
    from core.weights import SCORING_WEIGHTS, EUROPEAN_COMPETITIONS
except ImportError:
    try:
        from Scripts.rag_ingest.core.weights import SCORING_WEIGHTS, EUROPEAN_COMPETITIONS
    except ImportError:
        try:
            from rag_cli_v2 import SCORING_WEIGHTS, EUROPEAN_COMPETITIONS
        except ImportError:
            SCORING_WEIGHTS = {"referee": {"modifier_weight": 0.25}}
            EUROPEAN_COMPETITIONS = {"UCL", "UEL", "UECL"}

try:
    from core.projections import projected_team_corners, projected_team_cards
except ImportError:
    try:
        from Scripts.rag_ingest.core.projections import projected_team_corners, projected_team_cards
    except ImportError:
        try:
            from rag_cli_v2 import projected_team_corners, projected_team_cards
        except ImportError:
            def projected_team_corners(*a, **kw): return (None, None, None)
            def projected_team_cards(*a, **kw): return (None, None, None)

try:
    from core.odds_extraction import extract_team_total_line_options
except ImportError:
    try:
        from Scripts.rag_ingest.core.odds_extraction import extract_team_total_line_options
    except ImportError:
        try:
            from rag_cli_v2 import extract_team_total_line_options
        except ImportError:
            def extract_team_total_line_options(*a, **kw): return []

try:
    from core.line_selection import choose_best_total_line, confidence_from_edge, _best_model_only_line
except ImportError:
    try:
        from Scripts.rag_ingest.core.line_selection import choose_best_total_line, confidence_from_edge, _best_model_only_line
    except ImportError:
        try:
            from rag_cli_v2 import choose_best_total_line, confidence_from_edge, _best_model_only_line
        except ImportError:
            def choose_best_total_line(*a, **kw): return None
            def confidence_from_edge(*a, **kw): return "low"
            def _best_model_only_line(*a, **kw): return ("Over", 2.5, 0.50)

try:
    from core.team_resolution import get_blended_variance
except ImportError:
    try:
        from Scripts.rag_ingest.core.team_resolution import get_blended_variance
    except ImportError:
        try:
            from rag_cli_v2 import get_blended_variance
        except ImportError:
            def get_blended_variance(*a, **kw): return None

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

try:
    from prob_models import over_prob
except ImportError:
    try:
        from Scripts.rag_ingest.prob_models import over_prob
    except ImportError:
        def over_prob(*a, **kw): return 0.5

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
    from knockout_context import get_knockout_context, knockout_evidence_line
    _HAS_KNOCKOUT = True
except ImportError:
    try:
        from Scripts.rag_ingest.knockout_context import get_knockout_context, knockout_evidence_line
        _HAS_KNOCKOUT = True
    except ImportError:
        _HAS_KNOCKOUT = False
        def get_knockout_context(*a, **kw):
            from dataclasses import dataclass
            @dataclass
            class _KO:
                is_knockout: bool = False
                goals_modifier: float = 1.0
                corners_modifier: float = 1.0
                cards_modifier: float = 1.0
                sot_modifier: float = 1.0
                source: str = "none"
            return _KO()
        def knockout_evidence_line(ctx) -> None:
            return None


# ---------------------------------------------------------------------------
# Exported function
# ---------------------------------------------------------------------------


def render_team_totals_answer(user_q: str, league: str, events: List[Dict]) -> str:
    """Per-team corner and card line recommendations for each fixture."""
    lines: List[str] = ["Per-Team Totals Lines by fixture:"]

    for ev in events:
        home = str(ev.get("home_team") or "Home")
        away = str(ev.get("away_team") or "Away")
        lines.append(f"\n=== {home} vs {away} ===")

        # Knockout round context for European competitions
        ko_ctx = None
        if league in EUROPEAN_COMPETITIONS:
            try:
                ko_date = ev.get("commence_time", "")
                ko_ctx = get_knockout_context(home, away, league, ko_date)
            except Exception:
                ko_ctx = None
            if ko_ctx and ko_ctx.is_knockout:
                ko_line = knockout_evidence_line(ko_ctx)
                if ko_line:
                    lines.append(f"  {ko_line}")
            for _t in (home, away):
                _note = _euro_data_source_note(_t, league)
                if _note:
                    lines.append(f"  {_t}: {_note}")

        rw = SCORING_WEIGHTS["referee"]
        ref_mod = get_referee_modifier(home, away, league,
                                       weight=rw.get("modifier_weight", 0.25))

        # ---- CORNERS ----
        lines.append("  CORNERS:")
        _corner_projs = {}
        for team, opp, is_home in [(home, away, True), (away, home, False)]:
            c_bl, c_sea, c_rec = projected_team_corners(team, opp, league, is_home)
            # Apply knockout modifier
            if ko_ctx and ko_ctx.is_knockout and ko_ctx.corners_modifier != 1.0 and c_bl is not None:
                c_bl *= ko_ctx.corners_modifier
            if c_bl is None:
                lines.append(f"    {team}: insufficient data.")
                continue
            _corner_projs[team] = c_bl
            lines.append(f"    {team}: projected {c_bl:.2f} corners/match.")
            if c_sea is not None:
                lines.append(f"      Season model: {c_sea:.2f}.")
            if c_rec is not None:
                lines.append(f"      Recent/context anchor: {c_rec:.2f}.")
            t_var = get_blended_variance(team, league, "corners_for_var")
            options = extract_team_total_line_options(ev, team, "corners")
            best = choose_best_total_line(options, c_bl, t_var) if options else None
            if best:
                side = str(best.get("side") or "").title()
                pt = float(best.get("point"))
                odds = float(best.get("odds"))
                edge = (c_bl - pt) if side.lower() == "over" else (pt - c_bl)
                mp = best.get("_model_prob")
                ip = best.get("_implied_prob")
                ve = best.get("_value_edge")
                ev_val = best.get("_ev")
                conf = confidence_from_edge(edge, stat_group="corners",
                                           model_prob=mp, value_edge_pct=ve)
                lines.append(
                    f"      Recommended: {side} {pt:g} @ {odds:.2f} "
                    f"({best.get('bookmaker')}, {best.get('market_key')})."
                )
                if mp is not None and ip is not None and ve is not None:
                    lines.append(
                        f"      Model: P({side} {pt:g}) = {mp:.1%} | "
                        f"Fair implied: {ip:.1%} | Value edge: {ve:+.1%}"
                    )
                if ev_val is not None:
                    lines.append(f"      EV: {ev_val:+.3f} per unit ({conf} confidence).")
                else:
                    lines.append(f"      Edge {edge:+.2f} ({conf} confidence).")
            else:
                side, anchor, mp = _best_model_only_line(c_bl, t_var)
                conf = "high" if mp >= 0.65 else ("medium" if mp >= 0.55 else "low")
                lines.append(
                    f"      Recommended: {side} {anchor:g} "
                    f"(model-only, P({side} {anchor:g}) = {mp:.1%}, {conf} confidence)."
                )
                # Adjacent line probabilities
                for adj_offset in [-1.0, 1.0]:
                    adj_line = anchor + adj_offset
                    if adj_line >= 0.5:
                        adj_p_over = over_prob(c_bl, adj_line, t_var)
                        adj_side = "Over" if adj_p_over >= 0.5 else "Under"
                        adj_p = adj_p_over if adj_side == "Over" else (1.0 - adj_p_over)
                        lines.append(
                            f"        Alt: {adj_side} {adj_line:g} — P = {adj_p:.1%}."
                        )

        # Match total corners summary
        if len(_corner_projs) == 2:
            _c_vals = list(_corner_projs.values())
            lines.append(f"    Match total: {_c_vals[0] + _c_vals[1]:.1f} projected corners.")

        # ---- CARDS ----
        lines.append("  CARDS:")
        if ref_mod.source == "profile":
            rn = (ref_mod.referee_name or "Unknown").title()
            ra = ref_mod.avg_cards_per_match
            rs = ref_mod.strictness_ratio
            rns = ref_mod.sample_size
            label = "strict" if rs >= 1.10 else ("lenient" if rs <= 0.90 else "average")
            lines.append(
                f"    Referee: {rn} — {ra:.2f} cards/match ({label}, "
                f"{rs:.2f}x league avg, {rns} matches)."
            )
            ref_cpf = ref_mod.cards_per_foul
            if ref_cpf > 0:
                lines.append(
                    f"    Ref cards/foul: {ref_cpf:.3f} — "
                    f"avg fouls/match: {ref_mod.avg_fouls_per_match:.1f}."
                )
        else:
            lines.append("    Referee: not yet assigned or API-Football key not set.")

        # Get match-total detail for context
        try:
            from core.projections import projected_total_cards_detail as _ptcd
        except ImportError:
            try:
                from Scripts.rag_ingest.core.projections import projected_total_cards_detail as _ptcd
            except ImportError:
                _ptcd = None
        _match_detail = None
        if _ptcd:
            try:
                _match_detail = _ptcd(home, away, league, ref_mod=ref_mod)
            except Exception:
                pass

        _card_projs = {}
        for team, opp, is_home in [(home, away, True), (away, home, False)]:
            k_bl, k_sea, k_rec = projected_team_cards(team, opp, league, is_home,
                                                       ref_mod=ref_mod)
            # Apply knockout modifier
            if ko_ctx and ko_ctx.is_knockout and ko_ctx.cards_modifier != 1.0 and k_bl is not None:
                k_bl *= ko_ctx.cards_modifier
            if k_bl is None:
                lines.append(f"    {team}: insufficient data.")
                continue
            _card_projs[team] = k_bl
            lines.append(f"    {team}: projected {k_bl:.2f} cards/match.")
            if k_sea is not None:
                lines.append(f"      Season model: {k_sea:.2f}.")
            if k_rec is not None:
                lines.append(f"      Recent/context anchor: {k_rec:.2f}.")
            t_var = get_blended_variance(team, league, "cards_var")
            options = extract_team_total_line_options(ev, team, "cards")
            best = choose_best_total_line(options, k_bl, t_var) if options else None
            if best:
                side = str(best.get("side") or "").title()
                pt = float(best.get("point"))
                odds = float(best.get("odds"))
                edge = (k_bl - pt) if side.lower() == "over" else (pt - k_bl)
                mp = best.get("_model_prob")
                ip = best.get("_implied_prob")
                ve = best.get("_value_edge")
                ev_val = best.get("_ev")
                conf = confidence_from_edge(edge, stat_group="cards",
                                           model_prob=mp, value_edge_pct=ve)
                lines.append(
                    f"      Recommended: {side} {pt:g} @ {odds:.2f} "
                    f"({best.get('bookmaker')}, {best.get('market_key')})."
                )
                if mp is not None and ip is not None and ve is not None:
                    lines.append(
                        f"      Model: P({side} {pt:g}) = {mp:.1%} | "
                        f"Fair implied: {ip:.1%} | Value edge: {ve:+.1%}"
                    )
                if ev_val is not None:
                    lines.append(f"      EV: {ev_val:+.3f} per unit ({conf} confidence).")
                else:
                    lines.append(f"      Edge {edge:+.2f} ({conf} confidence).")
            else:
                side, anchor, mp = _best_model_only_line(k_bl, t_var)
                conf = "high" if mp >= 0.65 else ("medium" if mp >= 0.55 else "low")
                lines.append(
                    f"      Recommended: {side} {anchor:g} "
                    f"(model-only, P({side} {anchor:g}) = {mp:.1%}, {conf} confidence)."
                )
                # Adjacent line probabilities
                for adj_offset in [-1.0, 1.0]:
                    adj_line = anchor + adj_offset
                    if adj_line >= 0.5:
                        adj_p_over = over_prob(k_bl, adj_line, t_var)
                        adj_side = "Over" if adj_p_over >= 0.5 else "Under"
                        adj_p = adj_p_over if adj_side == "Over" else (1.0 - adj_p_over)
                        lines.append(
                            f"        Alt: {adj_side} {adj_line:g} — P = {adj_p:.1%}."
                        )

        # Match total cards summary with detail context
        if len(_card_projs) == 2:
            _k_vals = list(_card_projs.values())
            match_total_sum = _k_vals[0] + _k_vals[1]
            lines.append(f"    Match total: {match_total_sum:.1f} projected cards.")
            if _match_detail:
                mt_anchor = _match_detail.get("total_cards")
                if mt_anchor is not None:
                    lines.append(f"    Match-total anchor: {mt_anchor:.2f}.")
                t_y = _match_detail.get("total_yellows")
                t_r = _match_detail.get("total_reds")
                if t_y is not None and t_r is not None:
                    lines.append(f"    Yellow/red split: ~{t_y:.2f} yellows + ~{t_r:.2f} reds.")
                conf_bucket = _match_detail.get("confidence_bucket", "low")
                if conf_bucket == "low":
                    lines.append("    ⚠ Low confidence — treat team-card lines as estimates only.")
                for w in (_match_detail.get("warnings") or []):
                    lines.append(f"    Note: {w}")

    lines.append("Method: per-team Poisson/NegBin probability model from local KB.")
    return "\n".join(lines)
