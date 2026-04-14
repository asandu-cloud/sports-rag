"""
core/parlay.py — CandidateLeg, parlay builder, optimizer, scoring.

Extracted from rag_cli_v2.py without modification.
"""

from __future__ import annotations

import copy
import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from math import comb as _comb, log
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
            # Inline default — will never reach here in production
            SCORING_WEIGHTS = {}

try:
    from core.projections import (
        projected_total_goals, projected_total_corners, projected_total_cards,
        projected_total_sot, projected_goal_difference, projected_moneyline_probs,
        projected_btts_prob, projected_corners, projected_cards, compute_margin_std,
    )
except ImportError:
    try:
        from Scripts.rag_ingest.core.projections import (
            projected_total_goals, projected_total_corners, projected_total_cards,
            projected_total_sot, projected_goal_difference, projected_moneyline_probs,
            projected_btts_prob, projected_corners, projected_cards, compute_margin_std,
        )
    except ImportError:
        try:
            from rag_cli_v2 import (
                projected_total_goals, projected_total_corners, projected_total_cards,
                projected_total_sot, projected_goal_difference, projected_moneyline_probs,
                projected_btts_prob, projected_corners, projected_cards, compute_margin_std,
            )
        except ImportError:
            def projected_total_goals(*a, **kw): return (None, None, None)
            def projected_total_corners(*a, **kw): return (None, None, None)
            def projected_total_cards(*a, **kw): return (None, None, None, None)
            def projected_total_sot(*a, **kw): return (None, None, None)
            def projected_goal_difference(*a, **kw): return (None, None, None)
            def projected_moneyline_probs(*a, **kw): return (None, None, None, None, None)
            def projected_btts_prob(*a, **kw): return (None, None, None, None, None)
            def projected_corners(*a, **kw): return (None, None)
            def projected_cards(*a, **kw): return (None, None)
            def compute_margin_std(*a, **kw): return SCORING_WEIGHTS["prob"]["margin_std_default"]

try:
    from core.team_resolution import _profile_meta, _recent_stats, _numeric, get_team_recent_variance
except ImportError:
    try:
        from Scripts.rag_ingest.core.team_resolution import _profile_meta, _recent_stats, _numeric, get_team_recent_variance
    except ImportError:
        try:
            from rag_cli_v2 import _profile_meta, _recent_stats, _numeric, get_team_recent_variance
        except ImportError:
            def _profile_meta(team, league): return {}
            def _recent_stats(team, league, last_n=6): return {}
            def _numeric(value):
                try: return float(value)
                except Exception: return None
            def get_team_recent_variance(team, league, last_n=8): return {}

try:
    from prob_models import over_prob, under_prob
except ImportError:
    try:
        from Scripts.rag_ingest.prob_models import over_prob, under_prob
    except ImportError:
        def over_prob(*a, **kw): return 0.5
        def under_prob(*a, **kw): return 0.5

try:
    from core.line_selection import (
        _score_matrix_spread_profile, _normal_spread_profile,
        _spread_expected_value, _equivalent_binary_prob,
    )
except ImportError:
    try:
        from Scripts.rag_ingest.core.line_selection import (
            _score_matrix_spread_profile, _normal_spread_profile,
            _spread_expected_value, _equivalent_binary_prob,
        )
    except ImportError:
        def _score_matrix_spread_profile(*a, **kw): return None
        def _normal_spread_profile(*a, **kw):
            return {
                "full_win": 0.0, "half_win": 0.0, "push": 0.0,
                "half_loss": 0.0, "full_loss": 1.0,
            }
        def _spread_expected_value(*a, **kw): return -1.0
        def _equivalent_binary_prob(*a, **kw): return 0.5

try:
    from ml_edge import get_elo_edge
    _HAS_ML = True
except ImportError:
    try:
        from Scripts.rag_ingest.ml_edge import get_elo_edge
        _HAS_ML = True
    except ImportError:
        _HAS_ML = False
        def get_elo_edge(*a, **kw): return None

try:
    from heuristics import top_markets
except ImportError:
    try:
        from Scripts.rag_ingest.heuristics import top_markets
    except ImportError:
        def top_markets(*a, **kw): return []


# ---------------------------------------------------------------------------
# Constants also defined in rag_cli_v2.py — duplicated here for standalone use
# ---------------------------------------------------------------------------

HALF_MARKET_PATTERN = re.compile(r"(?:^|_)(h1|h2|1st_half|2nd_half|first_half|second_half)(?:_|$)")

AROUND_TARGET_TOLERANCE = 0.20

try:
    from rag_cli_v2 import MIN_LEG_ODDS, MIN_TOTALS_LEG_ODDS
except ImportError:
    MIN_LEG_ODDS = 1.10
    MIN_TOTALS_LEG_ODDS = 1.25


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateLeg:
    event_id: str
    fixture: str
    home_team: str
    away_team: str
    market_key: str
    outcome: str
    odds: float
    point: Optional[float]
    bookmaker: Optional[str]
    league: str = ""


# ---------------------------------------------------------------------------
# ConstraintSpec import (needed by score_combo / select_parlay)
# ---------------------------------------------------------------------------

try:
    from rag_cli_v2 import ConstraintSpec
except ImportError:
    @dataclass
    class ConstraintSpec:
        requested_markets: Set[str] = None
        hard_include_groups: Set[str] = None
        hard_exclude_groups: Set[str] = None
        soft_prefer_groups: Set[str] = None
        required_group_counts: Dict[str, int] = None
        forbid_spread_keys: bool = False
        require_total_corner_keys: bool = False
        target_multiplier: Optional[float] = None
        target_mode: str = "none"
        leg_count: Optional[int] = None
        per_match_mode: bool = False
        require_unique_events: bool = True
        time_window: str = "upcoming"
        league: str = "EPL"
        leagues: List[str] = None
        target_min: Optional[float] = None
        target_max: Optional[float] = None


# ---------------------------------------------------------------------------
# Helper functions (also in rag_cli_v2.py)
# ---------------------------------------------------------------------------


def market_group_from_key(market_key: str) -> str:
    k = (market_key or "").lower()
    if "corner" in k:
        return "corners"
    if "card" in k or "booking" in k:
        return "cards"
    # SoT must be checked BEFORE generic "total" catch-all
    if ("shot" in k and "target" in k) or k == "sot":
        return "sot"
    if "btts" in k or "both_teams" in k:
        return "btts"
    if "h2h" in k or "moneyline" in k or "1x2" in k:
        return "moneyline"
    if "spread" in k or "handicap" in k:
        return "spreads"
    if "total" in k:
        return "totals"
    return k


def is_full_game_market(market_key: str) -> bool:
    k = (market_key or "").lower().strip()
    if not k:
        return False
    if "lay" in k:
        return False
    if "player" in k:
        return False
    if HALF_MARKET_PATTERN.search(k):
        return False
    return True


def _leg_league(leg: CandidateLeg, fallback: str) -> str:
    """Return the league for a leg, falling back to constraint league."""
    return leg.league if leg.league else fallback


def combined_odds(legs: List[CandidateLeg]) -> float:
    p = 1.0
    for leg in legs:
        p *= leg.odds
    return p


def available_market_groups(candidates: List[CandidateLeg]) -> Set[str]:
    return {market_group_from_key(c.market_key) for c in candidates}


def heuristic_groups_for_event(home_meta: Dict, away_meta: Dict) -> Set[str]:
    if not home_meta or not away_meta:
        return set()
    groups: Set[str] = set()
    try:
        picks = top_markets(home_meta, away_meta, n=5)
    except Exception:
        picks = []
    for p in picks:
        label = str(p.get("market") or "").lower()
        if "corner" in label:
            groups.add("corners")
        elif "card" in label or "booking" in label:
            groups.add("cards")
        elif "shot" in label or "sot" in label:
            groups.add("sot")
        elif "btts" in label:
            groups.add("btts")
        elif "goal" in label or "under" in label or "over" in label:
            groups.add("totals")
    return groups


# ---------------------------------------------------------------------------
# Enrichment (enrich_events_for_groups)
# ---------------------------------------------------------------------------

# These helpers are needed by enrich_events_for_groups but live in rag_cli_v2.
# Import them with fallback stubs.
try:
    from rag_cli_v2 import (
        event_market_groups, discover_event_market_keys,
        fetch_event_odds_for_market_keys, merge_bookmakers,
    )
except ImportError:
    def event_market_groups(event: Dict) -> Set[str]:
        groups: Set[str] = set()
        for bm in event.get("bookmakers", []) or []:
            for mk in bm.get("markets", []) or []:
                key = str(mk.get("key") or "")
                if key:
                    groups.add(market_group_from_key(key))
        return groups

    def discover_event_market_keys(league, event_id):
        return set(), "stub: not available outside monolith"

    def fetch_event_odds_for_market_keys(league, event_id, market_keys):
        return None, "stub: not available outside monolith"

    def merge_bookmakers(base, extra):
        by_title = {str(b.get("title") or ""): dict(b) for b in base}
        for eb in extra:
            t = str(eb.get("title") or "")
            if t in by_title:
                existing_keys = {str(m.get("key") or "") for m in by_title[t].get("markets", [])}
                for m in eb.get("markets", []) or []:
                    if str(m.get("key") or "") not in existing_keys:
                        by_title[t]["markets"].append(m)
            else:
                by_title[t] = dict(eb)
        return list(by_title.values())


def enrich_events_for_groups(events: List[Dict], league: str, desired_groups: Set[str]) -> Tuple[List[Dict], List[str]]:
    """
    For groups like corners/cards that are often unavailable at sport-level odds,
    discover event-level market keys and fetch event-specific odds.
    """
    if not desired_groups:
        return events, []

    notes: List[str] = []
    out: List[Dict] = []
    for ev in events:
        existing_groups = event_market_groups(ev)
        missing = {g for g in desired_groups if g not in existing_groups}
        if not missing:
            out.append(ev)
            continue

        event_id = str(ev.get("id") or "")
        if not event_id:
            out.append(ev)
            continue

        keys, err = discover_event_market_keys(league, event_id)
        if err:
            notes.append(f"{ev.get('home_team')} vs {ev.get('away_team')}: market discovery failed ({err})")
            out.append(ev)
            continue

        wanted_keys = {k for k in keys if market_group_from_key(k) in missing}
        if not wanted_keys:
            notes.append(
                f"{ev.get('home_team')} vs {ev.get('away_team')}: no {', '.join(sorted(missing))} keys returned by provider."
            )
            out.append(ev)
            continue

        ev_payload, err2 = fetch_event_odds_for_market_keys(league, event_id, wanted_keys)
        if err2 or not ev_payload:
            notes.append(
                f"{ev.get('home_team')} vs {ev.get('away_team')}: failed fetching event odds for discovered keys."
            )
            out.append(ev)
            continue

        merged = dict(ev)
        merged["bookmakers"] = merge_bookmakers(ev.get("bookmakers", []) or [], ev_payload.get("bookmakers", []) or [])
        out.append(merged)

    return out, notes


# ---------------------------------------------------------------------------
# build_candidates
# ---------------------------------------------------------------------------


def build_candidates(events: List[Dict]) -> List[CandidateLeg]:
    best_rows: Dict[Tuple[str, str, str, str], CandidateLeg] = {}
    for ev in events:
        event_id = str(ev.get("id") or "")
        home = str(ev.get("home_team") or "Home")
        away = str(ev.get("away_team") or "Away")
        fixture = f"{home} vs {away}"
        ev_league = ev.get("_league", "")
        for bm in ev.get("bookmakers", []) or []:
            bm_title = bm.get("title")
            for mk in bm.get("markets", []) or []:
                market_key = str(mk.get("key") or "")
                if not is_full_game_market(market_key):
                    continue
                for out in mk.get("outcomes", []) or []:
                    name = str(out.get("name") or "")
                    if not name:
                        continue
                    point_raw = out.get("point")
                    try:
                        point = float(point_raw) if point_raw is not None else None
                    except Exception:
                        point = None
                    try:
                        odds = float(out.get("price"))
                    except Exception:
                        continue
                    if odds < MIN_LEG_ODDS:
                        continue
                    # Higher floor for totals-style markets to avoid trivially safe lines
                    grp = market_group_from_key(market_key)
                    if grp in {"totals", "corners", "cards", "sot"} and odds < MIN_TOTALS_LEG_ODDS:
                        continue
                    row = CandidateLeg(
                        event_id=event_id,
                        fixture=fixture,
                        home_team=home,
                        away_team=away,
                        market_key=market_key,
                        outcome=name,
                        odds=odds,
                        point=point,
                        bookmaker=bm_title,
                        league=ev_league,
                    )
                    k = (event_id, market_key, name, "" if point is None else str(point))
                    cur = best_rows.get(k)
                    if cur is None or row.odds > cur.odds:
                        best_rows[k] = row
    return list(best_rows.values())


# ---------------------------------------------------------------------------
# contradictory
# ---------------------------------------------------------------------------


def contradictory(a: CandidateLeg, b: CandidateLeg) -> bool:
    if a.event_id != b.event_id:
        return False
    ga = market_group_from_key(a.market_key)
    gb = market_group_from_key(b.market_key)

    # One-leg-per-market-family per fixture.
    if ga == gb and ga in {"moneyline", "spreads", "totals", "corners", "cards", "sot", "btts"}:
        return True

    if (
        ga == gb
        and ga in {"totals", "spreads"}
        and a.point == b.point
        and a.outcome.lower() != b.outcome.lower()
    ):
        return True
    if a.market_key == b.market_key and a.point == b.point and a.outcome.lower() != b.outcome.lower():
        return True

    def side_of_outcome(leg: CandidateLeg) -> Optional[str]:
        out = (leg.outcome or "").strip().lower()
        h = (leg.home_team or "").strip().lower()
        aw = (leg.away_team or "").strip().lower()
        if out == h:
            return "home"
        if out == aw:
            return "away"
        if out in {"draw", "tie", "x"}:
            return "draw"
        if h and out.startswith(h):
            return "home"
        if aw and out.startswith(aw):
            return "away"
        return None

    side_a = side_of_outcome(a)
    side_b = side_of_outcome(b)

    if {ga, gb} == {"moneyline", "spreads"} and side_a in {"home", "away"} and side_b in {"home", "away"}:
        ml = a if ga == "moneyline" else b
        sp = b if ga == "moneyline" else a
        ml_side = side_of_outcome(ml)
        sp_side = side_of_outcome(sp)
        sp_point = sp.point
        if sp_point is not None and ml_side == sp_side:
            if abs(float(sp_point)) <= 0.5:
                return True
            if (sp_side == "home" and sp_point <= -0.5) or (sp_side == "away" and sp_point >= 0.5):
                return True
    return False


# ---------------------------------------------------------------------------
# combo_valid
# ---------------------------------------------------------------------------


def combo_valid(combo: List[CandidateLeg], require_unique_events: bool) -> bool:
    if any(x.odds < MIN_LEG_ODDS for x in combo):
        return False

    if require_unique_events:
        ids = [x.event_id for x in combo]
        if len(ids) != len(set(ids)):
            return False

    groups = {market_group_from_key(x.market_key) for x in combo}
    if "moneyline" in groups and "spreads" in groups:
        return False

    for i in range(len(combo)):
        for j in range(i + 1, len(combo)):
            if contradictory(combo[i], combo[j]):
                return False
    return True


# ---------------------------------------------------------------------------
# kb_leg_quality
# ---------------------------------------------------------------------------


def kb_leg_quality(leg: CandidateLeg, league: str) -> float:
    """
    Positive score indicates stronger multi-signal support from local KB.
    """
    hm = _profile_meta(leg.home_team, league)
    am = _profile_meta(leg.away_team, league)
    h_recent = _recent_stats(leg.home_team, league, last_n=6)
    a_recent = _recent_stats(leg.away_team, league, last_n=6)
    heur_groups = heuristic_groups_for_event(hm, am)

    g = market_group_from_key(leg.market_key)
    score = 0.0

    h_ctrl = _numeric(hm.get("control_index"))
    a_ctrl = _numeric(am.get("control_index"))
    h_form = _numeric(hm.get("form_index_team"))
    a_form = _numeric(am.get("form_index_team"))
    h_dom = _numeric(hm.get("dominance_index"))
    a_dom = _numeric(am.get("dominance_index"))

    w = SCORING_WEIGHTS["kb_quality"]

    if g in heur_groups:
        score += w["heuristic_group_bonus"]

    if g == "moneyline":
        edge = 0.0
        home_sign = 1.0 if leg.outcome.lower().startswith(leg.home_team.lower()) else -1.0
        if h_form is not None and a_form is not None:
            edge += (h_form - a_form) * w["form_edge"] * home_sign
        if h_ctrl is not None and a_ctrl is not None:
            edge += (h_ctrl - a_ctrl) * w["control_edge"] * home_sign
        if h_dom is not None and a_dom is not None:
            edge += (h_dom - a_dom) * w["dominance_edge"] * home_sign
        h_sot = h_recent.get("sot_for_avg")
        a_sot = a_recent.get("sot_for_avg")
        if h_sot is not None and a_sot is not None:
            edge += (h_sot - a_sot) * w["sot_edge"] * home_sign
        score += edge
        # Probability-based moneyline scoring (consistent with count markets)
        try:
            ml_probs = projected_moneyline_probs(leg.home_team, leg.away_team, league)
            if ml_probs is not None:
                p_home, p_draw, p_away = ml_probs[:3]
                if p_home is not None and p_away is not None:
                    is_home = leg.outcome.lower().startswith(leg.home_team.lower())
                    model_p = p_home if is_home else p_away
                    prob_bonus = (model_p - 0.50) * SCORING_WEIGHTS["prob"]["prob_quality_weight"]
                    score += prob_bonus
        except Exception:
            pass

    if g == "spreads":
        proj_diff, _, _ = projected_goal_difference(leg.home_team, leg.away_team, league)
        if proj_diff is not None:
            is_home = leg.outcome.lower().startswith(leg.home_team.lower())
            sign = 1.0 if is_home else -1.0
            point = leg.point if leg.point is not None else 0.0
            edge = sign * proj_diff + point
            score += edge * w["spreads_edge_weight"]

    if g == "totals":
        goals_proj, _, _ = projected_total_goals(leg.home_team, leg.away_team, league)
        if goals_proj is not None:
            direction = leg.outcome.lower()
            if "over" in direction:
                score += goals_proj - w["goals_over_offset"]
            elif "under" in direction:
                score += w["goals_under_offset"] - goals_proj
            if leg.point is not None:
                if "over" in direction:
                    score += (goals_proj - leg.point) * w["goals_line_fit"]
                elif "under" in direction:
                    score += (leg.point - goals_proj) * w["goals_line_fit"]

    if g == "corners":
        corners_proj, _, _ = projected_total_corners(leg.home_team, leg.away_team, league)
        if corners_proj is not None:
            direction = leg.outcome.lower()
            if leg.point is not None and "total" in leg.market_key.lower():
                if "over" in direction:
                    score += (corners_proj - leg.point) * w["corners_line_fit"]
                elif "under" in direction:
                    score += (leg.point - corners_proj) * w["corners_line_fit"]
            # Side edge for corner-winner markets
            if leg.point is None or "total" not in leg.market_key.lower():
                h_proj_c, a_proj_c = projected_corners(hm, am)
                if h_proj_c is not None and a_proj_c is not None:
                    side_edge = h_proj_c - a_proj_c
                    h_recent_for = h_recent.get("corners_for_avg")
                    a_recent_for = a_recent.get("corners_for_avg")
                    if h_recent_for is not None and a_recent_for is not None:
                        recent_side = h_recent_for - a_recent_for
                        side_edge = 0.65 * side_edge + 0.35 * recent_side
                    if leg.outcome.lower().startswith(leg.home_team.lower()):
                        score += side_edge * w["corners_side_edge"]
                    elif leg.outcome.lower().startswith(leg.away_team.lower()):
                        score -= side_edge * w["corners_side_edge"]

    if g == "cards":
        cards_result = projected_total_cards(leg.home_team, leg.away_team, league)
        cards_proj = cards_result[0] if cards_result else None

        if cards_proj is not None:
            direction = leg.outcome.lower()
            if leg.point is not None and "total" in leg.market_key.lower():
                if "over" in direction:
                    score += (cards_proj - leg.point) * w["cards_line_fit"]
                elif "under" in direction:
                    score += (leg.point - cards_proj) * w["cards_line_fit"]

                # Foul momentum: recent fouls trending up -> cards more likely
                h_fouls = _numeric(hm.get("fouls_per_90_team"))
                a_fouls = _numeric(am.get("fouls_per_90_team"))
                h_recent_fouls = h_recent.get("fouls_avg")
                a_recent_fouls = a_recent.get("fouls_avg")
                foul_momentum = 0.0
                n_mom = 0
                if h_fouls is not None and h_recent_fouls is not None:
                    foul_momentum += h_recent_fouls - h_fouls; n_mom += 1
                if a_fouls is not None and a_recent_fouls is not None:
                    foul_momentum += a_recent_fouls - a_fouls; n_mom += 1
                if n_mom > 0:
                    avg_mom = foul_momentum / n_mom
                    if "over" in direction:
                        score += avg_mom * w["cards_foul_momentum"]
                    elif "under" in direction:
                        score -= avg_mom * w["cards_foul_momentum"]

            # Side edge for card-winner markets
            if leg.point is None or "total" not in leg.market_key.lower():
                h_cards_proj, a_cards_proj = projected_cards(hm, am)
                if h_cards_proj is not None and a_cards_proj is not None:
                    side_edge = h_cards_proj - a_cards_proj
                    if leg.outcome.lower().startswith(leg.home_team.lower()):
                        score += side_edge * w["cards_side_edge"]
                    elif leg.outcome.lower().startswith(leg.away_team.lower()):
                        score -= side_edge * w["cards_side_edge"]
                    # Card-induction side edge
                    h_opp_induced = _numeric(hm.get("opp_cards_induced_pm"))
                    a_opp_induced = _numeric(am.get("opp_cards_induced_pm"))
                    if h_opp_induced is not None and a_opp_induced is not None:
                        induction_edge = a_opp_induced - h_opp_induced
                        if leg.outcome.lower().startswith(leg.home_team.lower()):
                            score += induction_edge * w["cards_opp_induced_w"]
                        elif leg.outcome.lower().startswith(leg.away_team.lower()):
                            score -= induction_edge * w["cards_opp_induced_w"]

    if g == "sot":
        sot_total, sot_season, sot_recent = projected_total_sot(leg.home_team, leg.away_team, league)
        if sot_total is not None:
            direction = leg.outcome.lower()
            if "over" in direction:
                score += sot_total - w["sot_over_offset"]
            elif "under" in direction:
                score += w["sot_under_offset"] - sot_total
            if leg.point is not None:
                if "over" in direction:
                    score += (sot_total - leg.point) * w["sot_line_fit"]
                elif "under" in direction:
                    score += (leg.point - sot_total) * w["sot_line_fit"]

    if g == "btts":
        result = projected_btts_prob(leg.home_team, leg.away_team, league)
        p_btts = result[0] if result else None
        if p_btts is not None:
            direction = leg.outcome.lower().strip()
            if direction == "yes":
                score += (p_btts - 0.50) * w.get("btts_quality_weight", 1.5)
            elif direction == "no":
                score += (0.50 - p_btts) * w.get("btts_quality_weight", 1.5)

    if leg.odds >= w["long_odds_threshold"]:
        score -= (leg.odds - w["long_odds_offset"]) * w["long_odds_penalty"]

    # --- Group preference multiplier ---
    group_pref = SCORING_WEIGHTS["group_preference"].get(g, 1.0)
    score *= group_pref

    # --- Elo signal for count-based markets ---
    if g in ("corners", "cards", "sot", "totals"):
        stat_map = {"corners": "corners", "cards": "cards", "sot": "sot", "totals": "goals"}
        mw = SCORING_WEIGHTS["ml"]
        elo_diff = get_elo_edge(leg.home_team, leg.away_team, stat_map[g])
        if elo_diff is not None and abs(elo_diff) >= mw["min_elo_diff"]:
            elo_norm = elo_diff / mw["elo_scale"]
            direction = leg.outcome.lower()
            home_outcome = direction.startswith(leg.home_team.lower()) if hasattr(leg, 'home_team') else False
            if "over" in direction:
                score += elo_norm * mw["elo_signal_weight"]
            elif "under" in direction:
                score -= elo_norm * mw["elo_signal_weight"]
            elif home_outcome:
                score += elo_norm * mw["elo_signal_weight"]
            else:
                score -= elo_norm * mw["elo_signal_weight"]

    # --- Probability-aware quality bonus for count-based markets ---
    if leg.point is not None and g in ("corners", "cards", "sot", "totals"):
        proj_funcs = {
            "corners": lambda: projected_total_corners(leg.home_team, leg.away_team, league),
            "cards": lambda: projected_total_cards(leg.home_team, leg.away_team, league),
            "sot": lambda: projected_total_sot(leg.home_team, leg.away_team, league),
            "totals": lambda: projected_total_goals(leg.home_team, leg.away_team, league),
        }
        proj_result = proj_funcs[g]()
        proj_total = proj_result[0] if proj_result else None
        if proj_total is not None:
            direction = leg.outcome.lower()
            is_over = "over" in direction
            h_var = get_team_recent_variance(leg.home_team, league)
            a_var = get_team_recent_variance(leg.away_team, league)
            _stat_var_key = {"corners": "corners_for_var", "cards": "cards_var",
                             "sot": "sot_for_var", "totals": "goals_var"}[g]
            h_v = h_var.get(_stat_var_key)
            a_v = a_var.get(_stat_var_key)
            combined_var = (h_v + a_v) if (h_v is not None and a_v is not None) else None
            model_p = (over_prob(proj_total, leg.point, combined_var) if is_over
                       else under_prob(proj_total, leg.point, combined_var))
            prob_bonus = (model_p - 0.50) * SCORING_WEIGHTS["prob"]["prob_quality_weight"]
            score += prob_bonus

    return score


# ---------------------------------------------------------------------------
# combo probability helpers
# ---------------------------------------------------------------------------


def _leg_side(leg: CandidateLeg) -> Optional[str]:
    outcome = (leg.outcome or "").strip().lower()
    home = (leg.home_team or "").strip().lower()
    away = (leg.away_team or "").strip().lower()
    if outcome in {"draw", "tie", "x"}:
        return "draw"
    if outcome == home or (home and outcome.startswith(home)):
        return "home"
    if outcome == away or (away and outcome.startswith(away)):
        return "away"
    return None


def _combo_leg_model_probability(leg: CandidateLeg, fallback_league: str) -> Optional[float]:
    leg_lg = _leg_league(leg, fallback_league)
    g = market_group_from_key(leg.market_key)

    if leg.point is not None and g in ("corners", "cards", "sot", "totals"):
        proj_funcs = {
            "corners": lambda: projected_total_corners(leg.home_team, leg.away_team, leg_lg),
            "cards": lambda: projected_total_cards(leg.home_team, leg.away_team, leg_lg),
            "sot": lambda: projected_total_sot(leg.home_team, leg.away_team, leg_lg),
            "totals": lambda: projected_total_goals(leg.home_team, leg.away_team, leg_lg),
        }
        proj_result = proj_funcs[g]()
        proj_total = proj_result[0] if proj_result else None
        if proj_total is None:
            return None
        var_key = {
            "corners": "corners_for_var",
            "cards": "cards_var",
            "sot": "sot_for_var",
            "totals": "goals_var",
        }[g]
        h_var = get_team_recent_variance(leg.home_team, leg_lg)
        a_var = get_team_recent_variance(leg.away_team, leg_lg)
        h_v = h_var.get(var_key)
        a_v = a_var.get(var_key)
        combined_var = (h_v + a_v) if (h_v is not None and a_v is not None) else None
        is_over = "over" in (leg.outcome or "").lower()
        return over_prob(proj_total, leg.point, combined_var) if is_over else under_prob(proj_total, leg.point, combined_var)

    if g == "btts":
        result = projected_btts_prob(leg.home_team, leg.away_team, leg_lg)
        p_btts = result[0] if result else None
        if p_btts is None:
            return None
        return p_btts if (leg.outcome or "").strip().lower() == "yes" else (1.0 - p_btts)

    if g == "moneyline":
        p_home, p_draw, p_away, _, _ = projected_moneyline_probs(leg.home_team, leg.away_team, leg_lg)
        side = _leg_side(leg)
        if side == "home":
            return p_home
        if side == "away":
            return p_away
        if side == "draw":
            return p_draw
        return None

    if g == "spreads" and leg.point is not None:
        side = _leg_side(leg)
        if side not in {"home", "away"}:
            return None
        is_home = side == "home"
        proj_diff, _, _ = projected_goal_difference(leg.home_team, leg.away_team, leg_lg)
        if proj_diff is None:
            return None
        sign = 1.0 if is_home else -1.0
        margin_std = compute_margin_std(leg.home_team, leg.away_team, leg_lg)
        profile = _score_matrix_spread_profile(leg.home_team, leg.away_team, leg_lg, is_home, leg.point)
        if profile is None:
            profile = _normal_spread_profile(sign * proj_diff, margin_std, leg.point)
        ev = _spread_expected_value(profile, leg.odds)
        return _equivalent_binary_prob(ev, leg.odds)

    return None


# ---------------------------------------------------------------------------
# score_combo
# ---------------------------------------------------------------------------


def score_combo(combo: List[CandidateLeg], c: ConstraintSpec, leg_quality: Dict[CandidateLeg, float]) -> float:
    w = SCORING_WEIGHTS["combo"]
    combo_odds = 1.0
    for leg in combo:
        combo_odds *= leg.odds

    # Base scoring
    if c.target_multiplier and c.target_mode == "around":
        score = abs(log(combo_odds) - log(c.target_multiplier)) * w["around_target_log_penalty"]
    else:
        neutral_anchor = w["neutral_anchor_base"] ** max(1, len(combo))
        score = abs(log(combo_odds) - log(neutral_anchor)) * w["neutral_anchor_deviation"]

    if c.target_multiplier and c.target_multiplier > 0:
        tgt = c.target_multiplier
        if c.target_mode == "max":
            score += abs(combo_odds - tgt) * w["target_abs_deviation"]
            if combo_odds > tgt:
                score += (combo_odds - tgt) * w["target_over_limit_penalty"]
        elif c.target_mode == "min":
            score += abs(combo_odds - tgt) * w["target_abs_deviation"]
            if combo_odds < tgt:
                score += (tgt - combo_odds) * w["target_over_limit_penalty"]
        else:
            rel_dev = abs(combo_odds - tgt) / max(tgt, 1e-9)
            if rel_dev > AROUND_TARGET_TOLERANCE:
                score += (rel_dev - AROUND_TARGET_TOLERANCE) * w["around_over_tolerance_penalty"]

        desired_leg = tgt ** (1.0 / max(1, len(combo)))
        for leg in combo:
            if c.target_mode in {"around", "min"} and leg.odds < desired_leg * w["desired_leg_floor_ratio"]:
                score += (desired_leg * w["desired_leg_floor_ratio"] - leg.odds) * w["desired_leg_under_penalty"]

    # --- Odds range constraint ---
    if c.target_min is not None and c.target_max is not None:
        if not c.target_multiplier:
            range_mid = (c.target_min + c.target_max) / 2.0
            score = abs(log(combo_odds) - log(range_mid)) * w["around_target_log_penalty"] * 0.5
        if combo_odds < c.target_min:
            score += (c.target_min - combo_odds) * w["target_over_limit_penalty"]
        elif combo_odds > c.target_max:
            score += (combo_odds - c.target_max) * w["target_over_limit_penalty"]

    # Soft preferences
    if c.soft_prefer_groups:
        for leg in combo:
            g = market_group_from_key(leg.market_key)
            if g not in c.soft_prefer_groups:
                score += w["soft_prefer_miss"]

    # Global anti-conservative guardrail.
    for leg in combo:
        if leg.odds < w["conservative_threshold"]:
            score += (w["conservative_threshold"] - leg.odds) * w["conservative_penalty"]

    # Prefer higher multi-signal support from local KB.
    quality = sum(leg_quality.get(leg, 0.0) for leg in combo)
    score -= quality * w["kb_quality_weight"]

    # --- EV-aware scoring: reward positive-EV combos ---
    pw = SCORING_WEIGHTS["prob"]
    joint_prob = 1.0
    any_prob = False
    for leg in combo:
        model_p = _combo_leg_model_probability(leg, c.league)
        if model_p is not None:
            joint_prob *= model_p
            any_prob = True
        else:
            joint_prob *= 0.5
    if any_prob:
        combo_ev = (joint_prob * combo_odds) - 1.0
        score -= combo_ev * pw["ev_combo_weight"]

    # Diversity penalty for same-event, same-thesis stacking beyond contradictions.
    for i in range(len(combo)):
        for j in range(i + 1, len(combo)):
            a = combo[i]
            b = combo[j]
            if a.event_id != b.event_id:
                continue
            ga = market_group_from_key(a.market_key)
            gb = market_group_from_key(b.market_key)
            if {ga, gb} == {"moneyline", "spreads"}:
                score += w["ml_spread_stack_penalty"]

    # --- Market group diversity ---
    dw = SCORING_WEIGHTS["diversity"]
    group_counts = Counter(market_group_from_key(leg.market_key) for leg in combo)
    n_legs = len(combo)
    max_count = max(group_counts.values())
    concentration = max_count / n_legs
    if concentration > dw["concentration_threshold"]:
        score += (concentration - dw["concentration_threshold"]) * dw["concentration_penalty"] * n_legs
    for _grp, cnt in group_counts.items():
        if cnt > dw["max_same_group"]:
            score += (cnt - dw["max_same_group"]) * dw["excess_same_penalty"]
    unique_groups = len(group_counts)
    score -= unique_groups * dw["unique_group_bonus"]

    # --- League diversity (cross-league mode) ---
    if c.leagues:
        league_counts = Counter(_leg_league(leg, c.league) for leg in combo)
        unique_leagues = len(league_counts)
        score -= unique_leagues * dw.get("unique_league_bonus", 0.3)
        max_league_count = max(league_counts.values())
        if max_league_count > n_legs * 0.6:
            score += (max_league_count - n_legs * 0.6) * dw.get("league_concentration_penalty", 2.0)

    # Hard excludes should be strongly avoided.
    for leg in combo:
        g = market_group_from_key(leg.market_key)
        if g in c.hard_exclude_groups:
            score += w["hard_exclude_penalty"]
        if c.forbid_spread_keys and (("spread" in leg.market_key.lower()) or ("handicap" in leg.market_key.lower())):
            score += w["forbid_spread_penalty"]
        if c.require_total_corner_keys:
            k = leg.market_key.lower()
            if ("corner" not in k) or ("total" not in k):
                score += w["require_total_corner_penalty"]

    # Hard includes should appear when available.
    for req in c.hard_include_groups:
        if not any(market_group_from_key(x.market_key) == req for x in combo):
            score += w["hard_include_missing"]

    # Composition constraints (e.g., 1 corner leg + 1 booking leg)
    for grp, need in c.required_group_counts.items():
        have = sum(1 for x in combo if market_group_from_key(x.market_key) == grp)
        if have < need:
            score += (need - have) * w["composition_missing"]

    return score


# ---------------------------------------------------------------------------
# check_feasibility
# ---------------------------------------------------------------------------


def check_feasibility(candidates: List[CandidateLeg], leg_count: int, c: ConstraintSpec) -> Tuple[bool, List[str]]:
    """
    Pre-check whether constraints are satisfiable before the expensive solver.
    Returns (feasible, warnings). Non-feasible means the solver would waste time.
    """
    warnings: List[str] = []
    if leg_count <= 0:
        return False, ["Invalid leg count."]
    if not candidates:
        return False, ["No odds candidates available."]

    avail_groups = {market_group_from_key(c_leg.market_key) for c_leg in candidates}
    unique_events = len({c_leg.event_id for c_leg in candidates})

    if c.require_unique_events and unique_events < leg_count:
        warnings.append(
            f"Requested {leg_count} legs across unique fixtures, but only {unique_events} fixture(s) have odds. "
            f"Reducing leg count to {unique_events}."
        )

    for grp in c.hard_include_groups:
        if grp not in avail_groups:
            warnings.append(
                f"Requested '{grp}' markets, but none are available from the current odds feed. "
                f"Available groups: {', '.join(sorted(avail_groups)) or 'none'}."
            )

    for grp, need in c.required_group_counts.items():
        grp_legs = [c_leg for c_leg in candidates if market_group_from_key(c_leg.market_key) == grp]
        if len(grp_legs) < need:
            warnings.append(
                f"Requested {need} '{grp}' leg(s), but only {len(grp_legs)} candidate(s) available."
            )

    if c.require_total_corner_keys:
        corner_total_legs = [
            c_leg for c_leg in candidates
            if "corner" in c_leg.market_key.lower() and "total" in c_leg.market_key.lower()
        ]
        if not corner_total_legs:
            warnings.append(
                "Total-corners-only requested, but no full-game total corner market keys found in feed."
            )

    feasible = not any(w.startswith("Requested") and "none are available" in w for w in warnings)
    return feasible, warnings


# ---------------------------------------------------------------------------
# _brute_force_combos / _beam_search_combos
# ---------------------------------------------------------------------------


def _brute_force_combos(
    pool: List[CandidateLeg], leg_count: int, c: ConstraintSpec, leg_quality: Dict[CandidateLeg, float]
) -> List[Tuple[List[CandidateLeg], float, float]]:
    scored: List[Tuple[List[CandidateLeg], float, float]] = []
    for combo in combinations(pool, leg_count):
        combo_list = list(combo)
        if not combo_valid(combo_list, c.require_unique_events):
            continue
        s = score_combo(combo_list, c, leg_quality)
        scored.append((combo_list, s, combined_odds(combo_list)))
    return scored


def _beam_search_combos(
    pool: List[CandidateLeg], leg_count: int, c: ConstraintSpec, leg_quality: Dict[CandidateLeg, float]
) -> List[Tuple[List[CandidateLeg], float, float]]:
    """Greedy beam search: O(beam_width * pool_size * leg_count)."""
    pw = SCORING_WEIGHTS["pool"]
    cw = SCORING_WEIGHTS["combo"]
    beam_width = int(pw["beam_width"])

    # Seed beams with individual legs
    seeds: List[Tuple[List[CandidateLeg], float]] = []
    for leg in pool:
        if leg.odds < MIN_LEG_ODDS:
            continue
        seeds.append(([leg], -leg_quality.get(leg, 0.0)))
    seeds.sort(key=lambda x: x[1])
    beam = seeds[:beam_width]

    for step in range(1, leg_count):
        next_beam: List[Tuple[List[CandidateLeg], float]] = []
        seen_keys: Set[frozenset] = set()
        for combo, _ in beam:
            combo_set = set(id(l) for l in combo)
            for leg in pool:
                if id(leg) in combo_set:
                    continue
                extended = combo + [leg]
                if not combo_valid(extended, c.require_unique_events):
                    continue
                # Deduplicate by leg identity
                key = frozenset((l.event_id, l.market_key, l.outcome, l.point) for l in extended)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                # Heuristic for partial combos: proximity to target + KB quality
                partial_quality = sum(leg_quality.get(l, 0.0) for l in extended)
                combo_odds = 1.0
                for l in extended:
                    combo_odds *= l.odds
                if c.target_multiplier and c.target_multiplier > 0:
                    target_for_step = c.target_multiplier ** ((step + 1) / leg_count)
                    proximity = abs(log(max(1.01, combo_odds)) - log(max(1.01, target_for_step)))
                else:
                    neutral = cw["neutral_anchor_base"] ** (step + 1)
                    proximity = abs(log(max(1.01, combo_odds)) - log(neutral))
                h_score = proximity - partial_quality * 0.3
                next_beam.append((extended, h_score))
        if not next_beam:
            break
        next_beam.sort(key=lambda x: x[1])
        beam = next_beam[:beam_width]

    # Final scoring of complete combos
    scored: List[Tuple[List[CandidateLeg], float, float]] = []
    for combo, _ in beam:
        if len(combo) != leg_count:
            continue
        if not combo_valid(combo, c.require_unique_events):
            continue
        s = score_combo(combo, c, leg_quality)
        scored.append((combo, s, combined_odds(combo)))
    return scored


# ---------------------------------------------------------------------------
# select_parlay
# ---------------------------------------------------------------------------


def select_parlay(candidates: List[CandidateLeg], leg_count: int, c: ConstraintSpec) -> Tuple[List[CandidateLeg], List[str]]:
    notes: List[str] = []
    if leg_count <= 0:
        return [], ["Invalid leg count."]
    if not candidates:
        return [], ["No odds candidates available."]

    # ---- Hard pre-filter: remove legs that violate non-negotiable constraints ----
    pre_filter_count = len(candidates)

    if c.require_total_corner_keys:
        candidates = [
            leg for leg in candidates
            if "corner" in leg.market_key.lower() and "total" in leg.market_key.lower()
        ]

    if c.hard_exclude_groups:
        candidates = [
            leg for leg in candidates
            if market_group_from_key(leg.market_key) not in c.hard_exclude_groups
        ]

    if c.hard_include_groups:
        candidates = [
            leg for leg in candidates
            if market_group_from_key(leg.market_key) in c.hard_include_groups
        ]

    if c.forbid_spread_keys:
        candidates = [
            leg for leg in candidates
            if "spread" not in leg.market_key.lower() and "handicap" not in leg.market_key.lower()
        ]

    filtered_out = pre_filter_count - len(candidates)
    if filtered_out > 0:
        notes.append(f"Hard-filtered {filtered_out} legs that violate market constraints ({len(candidates)} remaining).")

    if not candidates:
        return [], ["No candidates remain after applying market constraints (hard filter)."]

    # Reduce search size while keeping odds diversity.
    by_bucket: Dict[Tuple[str, str], List[CandidateLeg]] = {}
    for row in sorted(candidates, key=lambda x: x.odds):
        k = (row.event_id, row.market_key)
        by_bucket.setdefault(k, [])
        by_bucket[k].append(row)

    target_leg_anchor = None
    if c.target_multiplier and c.target_multiplier > 0:
        target_leg_anchor = c.target_multiplier ** (1.0 / max(1, leg_count))
    if target_leg_anchor is None:
        target_leg_anchor = SCORING_WEIGHTS["pool"]["default_anchor"]

    pool: List[CandidateLeg] = []
    for rows in by_bucket.values():
        n = len(rows)
        if n <= 8:
            pool.extend(rows)
            continue
        picks: List[CandidateLeg] = []
        picks.extend(rows[:2])
        picks.extend(rows[-2:])
        rows_by_anchor = sorted(rows, key=lambda x: abs(log(max(1.01, x.odds)) - log(max(1.01, target_leg_anchor))))
        picks.extend(rows_by_anchor[:4])
        seen = set()
        dedup: List[CandidateLeg] = []
        for p in picks:
            key = (p.event_id, p.market_key, p.outcome, p.point, p.bookmaker, p.odds)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(p)
        pool.extend(dedup)

    # Final cap by proximity-to-target odds + KB quality
    leg_quality: Dict[CandidateLeg, float] = {}
    for leg in pool:
        leg_quality[leg] = kb_leg_quality(leg, _leg_league(leg, c.league))

    pw = SCORING_WEIGHTS["pool"]

    def leg_rank(leg: CandidateLeg) -> float:
        proximity = abs(log(max(1.01, leg.odds)) - log(max(1.01, target_leg_anchor)))
        quality_bonus = pw["quality_bonus"] * leg_quality.get(leg, 0.0)
        return proximity - quality_bonus

    cap = int(pw["cap"])
    if c.leagues and len(c.leagues) > 1:
        cap = min(cap * 2, 400)
    pool = sorted(pool, key=leg_rank)[:cap]

    if len(pool) < leg_count:
        return [], [f"Only {len(pool)} candidate legs available for {leg_count}-leg request."]

    total_combos = _comb(len(pool), leg_count)
    threshold = SCORING_WEIGHTS["pool"]["brute_force_threshold"]
    if total_combos <= threshold:
        scored = _brute_force_combos(pool, leg_count, c, leg_quality)
    else:
        scored = _beam_search_combos(pool, leg_count, c, leg_quality)
        notes.append(f"Used beam search ({len(pool)} candidates, {leg_count} legs, ~{total_combos:,} combos avoided).")

    if not scored:
        return [], ["No valid parlay combination found."]

    if c.target_multiplier and c.target_mode in {"around", "max", "min"}:
        tgt = c.target_multiplier
        if c.target_mode == "around":
            band = AROUND_TARGET_TOLERANCE * tgt
            in_band = [row for row in scored if abs(row[2] - tgt) <= band]
            if in_band:
                best = min(in_band, key=lambda x: x[1])
                return best[0], notes
            best = min(scored, key=lambda x: abs(log(x[2]) - log(tgt)))
            notes.append(
                f"No combo within \u00b1{int(AROUND_TARGET_TOLERANCE*100)}% of target {tgt:.2f}x; returning closest at {best[2]:.2f}x."
            )
            return best[0], notes

        if c.target_mode == "max":
            feasible = [row for row in scored if row[2] <= tgt]
            if feasible:
                best = min(feasible, key=lambda x: x[1])
                return best[0], notes
            best = min(scored, key=lambda x: abs(x[2] - tgt))
            notes.append(f"No combo <= {tgt:.2f}x; returning closest at {best[2]:.2f}x.")
            return best[0], notes

        if c.target_mode == "min":
            feasible = [row for row in scored if row[2] >= tgt]
            if feasible:
                best = min(feasible, key=lambda x: x[1])
                return best[0], notes
            best = min(scored, key=lambda x: abs(x[2] - tgt))
            notes.append(f"No combo >= {tgt:.2f}x; returning closest at {best[2]:.2f}x.")
            return best[0], notes

    # --- Odds range mode ---
    if c.target_min is not None and c.target_max is not None:
        in_range = [row for row in scored if c.target_min <= row[2] <= c.target_max]
        if in_range:
            best = min(in_range, key=lambda x: x[1])
            return best[0], notes
        best = min(scored, key=lambda x: min(abs(x[2] - c.target_min), abs(x[2] - c.target_max)))
        notes.append(
            f"No combo within odds range [{c.target_min:.1f}x, {c.target_max:.1f}x]; "
            f"returning closest at {best[2]:.2f}x."
        )
        return best[0], notes

    # default: best score
    best = min(scored, key=lambda x: x[1])
    return best[0], notes
