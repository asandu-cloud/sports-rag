#!/usr/bin/env python3
"""
Walk-forward backtest for team-specific corners and cards projections.

This does not rely on current Chroma profiles, so it avoids lookahead leakage.
It rebuilds team context from the engineered fixture rows that would have been
available before each match and compares:

- baseline formulas that mirror the old teamline logic
- improved formulas used by the patched projection layer

Usage:
    python teamline_backtest.py
    python teamline_backtest.py --league EPL
    python teamline_backtest.py --season 2025
    python teamline_backtest.py --skip-first 8
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEAGUE_DATA_DIRS = {
    "EPL": PROJECT_ROOT / "Output" / "Prem_feature_engineering",
    "LaLiga": PROJECT_ROOT / "Output" / "LaLiga_feature_engineering",
    "SerieA": PROJECT_ROOT / "Output" / "SeriaA_feature_engineering",
    "Bundesliga": PROJECT_ROOT / "Output" / "Bundesliga_feature_engineering",
    "Ligue1": PROJECT_ROOT / "Output" / "Ligue1_feature_engineering",
}


def load_fixture_pairs(league: str, season_year: int) -> List[Dict]:
    path = LEAGUE_DATA_DIRS[league] / f"team_engineered_features_{season_year}.json"
    rows = json.loads(path.read_text())
    has_fixture_id = "fixture_id" in rows[0]
    by_fixture: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        key = row.get("fixture_id") if has_fixture_id else row["fixture"]
        by_fixture[str(key)].append(row)

    pairs = []
    for fixture_id, frows in by_fixture.items():
        if len(frows) != 2:
            continue
        r0, r1 = frows
        if "home_team" in r0:
            if r0["team"] == r0["home_team"]:
                home_row, away_row = r0, r1
            elif r1["team"] == r1["home_team"]:
                home_row, away_row = r1, r0
            else:
                continue
        else:
            parts = str(r0["fixture"]).split(" vs ")
            if len(parts) != 2:
                continue
            home_name = parts[0].strip()
            if r0["team"] == home_name:
                home_row, away_row = r0, r1
            elif r1["team"] == home_name:
                home_row, away_row = r1, r0
            else:
                continue

        pairs.append({
            "fixture_id": fixture_id,
            "fixture_date": home_row.get("fixture_date", ""),
            "home_row": home_row,
            "away_row": away_row,
        })

    pairs.sort(key=lambda x: x["fixture_date"])
    return pairs


def _avg(history: List[Dict], key: str, venue: Optional[str] = None) -> Optional[float]:
    vals = [h.get(key) for h in history if (venue is None or h.get("venue") == venue) and h.get(key) is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _weighted_recent(
    history: List[Dict],
    key: str,
    venue: Optional[str] = None,
    last_n: int = 6,
    alpha: float = 0.85,
    min_count: int = 1,
) -> Optional[float]:
    vals = []
    for item in reversed(history):
        if venue is not None and item.get("venue") != venue:
            continue
        value = item.get(key)
        if value is not None:
            vals.append(value)
        if len(vals) >= last_n:
            break
    if len(vals) < min_count:
        return None
    weights = [alpha ** i for i in range(len(vals))]
    return sum(w * v for w, v in zip(weights, vals)) / sum(weights)


def _blend(a: Optional[float], b: Optional[float], w_a: float) -> Optional[float]:
    if a is not None and b is not None:
        return w_a * a + (1.0 - w_a) * b
    return a if a is not None else b


def _venue_blend(venue_value: Optional[float], overall_value: Optional[float], venue_weight: float = 0.5) -> Optional[float]:
    if venue_value is not None and overall_value is not None:
        return venue_weight * venue_value + (1.0 - venue_weight) * overall_value
    return venue_value if venue_value is not None else overall_value


def _weighted_sum(components: List[Tuple[float, Optional[float]]]) -> Optional[float]:
    total = 0.0
    seen = False
    for weight, value in components:
        if value is None or weight == 0.0:
            continue
        total += weight * value
        seen = True
    return total if seen else None


def _weighted_component_blend(components: List[Tuple[Optional[float], float]]) -> Optional[float]:
    total_weight = 0.0
    blended = 0.0
    for value, weight in components:
        if value is None or weight == 0.0:
            continue
        blended += weight * value
        total_weight += weight
    if total_weight <= 0:
        return None
    return blended / total_weight


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _share_from_scores(
    team_score: Optional[float],
    opp_score: Optional[float],
    *,
    shrink: float,
    floor: float,
    ceiling: float,
) -> Optional[float]:
    if team_score is None and opp_score is None:
        return None
    team_base = max(team_score if team_score is not None else 0.0, 0.05)
    opp_base = max(opp_score if opp_score is not None else 0.0, 0.05)
    raw_share = team_base / (team_base + opp_base)
    shrunk = 0.5 + (raw_share - 0.5) * (1.0 - shrink)
    return _clamp(shrunk, floor, ceiling)


def _weighted_slope(
    history: List[Dict],
    key: str,
    venue: Optional[str] = None,
    last_n: int = 6,
) -> Optional[float]:
    vals: List[float] = []
    for item in reversed(history):
        if venue is not None and item.get("venue") != venue:
            continue
        value = item.get(key)
        if value is not None:
            vals.append(float(value))
        if len(vals) >= last_n:
            break
    if len(vals) < 3:
        return None
    xs = list(range(len(vals)))
    x_mean = sum(xs) / len(xs)
    y_mean = sum(vals) / len(vals)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, vals))
    den = sum((x - x_mean) ** 2 for x in xs)
    return num / den if den else 0.0


def _metrics(rows: List[Dict], pred_key: str, actual_key: str) -> Dict:
    vals = [(r.get(pred_key), r.get(actual_key)) for r in rows if r.get(pred_key) is not None]
    n = len(vals)
    if n == 0:
        return {"n": 0}
    mae = sum(abs(p - a) for p, a in vals) / n
    rmse = math.sqrt(sum((p - a) ** 2 for p, a in vals) / n)
    mean_actual = sum(a for _, a in vals) / n
    mean_pred = sum(p for p, _ in vals) / n
    naive_mae = sum(abs(a - mean_actual) for _, a in vals) / n
    return {
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "bias": mean_pred - mean_actual,
        "skill_vs_naive": 1.0 - mae / naive_mae if naive_mae else None,
    }


def _line_hit(rows: List[Dict], pred_key: str, actual_key: str, line: float) -> float:
    vals = [(r.get(pred_key), r.get(actual_key)) for r in rows if r.get(pred_key) is not None]
    if not vals:
        return 0.0
    hits = sum(((p > line) == (a > line)) for p, a in vals)
    return hits / len(vals)


def _project_rows(leagues: Iterable[str], seasons: Iterable[int], skip_first: int) -> List[Dict]:
    rows: List[Dict] = []

    for league in leagues:
        for season in seasons:
            history: Dict[str, List[Dict]] = defaultdict(list)
            for pair in load_fixture_pairs(league, season):
                home_row = pair["home_row"]
                away_row = pair["away_row"]
                fixture_rows: List[Dict] = []

                for row, opp_row, venue, opp_venue in [
                    (home_row, away_row, "home", "away"),
                    (away_row, home_row, "away", "home"),
                ]:
                    team = row["team"]
                    opp = opp_row["team"]
                    team_hist = history[team]
                    opp_hist = history[opp]
                    if len(team_hist) < skip_first or len(opp_hist) < skip_first:
                        continue

                    corners_for = _venue_blend(
                        _avg(team_hist, "corners_for", venue=venue),
                        _avg(team_hist, "corners_for"),
                        0.5,
                    )
                    corners_opp_allow = _avg(opp_hist, "corners_against")
                    corners_season = _blend(corners_for, corners_opp_allow, 0.6)
                    corners_recent_team = _weighted_recent(team_hist, "corners_for")
                    corners_recent_team_venue = (
                        _weighted_recent(team_hist, "corners_for", venue=venue, min_count=2)
                        or corners_recent_team
                    )
                    corners_recent_opp_allow = (
                        _weighted_recent(opp_hist, "corners_against", venue=opp_venue, min_count=2)
                        or _weighted_recent(opp_hist, "corners_against")
                    )
                    corners_recent_matchup = _blend(corners_recent_team_venue, corners_recent_opp_allow, 0.5)
                    corners_style = _weighted_sum([
                        (0.01, _weighted_recent(team_hist, "shots_for", venue=venue, min_count=2) or _weighted_recent(team_hist, "shots_for")),
                        (0.05, _weighted_recent(team_hist, "sot_for", venue=venue, min_count=2) or _weighted_recent(team_hist, "sot_for")),
                        (1.50, _weighted_recent(team_hist, "control", venue=venue, min_count=2) or _weighted_recent(team_hist, "control")),
                        (0.85, _avg(team_hist, "dominance")),
                        (1.60, _weighted_recent(team_hist, "possession", venue=venue, min_count=2) or _weighted_recent(team_hist, "possession")),
                        (0.23, _weighted_recent(opp_hist, "sot_against", venue=opp_venue, min_count=2) or _weighted_recent(opp_hist, "sot_against")),
                    ])
                    corners_redesigned = _weighted_component_blend([
                        (corners_season, 0.43),
                        (corners_recent_matchup, 0.34),
                        (corners_style, 0.23),
                    ])
                    corners_trend = _weighted_slope(team_hist, "corners_for", venue=venue, last_n=6)
                    if corners_redesigned is not None and corners_trend is not None:
                        corners_redesigned += _clamp(0.06 * corners_trend, -0.25, 0.25)

                    cards_for = _venue_blend(
                        _avg(team_hist, "cards", venue=venue),
                        _avg(team_hist, "cards"),
                        0.5,
                    )
                    opp_induced_overall = _avg(opp_hist, "cards_induced")
                    opp_induced_venue = _avg(opp_hist, "cards_induced", venue=opp_venue)
                    agg = _avg(team_hist, "aggression")
                    cards_season_old = _blend(cards_for, opp_induced_overall, 0.7)
                    if cards_season_old is not None and agg is not None:
                        cards_season_old += 0.05 * (agg * 2.0)
                    cards_season_new = _blend(cards_for, opp_induced_venue, 0.7)
                    cards_recent_team = _weighted_recent(team_hist, "cards")
                    cards_recent_opp_induced = _weighted_recent(opp_hist, "cards_induced")
                    cards_recent_matchup = _blend(cards_recent_team, cards_recent_opp_induced, 0.5)
                    cards_stabilizer = _weighted_component_blend([
                        (cards_season_new, 0.65),
                        (cards_recent_matchup, 0.35),
                    ])

                    own_fouls = _venue_blend(
                        _avg(team_hist, "fouls", venue=venue),
                        _avg(team_hist, "fouls"),
                        0.5,
                    )
                    season_share_score = _weighted_sum([
                        (0.34, cards_for),
                        (0.43, opp_induced_venue if opp_induced_venue is not None else opp_induced_overall),
                        (0.06, own_fouls),
                        (0.02, agg),
                        (0.05, _avg(team_hist, "cards_per_foul")),
                        (0.97, _avg(opp_hist, "control")),
                    ])
                    recent_share_score = _weighted_sum([
                        (0.41, cards_recent_team),
                        (0.59, cards_recent_opp_induced),
                        (0.10, _weighted_recent(team_hist, "fouls")),
                        (0.46, _weighted_recent(team_hist, "aggression")),
                        (
                            1.47,
                            1.0 - _weighted_recent(team_hist, "control")
                            if _weighted_recent(team_hist, "control") is not None else None,
                        ),
                        (0.84, _weighted_recent(opp_hist, "control")),
                    ])

                    fixture_rows.append({
                        "league": league,
                        "season": season,
                        "team": team,
                        "venue": venue,
                        "actual_corners": float(row.get("corners") or 0.0),
                        "actual_cards": float(row.get("cards_total") or 0.0),
                        "corners_baseline": _blend(corners_season, corners_recent_team, 0.75),
                        "corners_improved": _blend(corners_season, corners_recent_matchup, 0.70),
                        "corners_redesigned": corners_redesigned,
                        "cards_baseline": _blend(cards_season_old, cards_recent_team, 0.75),
                        "cards_improved": _blend(cards_season_new, cards_recent_matchup, 0.65),
                        "cards_stabilizer": cards_stabilizer,
                        "cards_season_share_score": season_share_score,
                        "cards_recent_share_score": recent_share_score,
                    })

                if len(fixture_rows) == 2:
                    home_entry, away_entry = fixture_rows
                    season_total = (home_entry["cards_improved"] or 0.0) + (away_entry["cards_improved"] or 0.0)
                    recent_total = None
                    if home_entry["cards_stabilizer"] is not None and away_entry["cards_stabilizer"] is not None:
                        recent_total = home_entry["cards_stabilizer"] + away_entry["cards_stabilizer"]
                    total_cards = _blend(season_total, recent_total, 0.65)
                    if total_cards is not None:
                        season_share = _share_from_scores(
                            home_entry["cards_season_share_score"],
                            away_entry["cards_season_share_score"],
                            shrink=0.06,
                            floor=0.26,
                            ceiling=0.71,
                        )
                        recent_share = _share_from_scores(
                            home_entry["cards_recent_share_score"],
                            away_entry["cards_recent_share_score"],
                            shrink=0.06,
                            floor=0.26,
                            ceiling=0.71,
                        )
                        share_blended = _weighted_component_blend([
                            (season_share, 0.68),
                            (recent_share, 0.32),
                        ])
                        if share_blended is not None:
                            home_share_anchor = total_cards * share_blended
                            away_share_anchor = total_cards * (1.0 - share_blended)
                            home_entry["cards_redesigned"] = _weighted_component_blend([
                                (home_share_anchor, 0.50),
                                (home_entry["cards_stabilizer"], 0.50),
                            ])
                            away_entry["cards_redesigned"] = _weighted_component_blend([
                                (away_share_anchor, 0.50),
                                (away_entry["cards_stabilizer"], 0.50),
                            ])
                    rows.extend(fixture_rows)
                else:
                    rows.extend(fixture_rows)

                history[home_row["team"]].append({
                    "venue": "home",
                    "corners_for": float(home_row.get("corners") or 0.0),
                    "corners_against": float(away_row.get("corners") or 0.0),
                    "shots_for": float(home_row.get("shots_total") or 0.0),
                    "sot_for": float(home_row.get("shots_on") or 0.0),
                    "sot_against": float(away_row.get("shots_on") or 0.0),
                    "control": float(home_row.get("control_index") or 0.0),
                    "dominance": float(home_row.get("dominance_index") or 0.0),
                    "possession": float(home_row.get("possession") or 0.0),
                    "cards": float(home_row.get("cards_total") or 0.0),
                    "cards_induced": float(away_row.get("cards_total") or 0.0),
                    "fouls": float(home_row.get("fouls_per_90_team") or home_row.get("fouls_committed") or 0.0),
                    "aggression": float(home_row.get("aggression_index_norm") or 0.0),
                    "cards_per_foul": float(home_row.get("cards_per_foul_team") or 0.0),
                })
                history[away_row["team"]].append({
                    "venue": "away",
                    "corners_for": float(away_row.get("corners") or 0.0),
                    "corners_against": float(home_row.get("corners") or 0.0),
                    "shots_for": float(away_row.get("shots_total") or 0.0),
                    "sot_for": float(away_row.get("shots_on") or 0.0),
                    "sot_against": float(home_row.get("shots_on") or 0.0),
                    "control": float(away_row.get("control_index") or 0.0),
                    "dominance": float(away_row.get("dominance_index") or 0.0),
                    "possession": float(away_row.get("possession") or 0.0),
                    "cards": float(away_row.get("cards_total") or 0.0),
                    "cards_induced": float(home_row.get("cards_total") or 0.0),
                    "fouls": float(away_row.get("fouls_per_90_team") or away_row.get("fouls_committed") or 0.0),
                    "aggression": float(away_row.get("aggression_index_norm") or 0.0),
                    "cards_per_foul": float(away_row.get("cards_per_foul_team") or 0.0),
                })

    return rows


def _profile_from_history(history: List[Dict]) -> Dict:
    """Build the profile metadata shape consumed by core.projections."""
    return {
        "corners_pm": _avg(history, "corners_for"),
        "corners_home_pm": _avg(history, "corners_for", venue="home"),
        "corners_away_pm": _avg(history, "corners_for", venue="away"),
        "corners_against_pm": _avg(history, "corners_against"),
        "corners_against_home_pm": _avg(history, "corners_against", venue="home"),
        "corners_against_away_pm": _avg(history, "corners_against", venue="away"),
        "shots_for_pm": _avg(history, "shots_for"),
        "sot_for_pm": _avg(history, "sot_for"),
        "sot_against_pm": _avg(history, "sot_against"),
        "control_index": _avg(history, "control"),
        "dominance_index": _avg(history, "dominance"),
        "possession": _avg(history, "possession"),
    }


def _recent_from_history(history: List[Dict], venue: Optional[str], last_n: int) -> Dict:
    """Build the recent-stat shape consumed by core.projections."""
    sample_count = sum(1 for item in history if venue is None or item.get("venue") == venue)
    return {
        "n": min(sample_count, last_n),
        "corners_for_avg": _weighted_recent(history, "corners_for", venue=venue, last_n=last_n),
        "corners_against_avg": _weighted_recent(history, "corners_against", venue=venue, last_n=last_n),
        "shots_for_avg": _weighted_recent(history, "shots_for", venue=venue, last_n=last_n),
        "sot_for_avg": _weighted_recent(history, "sot_for", venue=venue, last_n=last_n),
        "control_avg": _weighted_recent(history, "control", venue=venue, last_n=last_n),
        "possession_avg": _weighted_recent(history, "possession", venue=venue, last_n=last_n),
        "corners_for_slope": _weighted_slope(history, "corners_for", venue=venue, last_n=last_n),
    }


def _append_fixture_history(history: Dict[str, List[Dict]], home_row: Dict, away_row: Dict) -> None:
    history[home_row["team"]].append({
        "venue": "home",
        "corners_for": float(home_row.get("corners") or 0.0),
        "corners_against": float(away_row.get("corners") or 0.0),
        "shots_for": float(home_row.get("shots_total") or 0.0),
        "sot_for": float(home_row.get("shots_on") or 0.0),
        "sot_against": float(away_row.get("shots_on") or 0.0),
        "control": float(home_row.get("control_index") or 0.0),
        "dominance": float(home_row.get("dominance_index") or 0.0),
        "possession": float(home_row.get("possession") or 0.0),
        "cards": float(home_row.get("cards_total") or 0.0),
        "cards_induced": float(away_row.get("cards_total") or 0.0),
        "fouls": float(home_row.get("fouls_per_90_team") or home_row.get("fouls_committed") or 0.0),
        "aggression": float(home_row.get("aggression_index_norm") or 0.0),
        "cards_per_foul": float(home_row.get("cards_per_foul_team") or 0.0),
    })
    history[away_row["team"]].append({
        "venue": "away",
        "corners_for": float(away_row.get("corners") or 0.0),
        "corners_against": float(home_row.get("corners") or 0.0),
        "shots_for": float(away_row.get("shots_total") or 0.0),
        "sot_for": float(away_row.get("shots_on") or 0.0),
        "sot_against": float(home_row.get("shots_on") or 0.0),
        "control": float(away_row.get("control_index") or 0.0),
        "dominance": float(away_row.get("dominance_index") or 0.0),
        "possession": float(away_row.get("possession") or 0.0),
        "cards": float(away_row.get("cards_total") or 0.0),
        "cards_induced": float(home_row.get("cards_total") or 0.0),
        "fouls": float(away_row.get("fouls_per_90_team") or away_row.get("fouls_committed") or 0.0),
        "aggression": float(away_row.get("aggression_index_norm") or 0.0),
        "cards_per_foul": float(away_row.get("cards_per_foul_team") or 0.0),
    })


def _project_rows_current_core(leagues: Iterable[str], seasons: Iterable[int], skip_first: int) -> List[Dict]:
    """Walk-forward backtest of the current production team-corner formula.

    It patches core.projections' Chroma resolvers with fixture-history snapshots,
    so the formula is production-faithful while the inputs remain pre-match only.
    """
    from core import projections

    rows: List[Dict] = []
    neutral_ctx = SimpleNamespace(adjustments={})

    for league in leagues:
        for season in seasons:
            history: Dict[str, List[Dict]] = defaultdict(list)
            for pair in load_fixture_pairs(league, season):
                home_row = pair["home_row"]
                away_row = pair["away_row"]
                home = home_row["team"]
                away = away_row["team"]

                if len(history[home]) >= skip_first and len(history[away]) >= skip_first:
                    def profile_side_effect(team: str, league_arg: str) -> Dict:
                        return _profile_from_history(history.get(team, []))

                    def recent_side_effect(
                        team: str,
                        league_arg: str,
                        last_n: int = 6,
                        venue: Optional[str] = None,
                        target_date: Optional[str] = None,
                    ) -> Dict:
                        return _recent_from_history(history.get(team, []), venue, last_n)

                    with mock.patch.object(projections, "_profile_meta", side_effect=profile_side_effect), \
                         mock.patch.object(projections, "_recent_stats", side_effect=recent_side_effect), \
                         mock.patch.object(projections, "_ml_blend_weight", return_value=0.0):
                        for row, opp_row, is_home in [
                            (home_row, away_row, True),
                            (away_row, home_row, False),
                        ]:
                            team = row["team"]
                            opp = opp_row["team"]
                            detail = projections.projected_team_corners_detail(
                                team,
                                opp,
                                league,
                                is_home=is_home,
                                league_ctx=neutral_ctx,
                                fixture_date=pair.get("fixture_date"),
                            )
                            rows.append({
                                "league": league,
                                "season": season,
                                "team": team,
                                "venue": "home" if is_home else "away",
                                "actual_corners": float(row.get("corners") or 0.0),
                                "corners_current_core": detail.get("final"),
                                "match_total": detail.get("match_total"),
                                "season_share": detail.get("season_share"),
                                "recent_share": detail.get("recent_share"),
                                "share_anchor": detail.get("share_anchor"),
                                "season_direct": detail.get("season_direct"),
                                "recent_direct": detail.get("recent_direct"),
                                "stabilizer": detail.get("stabilizer"),
                            })

                _append_fixture_history(history, home_row, away_row)

    return rows


def _print_metrics(rows: List[Dict], title: str) -> None:
    print(f"\n{title}")
    for label, pred_key, actual_key in [
        ("Corners current", "corners_current_core", "actual_corners"),
        ("Corners baseline", "corners_baseline", "actual_corners"),
        ("Corners improved", "corners_improved", "actual_corners"),
        ("Corners redesigned", "corners_redesigned", "actual_corners"),
        ("Cards baseline", "cards_baseline", "actual_cards"),
        ("Cards improved", "cards_improved", "actual_cards"),
        ("Cards redesigned", "cards_redesigned", "actual_cards"),
    ]:
        m = _metrics(rows, pred_key, actual_key)
        if not m.get("n"):
            continue
        print(
            f"{label:17s} n={m['n']:4d}  mae={m['mae']:.3f}  rmse={m['rmse']:.3f}  "
            f"bias={m['bias']:+.3f}  skill={m['skill_vs_naive']:+.1%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest for teamlines")
    parser.add_argument("--league", type=str, default=None, help="Single league to test")
    parser.add_argument("--season", type=int, default=None, help="Single season year suffix to test")
    parser.add_argument("--skip-first", type=int, default=8, help="Skip first N fixtures per team")
    parser.add_argument(
        "--current-core",
        action="store_true",
        help="Backtest the current core projected_team_corners() formula with walk-forward inputs",
    )
    args = parser.parse_args()

    leagues = [args.league] if args.league else list(LEAGUE_DATA_DIRS.keys())
    seasons = [args.season] if args.season else [2024, 2025]
    rows = (
        _project_rows_current_core(leagues, seasons, args.skip_first)
        if args.current_core else
        _project_rows(leagues, seasons, args.skip_first)
    )

    if args.league or args.season:
        _print_metrics(rows, "Selected slice")
    else:
        for league in leagues:
            lg_rows = [r for r in rows if r["league"] == league]
            _print_metrics(lg_rows, league)
        _print_metrics(rows, "Aggregate")

    print("\nLine hit rates")
    has_current = any(r.get("corners_current_core") is not None for r in rows)
    for line in [2.5, 3.5, 4.5, 5.5]:
        old_hit = _line_hit(rows, "corners_baseline", "actual_corners", line)
        new_hit = _line_hit(rows, "corners_improved", "actual_corners", line)
        redesign_hit = _line_hit(rows, "corners_redesigned", "actual_corners", line)
        current_hit = _line_hit(rows, "corners_current_core", "actual_corners", line)
        if has_current:
            print(f"Corners {line:.1f}: current={current_hit:.1%}")
        else:
            print(f"Corners {line:.1f}: baseline={old_hit:.1%} improved={new_hit:.1%} redesigned={redesign_hit:.1%}")
    if any(r.get("cards_baseline") is not None for r in rows):
        for line in [0.5, 1.5, 2.5, 3.5]:
            old_hit = _line_hit(rows, "cards_baseline", "actual_cards", line)
            new_hit = _line_hit(rows, "cards_improved", "actual_cards", line)
            redesign_hit = _line_hit(rows, "cards_redesigned", "actual_cards", line)
            print(f"Cards   {line:.1f}: baseline={old_hit:.1%} improved={new_hit:.1%} redesigned={redesign_hit:.1%}")


if __name__ == "__main__":
    main()
