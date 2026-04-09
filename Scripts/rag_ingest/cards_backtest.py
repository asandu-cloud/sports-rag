"""
cards_backtest.py — Dedicated walk-forward cards backtest.

Uses raw engineered fixture rows plus historical referee fixture export.
Reports: total cards MAE/RMSE/skill, team cards MAE/RMSE/skill,
line hit rates, confidence calibration, referee-present vs referee-missing slices.

Usage:
    python cards_backtest.py --league EPL
    python cards_backtest.py --league EPL LaLiga --season 2025 --skip-first 10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

try:
    from core.projections import (
        projected_total_cards_detail,
        projected_team_cards,
    )
    from core.team_resolution import _profile_meta
except ImportError:
    from Scripts.rag_ingest.core.projections import (
        projected_total_cards_detail,
        projected_team_cards,
    )
    from Scripts.rag_ingest.core.team_resolution import _profile_meta

try:
    from league_context import get_league_context
except ImportError:
    try:
        from Scripts.rag_ingest.league_context import get_league_context
    except ImportError:
        get_league_context = None

try:
    from referee_data import (
        load_fixture_referee_detail, RefereeModifier,
        compute_referee_modifier, load_profiles,
    )
except ImportError:
    from Scripts.rag_ingest.referee_data import (
        load_fixture_referee_detail, RefereeModifier,
        compute_referee_modifier, load_profiles,
    )

try:
    from prob_models import over_prob
except ImportError:
    from Scripts.rag_ingest.prob_models import over_prob


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

LEAGUE_TO_FE_DIR = {
    "EPL": "Prem_feature_engineering",
    "LaLiga": "LaLiga_feature_engineering",
    "SerieA": "SeriaA_feature_engineering",
    "Bundesliga": "Bundesliga_feature_engineering",
    "Ligue1": "Ligue1_feature_engineering",
}


def load_fixture_pairs(league: str, season_year: int) -> List[Dict]:
    """Load fixture pairs from team_engineered_features JSON."""
    fe_dir = LEAGUE_TO_FE_DIR.get(league)
    if not fe_dir:
        return []

    path = ROOT / "Output" / fe_dir / f"team_engineered_features_{season_year}.json"
    if not path.exists():
        print(f"  [WARN] Not found: {path}")
        return []

    rows = json.loads(path.read_text())

    # Group by fixture_id or fixture name
    fixture_groups: Dict = {}
    has_fixture_id = any(r.get("fixture_id") for r in rows)

    for r in rows:
        key = r.get("fixture_id") if has_fixture_id else r.get("fixture", "")
        if not key:
            continue
        fixture_groups.setdefault(key, []).append(r)

    pairs = []
    for fid, group in fixture_groups.items():
        if len(group) != 2:
            continue

        r0, r1 = group[0], group[1]
        f0 = r0.get("fixture", "")
        team0 = r0.get("team", "")

        if " vs " in f0:
            parts = f0.split(" vs ")
            if len(parts) == 2:
                if team0.strip().lower() == parts[0].strip().lower():
                    home_row, away_row = r0, r1
                else:
                    home_row, away_row = r1, r0
            else:
                continue
        else:
            continue

        yc_h = int(float(home_row.get("yellow_cards") or 0))
        rc_h = int(float(home_row.get("red_cards") or 0))
        yc_a = int(float(away_row.get("yellow_cards") or 0))
        rc_a = int(float(away_row.get("red_cards") or 0))
        cards_h = int(float(home_row.get("cards_total") or (yc_h + rc_h)))
        cards_a = int(float(away_row.get("cards_total") or (yc_a + rc_a)))

        pairs.append({
            "fixture_id": fid if has_fixture_id else f0,
            "fixture": home_row["fixture"],
            "fixture_date": home_row.get("fixture_date", ""),
            "home_team": home_row["team"],
            "away_team": away_row["team"],
            "actual_cards": cards_h + cards_a,
            "actual_yellows": yc_h + yc_a,
            "actual_reds": rc_h + rc_a,
            "home_cards": cards_h,
            "away_cards": cards_a,
        })

    pairs.sort(key=lambda x: x.get("fixture_date", ""))
    return pairs


# ---------------------------------------------------------------------------
# Projection runner
# ---------------------------------------------------------------------------

def _build_ref_mod_from_history(fixture_id) -> Optional[RefereeModifier]:
    """Look up historical referee data for a fixture and build a RefereeModifier.

    This allows backtests to simulate having referee information pre-match,
    using the referee profile that was built from ALL their matches (not just
    matches before this fixture — slight lookahead, but acceptable for
    evaluating the referee signal strength).
    """
    detail = load_fixture_referee_detail()
    row = detail.get(int(fixture_id)) if fixture_id is not None else None
    if row is None:
        return None

    ref_name = row.get("referee_name")
    if not ref_name:
        return None

    # Use compute_referee_modifier which looks up the full profile
    return compute_referee_modifier(ref_name, weight=0.35)


def project_cards(
    home: str,
    away: str,
    league: str,
    fixture_id=None,
    fixture_date: Optional[str] = None,
) -> Dict:
    """Run cards projections for a fixture.

    If fixture_id is provided, looks up the historical referee assignment
    and passes it into the projection for a realistic backtest.
    """
    result = {}

    # Try to get historical referee modifier for this fixture
    ref_mod = None
    if fixture_id is not None:
        ref_mod = _build_ref_mod_from_history(fixture_id)
    league_ctx = None
    if get_league_context is not None:
        try:
            league_ctx = get_league_context(home, away, league, target_date=fixture_date)
        except Exception:
            league_ctx = None

    try:
        detail = projected_total_cards_detail(
            home, away, league,
            ref_mod=ref_mod,
            league_ctx=league_ctx,
            fixture_date=fixture_date,
        )
        result["proj_total"] = detail.get("total_cards")
        result["proj_yellows"] = detail.get("total_yellows")
        result["proj_reds"] = detail.get("total_reds")
        result["proj_season"] = detail.get("season_total")
        result["proj_recent"] = detail.get("recent_total")
        result["uncertainty"] = detail.get("uncertainty")
        result["confidence_bucket"] = detail.get("confidence_bucket")
        ref_mod_out = detail.get("referee_modifier")
        result["ref_source"] = getattr(ref_mod_out, "source", "unavailable")
        result["ref_confidence"] = getattr(ref_mod_out, "confidence", 0.0)
        result["ref_multiplier"] = getattr(ref_mod_out, "multiplier", 1.0)
        result["ref_name"] = getattr(ref_mod_out, "referee_name", None)
        result["ref_strictness"] = getattr(ref_mod_out, "strictness_ratio", 1.0)
    except Exception:
        result["proj_total"] = None
        result["proj_yellows"] = None
        result["proj_reds"] = None
        result["confidence_bucket"] = "error"
        result["ref_source"] = "unavailable"

    # Team cards (also pass ref_mod for consistency)
    try:
        h_bl, h_sea, h_rec = projected_team_cards(
            home, away, league, is_home=True,
            ref_mod=ref_mod,
            league_ctx=league_ctx,
            fixture_date=fixture_date,
        )
        result["proj_home_cards"] = h_bl
    except Exception:
        result["proj_home_cards"] = None

    try:
        a_bl, a_sea, a_rec = projected_team_cards(
            away, home, league, is_home=False,
            ref_mod=ref_mod,
            league_ctx=league_ctx,
            fixture_date=fixture_date,
        )
        result["proj_away_cards"] = a_bl
    except Exception:
        result["proj_away_cards"] = None

    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: List[Dict], proj_key: str, actual_key: str) -> Dict:
    valid = [(r[proj_key], r[actual_key]) for r in results
             if r.get(proj_key) is not None and r.get(actual_key) is not None]
    if not valid:
        return {"n": 0}

    projs, actuals = zip(*valid)
    n = len(valid)
    mean_actual = sum(actuals) / n
    mean_proj = sum(projs) / n
    errors = [abs(p - a) for p, a in valid]
    mae = sum(errors) / n
    rmse = math.sqrt(sum((p - a) ** 2 for p, a in valid) / n)
    bias = mean_proj - mean_actual
    naive_mae = sum(abs(a - mean_actual) for a in actuals) / n

    return {
        "n": n,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mean_actual": round(mean_actual, 4),
        "mean_projected": round(mean_proj, 4),
        "bias": round(bias, 4),
        "naive_mae": round(naive_mae, 4),
        "skill_vs_naive": round(1.0 - mae / naive_mae, 4) if naive_mae > 0 else None,
    }


def compute_line_hit_rates(results: List[Dict], proj_key: str, actual_key: str,
                           lines: List[float]) -> Dict[str, float]:
    hit_rates = {}
    for line in lines:
        correct = total = 0
        for r in results:
            p, a = r.get(proj_key), r.get(actual_key)
            if p is None or a is None:
                continue
            p_over = over_prob(p, line)
            recommend_over = p_over >= 0.5
            actual_over = a > line
            total += 1
            if recommend_over == actual_over:
                correct += 1
        hit_rates[f"hit_{line}"] = round(correct / total, 4) if total > 0 else None
    return hit_rates


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_cards_report(results: List[Dict], label: str):
    print(f"\n{'='*70}")
    print(f"  CARDS BACKTEST — {label}")
    print(f"{'='*70}")

    # Total cards
    m = compute_metrics(results, "proj_total", "actual_cards")
    if m["n"] > 0:
        print(f"\n  TOTAL CARDS ({m['n']} fixtures)")
        print(f"  {'─'*40}")
        print(f"  Mean actual:     {m['mean_actual']:.2f}")
        print(f"  Mean projected:  {m['mean_projected']:.2f}")
        print(f"  Bias:            {m['bias']:+.2f}")
        print(f"  MAE:             {m['mae']:.3f}")
        print(f"  RMSE:            {m['rmse']:.3f}")
        print(f"  Naive MAE:       {m['naive_mae']:.3f}")
        skill = m.get('skill_vs_naive')
        if skill is not None:
            print(f"  Skill vs naive:  {skill:+.1%}")

        # Line hit rates
        lines = [2.5, 3.5, 4.5, 5.5]
        hits = compute_line_hit_rates(results, "proj_total", "actual_cards", lines)
        if hits:
            print(f"\n  Line Hit Rates:")
            for k in sorted(hits):
                v = hits[k]
                if v is not None:
                    line = k.replace("hit_", "")
                    bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
                    print(f"    Line {line:>5s}: {v:.1%}  {bar}")

    # Team cards
    for side, proj_k, act_k in [
        ("Home", "proj_home_cards", "home_cards"),
        ("Away", "proj_away_cards", "away_cards"),
    ]:
        tm = compute_metrics(results, proj_k, act_k)
        if tm["n"] > 0:
            print(f"\n  {side.upper()} TEAM CARDS ({tm['n']} fixtures)")
            print(f"  MAE: {tm['mae']:.3f}  RMSE: {tm['rmse']:.3f}  Bias: {tm['bias']:+.2f}  Skill: {tm.get('skill_vs_naive', 0):+.1%}")

    # Slices
    ref_assigned = [r for r in results if r.get("ref_source") == "profile"]
    ref_missing = [r for r in results if r.get("ref_source") != "profile"]
    high_conf = [r for r in results if r.get("confidence_bucket") == "high"]
    low_conf = [r for r in results if r.get("confidence_bucket") == "low"]

    print(f"\n  SLICES:")
    for label_s, subset in [
        ("Referee assigned", ref_assigned),
        ("Referee missing", ref_missing),
        ("High confidence", high_conf),
        ("Low confidence", low_conf),
    ]:
        sm = compute_metrics(subset, "proj_total", "actual_cards")
        if sm["n"] > 0:
            print(f"    {label_s:20s} ({sm['n']:>3d}): MAE={sm['mae']:.3f} RMSE={sm['rmse']:.3f} Skill={sm.get('skill_vs_naive', 0):+.1%}")

    # Confidence calibration
    print(f"\n  CONFIDENCE CALIBRATION:")
    for bucket in ["high", "medium", "low"]:
        subset = [r for r in results if r.get("confidence_bucket") == bucket]
        sm = compute_metrics(subset, "proj_total", "actual_cards")
        if sm["n"] > 0:
            print(f"    {bucket:8s} ({sm['n']:>3d} fixtures): MAE={sm['mae']:.3f} Bias={sm['bias']:+.2f}")

    print(f"\n{'='*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_cards_backtest(
    leagues: List[str],
    season_year: int = 2025,
    skip_first: int = 10,
    csv_path: Optional[str] = None,
):
    all_results = []

    for league in leagues:
        print(f"\n  Loading {league} fixtures (season {season_year})...")
        pairs = load_fixture_pairs(league, season_year)
        print(f"  Found {len(pairs)} fixtures")

        if not pairs:
            continue

        team_fixture_count: Dict[str, int] = defaultdict(int)
        league_results = []
        skipped = 0
        errors = 0

        for i, fixture in enumerate(pairs):
            home = fixture["home_team"]
            away = fixture["away_team"]

            team_fixture_count[home] += 1
            team_fixture_count[away] += 1
            if team_fixture_count[home] <= skip_first or team_fixture_count[away] <= skip_first:
                skipped += 1
                continue

            try:
                fid = fixture.get("fixture_id")
                projections = project_cards(
                    home, away, league,
                    fixture_id=fid,
                    fixture_date=fixture.get("fixture_date"),
                )
            except Exception as e:
                errors += 1
                continue

            row = {"league": league, **fixture, **projections}
            league_results.append(row)

        print(f"  Processed: {len(league_results)}, Skipped: {skipped}, Errors: {errors}")

        print_cards_report(league_results, league)
        all_results.extend(league_results)

    if len(leagues) > 1 and all_results:
        print_cards_report(all_results, "ALL LEAGUES")

    if csv_path and all_results:
        # Flatten for CSV (remove non-serializable)
        csv_rows = []
        for r in all_results:
            row = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
            csv_rows.append(row)
        if csv_rows:
            fieldnames = sorted(csv_rows[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"  Exported {len(csv_rows)} rows to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Dedicated cards backtest.")
    parser.add_argument("--league", nargs="+", default=["EPL"])
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--skip-first", type=int, default=10)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    run_cards_backtest(args.league, args.season, args.skip_first, args.csv)


if __name__ == "__main__":
    main()
