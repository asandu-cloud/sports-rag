"""
league_eval.py — Bet-centric evaluation across leagues.

Provides edge-bucket analysis, confidence-bucket calibration,
short-rest vs normal slices, and cross-league comparison tables.

Usage:
    python league_eval.py --league EPL LaLiga
    python league_eval.py --league EPL --stat cards --csv cards_eval.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from backtest import load_fixture_pairs, project_fixture, compute_metrics, over_prob
except ImportError:
    from Scripts.rag_ingest.backtest import load_fixture_pairs, project_fixture, compute_metrics, over_prob

try:
    from league_context import get_league_context
except ImportError:
    try:
        from Scripts.rag_ingest.league_context import get_league_context
    except ImportError:
        get_league_context = None


# ---------------------------------------------------------------------------
# Edge-bucket analysis
# ---------------------------------------------------------------------------

def compute_edge_buckets(results: List[Dict], stat: str,
                         lines: List[float]) -> List[Dict]:
    """Evaluate hit rate by model-edge bucket.

    For each common line, groups fixtures by the size of the model edge
    (projected - line for over, line - projected for under) and reports
    hit rate per bucket.
    """
    proj_key = f"proj_{stat}"
    actual_key = f"actual_{stat}"
    buckets = []

    edge_ranges = [
        (0.0, 0.25, "0.0-0.25"),
        (0.25, 0.50, "0.25-0.50"),
        (0.50, 1.0, "0.50-1.0"),
        (1.0, 2.0, "1.0-2.0"),
        (2.0, 999, "2.0+"),
    ]

    for line in lines:
        for lo, hi, label in edge_ranges:
            correct = total = 0
            for r in results:
                p, a = r.get(proj_key), r.get(actual_key)
                if p is None or a is None:
                    continue

                edge = abs(p - line)
                if edge < lo or edge >= hi:
                    continue

                recommend_over = p > line
                actual_over = a > line
                total += 1
                if recommend_over == actual_over:
                    correct += 1

            if total >= 5:
                buckets.append({
                    "stat": stat,
                    "line": line,
                    "edge_bucket": label,
                    "n": total,
                    "hit_rate": round(correct / total, 4),
                })

    return buckets


# ---------------------------------------------------------------------------
# Rest-day slicing
# ---------------------------------------------------------------------------

def _enrich_with_rest(results: List[Dict]) -> List[Dict]:
    """Add rest-day and congestion info to result rows using league_context."""
    if get_league_context is None:
        return results

    for r in results:
        try:
            ctx = get_league_context(r["home_team"], r["away_team"], r["league"],
                                     target_date=r.get("fixture_date"))
            r["home_rest_days"] = ctx.home_rest_days
            r["away_rest_days"] = ctx.away_rest_days
            r["home_short_rest"] = ctx.home_short_rest
            r["away_short_rest"] = ctx.away_short_rest
            r["any_short_rest"] = ctx.home_short_rest or ctx.away_short_rest
        except Exception:
            r["any_short_rest"] = None
    return results


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

def print_league_eval(results: List[Dict], league: str):
    print(f"\n{'='*70}")
    print(f"  BET-CENTRIC EVAL — {league}")
    print(f"{'='*70}")

    stat_lines = {
        "goals": [1.5, 2.5, 3.5],
        "corners": [8.5, 9.5, 10.5],
        "cards": [2.5, 3.5, 4.5, 5.5],
        "sot": [7.5, 8.5, 9.5],
    }

    for stat in ["goals", "corners", "cards", "sot"]:
        m = compute_metrics(results, stat)
        if m.get("n", 0) == 0:
            continue

        print(f"\n  {stat.upper()} ({m['n']} fixtures)")
        print(f"  MAE: {m['mae']:.3f}  RMSE: {m['rmse']:.3f}  Bias: {m['bias']:+.2f}  Skill: {m.get('skill_vs_naive', 0):+.1%}")

        # Edge buckets
        lines = stat_lines.get(stat, [])
        edge_b = compute_edge_buckets(results, stat, lines)
        if edge_b:
            print(f"\n  Edge Buckets (hit rate by edge size):")
            for b in edge_b:
                bar = "█" * int(b["hit_rate"] * 20) + "░" * (20 - int(b["hit_rate"] * 20))
                print(f"    Line {b['line']:>5.1f} | Edge {b['edge_bucket']:>8s} | "
                      f"n={b['n']:>3d} | Hit: {b['hit_rate']:.1%}  {bar}")

    # Rest-day slices
    short_rest = [r for r in results if r.get("any_short_rest") is True]
    normal_rest = [r for r in results if r.get("any_short_rest") is False]
    if short_rest and normal_rest:
        print(f"\n  REST-DAY SLICES:")
        for stat in ["goals", "corners", "cards"]:
            ms = compute_metrics(short_rest, stat)
            mn = compute_metrics(normal_rest, stat)
            if ms.get("n", 0) >= 5 and mn.get("n", 0) >= 5:
                print(f"    {stat:>8s} | Short rest ({ms['n']:>3d}): MAE={ms['mae']:.3f} | "
                      f"Normal ({mn['n']:>3d}): MAE={mn['mae']:.3f}")

    print(f"\n{'='*70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_league_eval(
    leagues: List[str],
    season_year: int = 2025,
    skip_first: int = 10,
    csv_path: Optional[str] = None,
):
    all_results = []

    for league in leagues:
        print(f"\n  Loading {league} fixtures...")
        pairs = load_fixture_pairs(league, season_year)
        if not pairs:
            continue

        team_count: Dict[str, int] = defaultdict(int)
        league_results = []

        for fixture in pairs:
            home, away = fixture["home_team"], fixture["away_team"]
            team_count[home] += 1
            team_count[away] += 1
            if team_count[home] <= skip_first or team_count[away] <= skip_first:
                continue

            try:
                projections = project_fixture(home, away, league)
            except Exception:
                continue

            row = {"league": league, **fixture, **projections}
            league_results.append(row)

        print(f"  {len(league_results)} fixtures evaluated")

        # Enrich with rest data
        league_results = _enrich_with_rest(league_results)

        print_league_eval(league_results, league)
        all_results.extend(league_results)

    if len(leagues) > 1 and all_results:
        print_league_eval(all_results, "ALL LEAGUES")

    if csv_path and all_results:
        csv_rows = [{k: v for k, v in r.items() if not isinstance(v, (dict, list))}
                    for r in all_results]
        if csv_rows:
            fn = sorted(csv_rows[0].keys())
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fn)
                w.writeheader()
                w.writerows(csv_rows)
            print(f"  Exported {len(csv_rows)} rows to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Bet-centric league evaluation.")
    parser.add_argument("--league", nargs="+", default=["EPL"])
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--skip-first", type=int, default=10)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()
    run_league_eval(args.league, args.season, args.skip_first, args.csv)


if __name__ == "__main__":
    main()
