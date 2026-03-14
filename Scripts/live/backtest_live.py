#!/usr/bin/env python3
"""Backtesting harness for the live projection engine.

Replays historical fixtures to calibrate prior_strength values and measure
projection accuracy at various match checkpoints (15', 30', 45', 60', 75').

For each fixture, simulates minute-by-minute observation by distributing
actual match events across time windows (using empirical time distributions),
then calls bayesian_update_lambda() at each checkpoint and measures the
error between projected remaining count and actual remaining count.

Tests multiple prior_strength values and reports which produces the lowest
error for each stat type.

Usage:
    cd Scripts/live
    python backtest_live.py                           # all leagues, default params
    python backtest_live.py --league EPL              # single league
    python backtest_live.py --stat goals              # single stat
    python backtest_live.py --csv live_backtest.csv   # export results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_ROOT / "Scripts" / "rag_ingest"))

from live_engine import bayesian_update_lambda, situational_adjustments, _apply_pace_decay
from live_config import LIVE_PRIOR_STRENGTH, STAT_TIME_DISTRIBUTIONS
from live_state import LiveMatchState

# ---------------------------------------------------------------------------
# League -> data file mapping
# ---------------------------------------------------------------------------
LEAGUE_DATA_DIRS = {
    "EPL": _ROOT / "Output" / "Prem_feature_engineering",
    "LaLiga": _ROOT / "Output" / "LaLiga_feature_engineering",
    "SerieA": _ROOT / "Output" / "SeriaA_feature_engineering",
    "Bundesliga": _ROOT / "Output" / "Bundesliga_feature_engineering",
    "Ligue1": _ROOT / "Output" / "Ligue1_feature_engineering",
}

ALL_STATS = ["goals", "corners", "cards", "sot"]


# ---------------------------------------------------------------------------
# Load fixture data
# ---------------------------------------------------------------------------

def load_fixture_pairs(league: str, season_year: int) -> List[Dict]:
    """Load completed fixtures from team_engineered_features.

    Returns list of dicts with actual totals per fixture:
        fixture_id, home_team, away_team, fixture_date,
        actual_goals, actual_corners, actual_cards, actual_sot,
        home_goals, away_goals
    """
    data_dir = LEAGUE_DATA_DIRS.get(league)
    if data_dir is None:
        print(f"  [WARN] No data dir for league {league}")
        return []

    path = data_dir / f"team_engineered_features_{season_year}.json"
    if not path.exists():
        print(f"  [WARN] File not found: {path}")
        return []

    with open(path, "r") as f:
        data = json.load(f)

    # Group by fixture_id to pair home and away rows
    by_fixture: Dict[int, List[Dict]] = defaultdict(list)
    for row in data:
        fid = row.get("fixture_id")
        if fid is not None:
            by_fixture[fid].append(row)

    fixtures = []
    seen = set()
    for fid, rows in by_fixture.items():
        if fid in seen or len(rows) < 2:
            continue
        seen.add(fid)

        # Identify home and away
        home_row = None
        away_row = None
        for r in rows:
            if r.get("is_home") is True or r.get("is_home") == 1:
                home_row = r
            else:
                away_row = r
        if home_row is None or away_row is None:
            # Fallback: first row is home
            home_row = rows[0]
            away_row = rows[1]

        h_goals = _safe_int(home_row.get("goals_for", home_row.get("goals_scored", 0)))
        a_goals = _safe_int(away_row.get("goals_for", away_row.get("goals_scored", 0)))
        h_corners = _safe_int(home_row.get("corners", 0))
        a_corners = _safe_int(away_row.get("corners", 0))
        h_cards = (
            _safe_int(home_row.get("yellow_cards", 0))
            + _safe_int(home_row.get("red_cards", 0))
        )
        a_cards = (
            _safe_int(away_row.get("yellow_cards", 0))
            + _safe_int(away_row.get("red_cards", 0))
        )
        h_sot = _safe_int(home_row.get("shots_on_target", home_row.get("sot", 0)))
        a_sot = _safe_int(away_row.get("shots_on_target", away_row.get("sot", 0)))

        fixtures.append({
            "fixture_id": fid,
            "home_team": home_row.get("team", ""),
            "away_team": away_row.get("team", ""),
            "fixture_date": home_row.get("fixture_date", ""),
            "actual_goals": h_goals + a_goals,
            "actual_corners": h_corners + a_corners,
            "actual_cards": h_cards + a_cards,
            "actual_sot": h_sot + a_sot,
            "home_goals": h_goals,
            "away_goals": a_goals,
        })

    # Sort by fixture date
    fixtures.sort(key=lambda x: x.get("fixture_date", ""))
    return fixtures


def _safe_int(val) -> int:
    if val is None:
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Simulated event distribution at checkpoints
# ---------------------------------------------------------------------------

def simulate_observed_at_minute(
    actual_total: int,
    elapsed_min: float,
    stat: str,
) -> int:
    """Estimate how many events would have been observed by a given minute.

    Uses the empirical time distribution for the stat to compute the
    expected fraction of total events that would have occurred by the
    checkpoint minute. Rounds to nearest integer.

    This is NOT uniform: goals cluster late, cards accelerate late, etc.
    """
    if elapsed_min >= 90:
        return actual_total
    if elapsed_min <= 0:
        return 0

    dist = STAT_TIME_DISTRIBUTIONS.get(stat)
    if dist is None:
        # Fallback to uniform
        frac = elapsed_min / 90.0
        return round(actual_total * frac)

    elapsed_weight = 0.0
    for (lo, hi), weight in dist.items():
        if elapsed_min >= hi:
            elapsed_weight += weight
        elif elapsed_min > lo:
            window_len = hi - lo
            partial = (elapsed_min - lo) / window_len
            elapsed_weight += weight * partial

    # Normalize (distribution weights should sum to ~1.0 but be safe)
    total_weight = sum(dist.values())
    if total_weight > 0:
        fraction_elapsed = elapsed_weight / total_weight
    else:
        fraction_elapsed = elapsed_min / 90.0

    return round(actual_total * fraction_elapsed)


# ---------------------------------------------------------------------------
# Pre-match prior estimation (simplified)
# ---------------------------------------------------------------------------

def estimate_prior(
    fixtures: List[Dict],
    fixture_idx: int,
    stat: str,
    team_key: str = "home_team",
) -> float:
    """Estimate a pre-match prior for a stat from preceding fixtures.

    Uses league-wide rolling average of the last 30 fixtures before this one.
    Falls back to a sensible default if not enough data.
    """
    actual_key = f"actual_{stat}"
    defaults = {"goals": 2.7, "corners": 10.5, "cards": 4.1, "sot": 8.5}
    default = defaults.get(stat, 5.0)

    # Use preceding fixtures (up to 30)
    lookback = fixtures[max(0, fixture_idx - 30):fixture_idx]
    if len(lookback) < 5:
        return default

    vals = [f[actual_key] for f in lookback if actual_key in f]
    if not vals:
        return default
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Single fixture backtest
# ---------------------------------------------------------------------------

def backtest_fixture(
    fixture: Dict,
    prior_lambda: float,
    stat: str,
    prior_strength: float,
    checkpoints: List[int],
) -> List[Dict]:
    """Backtest a single fixture at multiple checkpoints.

    Returns list of result dicts, one per checkpoint:
        checkpoint, prior_strength, prior_lambda, observed, remaining_actual,
        remaining_projected, error, error_sq
    """
    actual_total = fixture.get(f"actual_{stat}", 0)
    results = []

    for minute in checkpoints:
        observed = simulate_observed_at_minute(actual_total, minute, stat)
        remaining_actual = actual_total - observed

        # Bayesian update
        remaining_projected = bayesian_update_lambda(
            pre_match_lambda=prior_lambda,
            elapsed_min=float(minute),
            observed_count=observed,
            total_min=90.0,
            prior_strength=prior_strength,
        )

        # Apply pace decay correction
        remaining_projected = _apply_pace_decay(
            remaining_projected, float(minute), stat, prior_lambda,
        )

        error = remaining_projected - remaining_actual
        results.append({
            "fixture_id": fixture["fixture_id"],
            "home_team": fixture.get("home_team", ""),
            "away_team": fixture.get("away_team", ""),
            "stat": stat,
            "checkpoint": minute,
            "prior_strength": prior_strength,
            "prior_lambda": round(prior_lambda, 3),
            "observed": observed,
            "actual_total": actual_total,
            "remaining_actual": remaining_actual,
            "remaining_projected": round(remaining_projected, 3),
            "error": round(error, 4),
            "error_sq": round(error ** 2, 4),
            "abs_error": round(abs(error), 4),
        })

    return results


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: List[Dict]) -> Dict[str, float]:
    """Compute MAE, RMSE, bias from a list of backtest result dicts."""
    if not results:
        return {"mae": 0.0, "rmse": 0.0, "bias": 0.0, "n": 0}

    errors = [r["error"] for r in results]
    abs_errors = [r["abs_error"] for r in results]
    sq_errors = [r["error_sq"] for r in results]
    n = len(errors)

    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(sq_errors) / n)
    bias = sum(errors) / n

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Main backtest function
# ---------------------------------------------------------------------------

def backtest_live_model(
    league: str = "EPL",
    season: int = 2025,
    stats: Optional[List[str]] = None,
    prior_strengths: Optional[List[float]] = None,
    checkpoints: Optional[List[int]] = None,
    skip_first: int = 5,
) -> Dict:
    """Run the full backtest sweep.

    Parameters
    ----------
    league : str
        League to backtest (e.g. "EPL", "LaLiga").
    season : int
        Season year (e.g. 2025 for 2024/25).
    stats : list of str, optional
        Stats to test (default: all four).
    prior_strengths : list of float, optional
        Prior strength values to sweep (default: [2, 3, 4, 5, 6, 7, 8, 10]).
    checkpoints : list of int, optional
        Minutes to evaluate at (default: [15, 30, 45, 60, 75]).
    skip_first : int
        Skip first N fixtures (cold start for prior estimation).

    Returns
    -------
    dict
        {
            "league": str,
            "season": int,
            "n_fixtures": int,
            "results": {
                stat: {
                    prior_strength: {"mae": float, "rmse": float, "bias": float},
                    ...
                },
                ...
            },
            "best": {stat: {"prior_strength": float, "mae": float, "rmse": float}},
            "all_rows": list of dict (raw per-fixture-per-checkpoint results),
        }
    """
    if stats is None:
        stats = ALL_STATS
    if prior_strengths is None:
        prior_strengths = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    if checkpoints is None:
        checkpoints = [15, 30, 45, 60, 75]

    fixtures = load_fixture_pairs(league, season)
    if not fixtures:
        print(f"  No fixtures loaded for {league} {season}")
        return {"league": league, "season": season, "n_fixtures": 0, "results": {}, "best": {}, "all_rows": []}

    # Skip cold-start fixtures
    usable = fixtures[skip_first:]
    print(f"  {league} {season}: {len(fixtures)} fixtures, using {len(usable)} (skip_first={skip_first})")

    all_rows: List[Dict] = []
    results: Dict[str, Dict[float, Dict]] = {}
    best: Dict[str, Dict] = {}

    for stat in stats:
        results[stat] = {}
        best_mae = float("inf")
        best_s = prior_strengths[0]

        for s in prior_strengths:
            stat_results = []
            for i, fixture in enumerate(usable):
                # Compute prior from preceding fixtures
                orig_idx = skip_first + i
                prior = estimate_prior(fixtures, orig_idx, stat)

                fixture_results = backtest_fixture(
                    fixture, prior, stat, s, checkpoints,
                )
                stat_results.extend(fixture_results)

            metrics = compute_metrics(stat_results)
            results[stat][s] = metrics
            all_rows.extend(stat_results)

            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best_s = s

        best[stat] = {
            "prior_strength": best_s,
            "mae": results[stat][best_s]["mae"],
            "rmse": results[stat][best_s]["rmse"],
            "bias": results[stat][best_s]["bias"],
        }

    return {
        "league": league,
        "season": season,
        "n_fixtures": len(usable),
        "results": results,
        "best": best,
        "all_rows": all_rows,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(report: Dict) -> None:
    """Print formatted backtest results to stdout."""
    league = report["league"]
    season = report["season"]
    n = report["n_fixtures"]

    print(f"\nLive Model Backtest \u2014 {league} {season} ({n} fixtures)")
    print("\u2501" * 50)

    results = report["results"]
    best = report["best"]

    for stat in sorted(results.keys()):
        print(f"\n{stat.upper()} (prior_strength sweep):")
        stat_results = results[stat]

        for s in sorted(stat_results.keys()):
            m = stat_results[s]
            marker = " \u2190 best" if s == best[stat]["prior_strength"] else ""
            print(
                f"  S={s:<5.1f} MAE={m['mae']:.4f}  "
                f"RMSE={m['rmse']:.4f}  "
                f"Bias={m['bias']:+.4f}  "
                f"(n={m['n']}){marker}"
            )

    print(f"\nRecommended prior_strengths:")
    for stat in sorted(best.keys()):
        b = best[stat]
        print(f"  {stat}: {b['prior_strength']:.1f} (MAE={b['mae']:.4f})")

    # Compare to current config
    print(f"\nCurrent config (live_config.py):")
    for stat in sorted(best.keys()):
        current = LIVE_PRIOR_STRENGTH.get(stat, "N/A")
        recommended = best[stat]["prior_strength"]
        match = "\u2713" if current == recommended else f"\u2192 change to {recommended}"
        print(f"  {stat}: {current} {match}")


def export_csv(all_rows: List[Dict], path: str) -> None:
    """Export raw results to CSV."""
    if not all_rows:
        print("  No rows to export")
        return

    fieldnames = [
        "fixture_id", "home_team", "away_team", "stat", "checkpoint",
        "prior_strength", "prior_lambda", "observed", "actual_total",
        "remaining_actual", "remaining_projected", "error", "abs_error",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Exported {len(all_rows)} rows to {path}")


# ---------------------------------------------------------------------------
# Per-checkpoint breakdown
# ---------------------------------------------------------------------------

def print_checkpoint_breakdown(report: Dict) -> None:
    """Print MAE broken down by checkpoint minute."""
    all_rows = report.get("all_rows", [])
    if not all_rows:
        return

    best = report["best"]

    print(f"\nPer-checkpoint breakdown (using best prior_strength):")
    for stat in sorted(best.keys()):
        best_s = best[stat]["prior_strength"]
        stat_rows = [r for r in all_rows if r["stat"] == stat and r["prior_strength"] == best_s]

        by_checkpoint: Dict[int, List[float]] = defaultdict(list)
        for r in stat_rows:
            by_checkpoint[r["checkpoint"]].append(r["abs_error"])

        print(f"  {stat.upper()} (S={best_s:.1f}):")
        for cp in sorted(by_checkpoint.keys()):
            errors = by_checkpoint[cp]
            mae = sum(errors) / len(errors)
            print(f"    {cp}': MAE={mae:.4f} (n={len(errors)})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the live projection engine")
    parser.add_argument("--league", type=str, default=None,
                        help="Single league (EPL, LaLiga, etc.). Default: all domestic.")
    parser.add_argument("--season", type=int, default=2025,
                        help="Season year (default: 2025)")
    parser.add_argument("--stat", type=str, default=None,
                        help="Single stat (goals, corners, cards, sot). Default: all.")
    parser.add_argument("--skip-first", type=int, default=5,
                        help="Skip first N fixtures per league (cold start, default: 5)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Export raw results to CSV file")
    parser.add_argument("--strengths", type=str, default=None,
                        help="Comma-separated prior_strength values to test (default: 2,3,4,5,6,7,8,10)")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Comma-separated checkpoint minutes (default: 15,30,45,60,75)")
    args = parser.parse_args()

    leagues = [args.league] if args.league else list(LEAGUE_DATA_DIRS.keys())
    stats = [args.stat] if args.stat else None
    strengths = [float(x) for x in args.strengths.split(",")] if args.strengths else None
    checkpoints = [int(x) for x in args.checkpoints.split(",")] if args.checkpoints else None

    all_csv_rows: List[Dict] = []

    for league in leagues:
        report = backtest_live_model(
            league=league,
            season=args.season,
            stats=stats,
            prior_strengths=strengths,
            checkpoints=checkpoints,
            skip_first=args.skip_first,
        )
        if report["n_fixtures"] == 0:
            continue

        print_results(report)
        print_checkpoint_breakdown(report)
        all_csv_rows.extend(report.get("all_rows", []))

    if args.csv and all_csv_rows:
        export_csv(all_csv_rows, args.csv)

    if not all_csv_rows:
        print("\nNo fixtures found. Ensure data files exist in Output/*_feature_engineering/")


if __name__ == "__main__":
    main()
