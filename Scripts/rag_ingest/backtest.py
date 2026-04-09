#!/usr/bin/env python3
"""
Backtesting harness for the RAG prediction engine.

Compares projected totals (goals, corners, cards, SoT) against actual fixture
outcomes.  Uses the same projection pipeline as the live system (Chroma
profiles + recent stats + ML blend) so results reflect real-world accuracy.

Usage:
    cd Scripts/rag_ingest
    python backtest.py                          # all domestic leagues, current season
    python backtest.py --league EPL             # single league
    python backtest.py --season 2024            # specific season file
    python backtest.py --skip-first 10          # skip first N fixtures per team (cold start)
    python backtest.py --csv results.csv        # export raw results
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

# ---------------------------------------------------------------------------
# Imports from the RAG engine
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prob_models import over_prob, under_prob

# We import projection functions from rag_cli_v2.  These pull from Chroma
# and ML models, so the Chroma DB must be populated.
import rag_cli_v2 as rag

try:
    from league_context import get_league_context
except ImportError:
    try:
        from Scripts.rag_ingest.league_context import get_league_context
    except ImportError:
        get_league_context = None

try:
    from referee_data import load_fixture_referee_detail, compute_referee_modifier
except ImportError:
    try:
        from Scripts.rag_ingest.referee_data import load_fixture_referee_detail, compute_referee_modifier
    except ImportError:
        load_fixture_referee_detail = None
        compute_referee_modifier = None

# ---------------------------------------------------------------------------
# League ↔ data file mapping
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

LEAGUE_DATA_DIRS = {
    "EPL": PROJECT_ROOT / "Output" / "Prem_feature_engineering",
    "LaLiga": PROJECT_ROOT / "Output" / "LaLiga_feature_engineering",
    "SerieA": PROJECT_ROOT / "Output" / "SeriaA_feature_engineering",
    "Bundesliga": PROJECT_ROOT / "Output" / "Bundesliga_feature_engineering",
    "Ligue1": PROJECT_ROOT / "Output" / "Ligue1_feature_engineering",
}

# Season year in filename → season string (e.g. 2025 → "2024/25")
def _season_str(year: int) -> str:
    return f"{year - 1}/{str(year)[2:]}"


# ---------------------------------------------------------------------------
# Load fixture pairs from raw engineered features
# ---------------------------------------------------------------------------
def load_fixture_pairs(league: str, season_year: int) -> List[Dict]:
    """
    Load fixture pairs from team_engineered_features_{year}.json.

    Returns list of dicts with keys:
        fixture_id, fixture, fixture_date, home_team, away_team,
        actual_goals, actual_corners, actual_cards, actual_sot,
        home_goals, away_goals, home_corners, away_corners,
        home_cards, away_cards, home_sot, away_sot
    """
    data_dir = LEAGUE_DATA_DIRS.get(league)
    if data_dir is None:
        print(f"  [WARN] No data dir for league {league}")
        return []

    # Try season year file
    path = data_dir / f"team_engineered_features_{season_year}.json"
    if not path.exists():
        print(f"  [WARN] File not found: {path}")
        return []

    with open(path) as f:
        rows = json.load(f)

    # Group by fixture identifier.
    # EPL has fixture_id; other leagues use fixture name as key.
    has_fixture_id = "fixture_id" in rows[0]
    by_fixture = defaultdict(list)
    for row in rows:
        key = row.get("fixture_id") if has_fixture_id else row["fixture"]
        by_fixture[key].append(row)

    pairs = []
    for fid, frows in by_fixture.items():
        if len(frows) != 2:
            continue

        r0, r1 = frows

        # Identify home and away
        if "home_team" in r0:
            # EPL style: explicit home_team/away_team fields
            if r0["team"] == r0["home_team"]:
                home_row, away_row = r0, r1
            elif r1["team"] == r1["home_team"]:
                home_row, away_row = r1, r0
            else:
                continue
        else:
            # Other leagues: parse from fixture string "Home vs Away"
            fixture_str = r0["fixture"]
            parts = fixture_str.split(" vs ")
            if len(parts) != 2:
                continue
            home_name = parts[0].strip()
            if r0["team"] == home_name:
                home_row, away_row = r0, r1
            elif r1["team"] == home_name:
                home_row, away_row = r1, r0
            else:
                continue

        # Resolve shots_on (some leagues use 'shots_on_x' from merge artifacts)
        def _sot(row):
            return row.get("shots_on") or row.get("shots_on_x") or 0

        pairs.append({
            "fixture_id": fid if has_fixture_id else fixture_str,
            "fixture": home_row["fixture"],
            "fixture_date": home_row.get("fixture_date", ""),
            "home_team": home_row["team"],
            "away_team": away_row["team"],
            # Actuals
            "actual_goals": home_row["goals"] + away_row["goals"],
            "actual_corners": home_row["corners"] + away_row["corners"],
            "actual_cards": home_row["cards_total"] + away_row["cards_total"],
            "actual_sot": _sot(home_row) + _sot(away_row),
            # Per-team actuals
            "home_goals": home_row["goals"],
            "away_goals": away_row["goals"],
            "home_corners": home_row["corners"],
            "away_corners": away_row["corners"],
            "home_cards": home_row["cards_total"],
            "away_cards": away_row["cards_total"],
            "home_sot": _sot(home_row),
            "away_sot": _sot(away_row),
        })

    # Sort by date (use empty string for leagues without dates)
    pairs.sort(key=lambda x: x["fixture_date"])
    return pairs


# ---------------------------------------------------------------------------
# Run projections for a fixture
# ---------------------------------------------------------------------------
def _historical_ref_mod(fixture_id) -> Optional[object]:
    """Build a referee modifier from the stored historical assignment for a fixture."""
    if fixture_id is None or load_fixture_referee_detail is None or compute_referee_modifier is None:
        return None
    try:
        detail = load_fixture_referee_detail()
        row = detail.get(int(fixture_id))
    except Exception:
        return None
    if not row:
        return None
    ref_name = row.get("referee_name")
    if not ref_name:
        return None
    try:
        return compute_referee_modifier(ref_name, weight=0.35)
    except Exception:
        return None


def project_fixture(
    home: str,
    away: str,
    league: str,
    fixture_date: Optional[str] = None,
    fixture_id=None,
) -> Dict:
    """
    Run all projection functions for a fixture.
    Returns dict with projected totals (or None if unavailable).
    """
    result = {}
    league_ctx = None
    if get_league_context is not None:
        try:
            league_ctx = get_league_context(home, away, league, target_date=fixture_date)
        except Exception:
            league_ctx = None
    ref_mod = _historical_ref_mod(fixture_id)

    # Goals
    try:
        goals_bl, goals_s, goals_r = rag.projected_total_goals(
            home, away, league,
            league_ctx=league_ctx,
            fixture_date=fixture_date,
        )
        result["proj_goals"] = goals_bl
        result["proj_goals_season"] = goals_s
        result["proj_goals_recent"] = goals_r
    except Exception:
        result["proj_goals"] = None
        result["proj_goals_season"] = None
        result["proj_goals_recent"] = None

    # Corners
    try:
        corners_bl, corners_s, corners_r = rag.projected_total_corners(
            home, away, league,
            league_ctx=league_ctx,
            fixture_date=fixture_date,
        )
        result["proj_corners"] = corners_bl
        result["proj_corners_season"] = corners_s
        result["proj_corners_recent"] = corners_r
    except Exception:
        result["proj_corners"] = None
        result["proj_corners_season"] = None
        result["proj_corners_recent"] = None

    # Cards (use detail helper for richer output)
    try:
        detail_fn = getattr(rag, "projected_total_cards_detail", None)
        if detail_fn:
            detail = detail_fn(
                home, away, league,
                ref_mod=ref_mod,
                league_ctx=league_ctx,
                fixture_date=fixture_date,
            )
            result["proj_cards"] = detail.get("total_cards")
            result["proj_cards_season"] = detail.get("season_total")
            result["proj_cards_recent"] = detail.get("recent_total")
            ref_mod = detail.get("referee_modifier")
            result["ref_multiplier"] = ref_mod.multiplier if ref_mod else None
            result["ref_source"] = getattr(ref_mod, "source", "unavailable") if ref_mod else "unavailable"
            result["ref_confidence"] = getattr(ref_mod, "confidence", 0.0) if ref_mod else 0.0
            result["proj_cards_yellows"] = detail.get("total_yellows")
            result["proj_cards_reds"] = detail.get("total_reds")
            result["cards_uncertainty"] = detail.get("uncertainty")
            result["cards_confidence_bucket"] = detail.get("confidence_bucket")
        else:
            cards_bl, cards_s, cards_r, ref_mod = rag.projected_total_cards(
                home, away, league,
                ref_mod=ref_mod,
                league_ctx=league_ctx,
                fixture_date=fixture_date,
            )
            result["proj_cards"] = cards_bl
            result["proj_cards_season"] = cards_s
            result["proj_cards_recent"] = cards_r
            result["ref_multiplier"] = ref_mod.multiplier if ref_mod else None
            result["ref_source"] = getattr(ref_mod, "source", "unavailable") if ref_mod else "unavailable"
            result["ref_confidence"] = getattr(ref_mod, "confidence", 0.0) if ref_mod else 0.0
    except Exception:
        result["proj_cards"] = None
        result["proj_cards_season"] = None
        result["proj_cards_recent"] = None
        result["ref_multiplier"] = None
        result["ref_source"] = "unavailable"
        result["ref_confidence"] = 0.0

    # SoT
    try:
        sot_bl, sot_s, sot_r = rag.projected_total_sot(
            home, away, league,
            league_ctx=league_ctx,
            fixture_date=fixture_date,
        )
        result["proj_sot"] = sot_bl
        result["proj_sot_season"] = sot_s
        result["proj_sot_recent"] = sot_r
    except Exception:
        result["proj_sot"] = None
        result["proj_sot_season"] = None
        result["proj_sot_recent"] = None

    if league_ctx is not None:
        result["home_rest_days"] = league_ctx.home_rest_days
        result["away_rest_days"] = league_ctx.away_rest_days
        result["home_short_rest"] = league_ctx.home_short_rest
        result["away_short_rest"] = league_ctx.away_short_rest
        result["any_short_rest"] = league_ctx.home_short_rest or league_ctx.away_short_rest
        result["stake_description"] = league_ctx.stake_description
        result["ctx_goals_adj"] = league_ctx.adjustments.get("goals")
        result["ctx_corners_adj"] = league_ctx.adjustments.get("corners")
        result["ctx_cards_adj"] = league_ctx.adjustments.get("cards")
        result["ctx_sot_adj"] = league_ctx.adjustments.get("sot")
    else:
        result["home_rest_days"] = None
        result["away_rest_days"] = None
        result["home_short_rest"] = None
        result["away_short_rest"] = None
        result["any_short_rest"] = None
        result["stake_description"] = ""
        result["ctx_goals_adj"] = None
        result["ctx_corners_adj"] = None
        result["ctx_cards_adj"] = None
        result["ctx_sot_adj"] = None

    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(results: List[Dict], stat: str) -> Dict:
    """
    Compute accuracy metrics for a given stat (goals, corners, cards, sot).

    Returns dict with:
        n, mae, rmse, mean_actual, mean_projected, bias,
        hit_rate_XX (% of correct over/under calls at common lines),
        calibration buckets
    """
    proj_key = f"proj_{stat}"
    actual_key = f"actual_{stat}"

    # Filter to rows with valid projections
    valid = [(r[proj_key], r[actual_key]) for r in results
             if r.get(proj_key) is not None and r.get(actual_key) is not None]

    if not valid:
        return {"n": 0}

    projs, actuals = zip(*valid)
    n = len(valid)
    mean_actual = sum(actuals) / n
    mean_proj = sum(projs) / n

    # MAE
    errors = [abs(p - a) for p, a in valid]
    mae = sum(errors) / n

    # RMSE
    sq_errors = [(p - a) ** 2 for p, a in valid]
    rmse = math.sqrt(sum(sq_errors) / n)

    # Bias (positive = over-projecting)
    bias = mean_proj - mean_actual

    # Naive baseline MAE (always predict the mean of actuals)
    naive_mae = sum(abs(a - mean_actual) for a in actuals) / n

    # Hit rates at common lines
    lines = _common_lines(stat)
    hit_rates = {}
    for line in lines:
        correct = 0
        total = 0
        for p, a in valid:
            # Would we recommend over or under?
            p_over = over_prob(p, line)
            recommend_over = p_over >= 0.5
            actual_over = a > line

            total += 1
            if recommend_over == actual_over:
                correct += 1

        hit_rates[f"hit_{line}"] = correct / total if total > 0 else None

    # Model probability calibration
    calibration = _calibration_buckets(valid, stat)

    return {
        "n": n,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mean_actual": round(mean_actual, 4),
        "mean_projected": round(mean_proj, 4),
        "bias": round(bias, 4),
        "naive_mae": round(naive_mae, 4),
        "skill_vs_naive": round(1.0 - mae / naive_mae, 4) if naive_mae > 0 else None,
        **{k: round(v, 4) if v is not None else None for k, v in hit_rates.items()},
        "calibration": calibration,
    }


def _common_lines(stat: str) -> List[float]:
    """Common betting lines per stat."""
    if stat == "goals":
        return [1.5, 2.5, 3.5, 4.5]
    elif stat == "corners":
        return [8.5, 9.5, 10.5, 11.5]
    elif stat == "cards":
        return [2.5, 3.5, 4.5, 5.5]
    elif stat == "sot":
        return [7.5, 8.5, 9.5, 10.5]
    return []


def _calibration_buckets(valid: List[Tuple[float, float]], stat: str) -> List[Dict]:
    """
    For each common line, bucket fixtures by model confidence and check
    actual hit rate.  Good calibration = model P(over) ≈ actual % over.
    """
    lines = _common_lines(stat)
    buckets = []

    for line in lines:
        # Group by probability range
        ranges = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
                  (0.70, 0.75), (0.75, 0.80), (0.80, 1.01)]

        for lo, hi in ranges:
            correct = 0
            total = 0
            for proj, actual in valid:
                p_over = over_prob(proj, line)
                p_best = max(p_over, 1.0 - p_over)
                if lo <= p_best < hi:
                    recommend_over = p_over >= 0.5
                    actual_over = actual > line
                    total += 1
                    if recommend_over == actual_over:
                        correct += 1

            if total >= 5:  # Need at least 5 samples for meaningful bucket
                buckets.append({
                    "line": line,
                    "conf_range": f"{lo:.0%}-{hi:.0%}",
                    "n": total,
                    "hit_rate": round(correct / total, 4),
                    "expected_mid": round((lo + hi) / 2, 4),
                })

    return buckets


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(all_metrics: Dict[str, Dict], league: str):
    """Print a formatted backtesting report."""
    print(f"\n{'='*70}")
    print(f"  BACKTEST REPORT — {league}")
    print(f"{'='*70}")

    for stat in ["goals", "corners", "cards", "sot"]:
        m = all_metrics.get(stat, {})
        if m.get("n", 0) == 0:
            print(f"\n  {stat.upper()}: No data")
            continue

        print(f"\n  {stat.upper()} ({m['n']} fixtures)")
        print(f"  {'─'*40}")
        print(f"  Mean actual:     {m['mean_actual']:.2f}")
        print(f"  Mean projected:  {m['mean_projected']:.2f}")
        print(f"  Bias:            {m['bias']:+.2f} ({'over' if m['bias'] > 0 else 'under'}-projecting)")
        print(f"  MAE:             {m['mae']:.3f}")
        print(f"  RMSE:            {m['rmse']:.3f}")
        print(f"  Naive MAE:       {m['naive_mae']:.3f}")
        skill = m.get('skill_vs_naive')
        if skill is not None:
            emoji = "BETTER" if skill > 0 else "WORSE"
            print(f"  Skill vs naive:  {skill:+.1%} ({emoji} than always-predict-mean)")

        # Hit rates
        hit_keys = [k for k in m if k.startswith("hit_")]
        if hit_keys:
            print(f"\n  Over/Under Hit Rates:")
            for k in sorted(hit_keys):
                line = k.replace("hit_", "")
                rate = m[k]
                if rate is not None:
                    bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
                    print(f"    Line {line:>5s}: {rate:.1%}  {bar}")

        # Calibration
        cal = m.get("calibration", [])
        if cal:
            print(f"\n  Calibration (model confidence → actual hit rate):")
            for b in cal:
                delta = b["hit_rate"] - b["expected_mid"]
                print(f"    Line {b['line']:>5.1f} | Conf {b['conf_range']:>8s} | "
                      f"n={b['n']:>3d} | Hit={b['hit_rate']:.1%} | "
                      f"Expected≈{b['expected_mid']:.0%} | "
                      f"Delta={delta:+.1%}")

    print(f"\n{'='*70}\n")


def export_csv(results: List[Dict], path: str):
    """Export raw backtest results to CSV."""
    if not results:
        return

    fieldnames = sorted(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"  Exported {len(results)} rows to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_backtest(
    leagues: List[str],
    season_year: int = 2025,
    skip_first: int = 10,
    csv_path: Optional[str] = None,
    verbose: bool = False,
):
    """
    Run backtesting across specified leagues.

    Args:
        leagues: List of league codes (e.g. ["EPL", "LaLiga"])
        season_year: Year suffix for data files (e.g. 2025 for 2024/25 season)
        skip_first: Skip first N fixtures per team (cold-start period where
                    recent stats are unreliable)
        csv_path: Optional path to export raw results
        verbose: Print per-fixture details
    """
    all_results = []

    for league in leagues:
        print(f"\n  Loading {league} fixtures (season {season_year})...")
        pairs = load_fixture_pairs(league, season_year)
        print(f"  Found {len(pairs)} fixtures")

        if not pairs:
            continue

        # Track fixture count per team for cold-start filtering
        team_fixture_count: Dict[str, int] = defaultdict(int)

        league_results = []
        skipped = 0
        errors = 0

        for i, fixture in enumerate(pairs):
            home = fixture["home_team"]
            away = fixture["away_team"]

            # Skip cold-start fixtures
            team_fixture_count[home] += 1
            team_fixture_count[away] += 1
            if team_fixture_count[home] <= skip_first or team_fixture_count[away] <= skip_first:
                skipped += 1
                continue

            # Run projections
            try:
                projections = project_fixture(
                    home, away, league,
                    fixture_date=fixture.get("fixture_date"),
                    fixture_id=fixture.get("fixture_id"),
                )
            except Exception as e:
                errors += 1
                if verbose:
                    print(f"    [ERR] {fixture['fixture']}: {e}")
                continue

            # Combine actuals + projections
            row = {
                "league": league,
                **fixture,
                **projections,
            }
            league_results.append(row)

            if verbose and (i + 1) % 20 == 0:
                print(f"    Processed {i + 1}/{len(pairs)} fixtures...")

        print(f"  Processed: {len(league_results)}, Skipped (cold start): {skipped}, Errors: {errors}")

        # Compute metrics per stat
        metrics = {}
        for stat in ["goals", "corners", "cards", "sot"]:
            metrics[stat] = compute_metrics(league_results, stat)

        print_report(metrics, league)

        # Cards slices: referee assigned vs missing, high/low confidence
        _cards_ref_assigned = [r for r in league_results if r.get("ref_source") == "profile"]
        _cards_ref_missing = [r for r in league_results if r.get("ref_source") != "profile"]
        _cards_high_conf = [r for r in league_results if r.get("cards_confidence_bucket") == "high"]
        _cards_low_conf = [r for r in league_results if r.get("cards_confidence_bucket") == "low"]

        if _cards_ref_assigned:
            m = compute_metrics(_cards_ref_assigned, "cards")
            if m.get("n", 0) > 0:
                print(f"  CARDS — Referee assigned ({m['n']} fixtures): MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} Skill={m.get('skill_vs_naive', 0):+.1%}")
        if _cards_ref_missing:
            m = compute_metrics(_cards_ref_missing, "cards")
            if m.get("n", 0) > 0:
                print(f"  CARDS — Referee missing  ({m['n']} fixtures): MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} Skill={m.get('skill_vs_naive', 0):+.1%}")
        if _cards_high_conf:
            m = compute_metrics(_cards_high_conf, "cards")
            if m.get("n", 0) > 0:
                print(f"  CARDS — High confidence  ({m['n']} fixtures): MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} Skill={m.get('skill_vs_naive', 0):+.1%}")
        if _cards_low_conf:
            m = compute_metrics(_cards_low_conf, "cards")
            if m.get("n", 0) > 0:
                print(f"  CARDS — Low confidence   ({m['n']} fixtures): MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} Skill={m.get('skill_vs_naive', 0):+.1%}")

        _short_rest = [r for r in league_results if r.get("any_short_rest") is True]
        _normal_rest = [r for r in league_results if r.get("any_short_rest") is False]
        if _short_rest and _normal_rest:
            for stat in ["goals", "corners", "cards"]:
                m_short = compute_metrics(_short_rest, stat)
                m_norm = compute_metrics(_normal_rest, stat)
                if m_short.get("n", 0) >= 5 and m_norm.get("n", 0) >= 5:
                    print(
                        f"  {stat.upper():<6} — Short rest ({m_short['n']} fixtures): "
                        f"MAE={m_short['mae']:.3f} Skill={m_short.get('skill_vs_naive', 0):+.1%} | "
                        f"Normal rest ({m_norm['n']}): MAE={m_norm['mae']:.3f} Skill={m_norm.get('skill_vs_naive', 0):+.1%}"
                    )

        _high_stakes = [r for r in league_results if r.get("stake_description")]
        _neutral_stakes = [r for r in league_results if not r.get("stake_description")]
        if _high_stakes and _neutral_stakes:
            for stat in ["goals", "corners", "cards"]:
                m_stakes = compute_metrics(_high_stakes, stat)
                m_neutral = compute_metrics(_neutral_stakes, stat)
                if m_stakes.get("n", 0) >= 5 and m_neutral.get("n", 0) >= 5:
                    print(
                        f"  {stat.upper():<6} — Stake context ({m_stakes['n']} fixtures): "
                        f"MAE={m_stakes['mae']:.3f} Skill={m_stakes.get('skill_vs_naive', 0):+.1%} | "
                        f"Neutral ({m_neutral['n']}): MAE={m_neutral['mae']:.3f} Skill={m_neutral.get('skill_vs_naive', 0):+.1%}"
                    )

        all_results.extend(league_results)

    # Cross-league aggregate
    if len(leagues) > 1 and all_results:
        print("\n" + "=" * 70)
        print("  AGGREGATE ACROSS ALL LEAGUES")
        agg_metrics = {}
        for stat in ["goals", "corners", "cards", "sot"]:
            agg_metrics[stat] = compute_metrics(all_results, stat)
        print_report(agg_metrics, "ALL LEAGUES")

    # Export CSV
    if csv_path and all_results:
        # Remove calibration dict from CSV rows
        csv_results = []
        for r in all_results:
            row = {k: v for k, v in r.items() if not isinstance(v, list)}
            csv_results.append(row)
        export_csv(csv_results, csv_path)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Backtest RAG prediction engine")
    parser.add_argument("--league", type=str, default=None,
                        help="Single league to test (default: all domestic)")
    parser.add_argument("--season", type=int, default=2025,
                        help="Season year for data files (default: 2025)")
    parser.add_argument("--skip-first", type=int, default=10,
                        help="Skip first N fixtures per team (cold start, default: 10)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Export raw results to CSV")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-fixture progress")

    args = parser.parse_args()

    if args.league:
        leagues = [args.league]
    else:
        leagues = list(LEAGUE_DATA_DIRS.keys())

    run_backtest(
        leagues=leagues,
        season_year=args.season,
        skip_first=args.skip_first,
        csv_path=args.csv,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
