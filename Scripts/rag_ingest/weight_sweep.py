#!/usr/bin/env python3
"""
Weight sweep — systematic backtesting of different projection configurations.

Tests combinations of:
  - Season/recent blend ratios per stat
  - Home advantage values
  - xG blend weights
  - ML blend thresholds

Uses the existing backtest.py infrastructure but mutates SCORING_WEIGHTS
between runs to compare configurations.

Usage:
    cd Scripts/rag_ingest
    python weight_sweep.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rag_cli_v2 as rag
from backtest import load_fixture_pairs, project_fixture, compute_metrics
from prob_models import over_prob

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LEAGUES = ["EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1"]
SEASON_YEAR = 2025
SKIP_FIRST = 10  # cold-start
STATS = ["goals", "corners", "cards", "sot"]

# Save original weights to restore between experiments
_ORIGINAL_WEIGHTS = deepcopy(rag.SCORING_WEIGHTS)


def _reset_weights():
    """Restore original weights."""
    for k, v in _ORIGINAL_WEIGHTS.items():
        rag.SCORING_WEIGHTS[k] = deepcopy(v)


# ---------------------------------------------------------------------------
# Data loading (cache across experiments)
# ---------------------------------------------------------------------------
_fixture_cache: Dict[str, List[Dict]] = {}


def get_fixtures(league: str) -> List[Dict]:
    if league not in _fixture_cache:
        pairs = load_fixture_pairs(league, SEASON_YEAR)
        # Apply skip_first filter
        team_count: Dict[str, int] = defaultdict(int)
        filtered = []
        for fx in pairs:
            team_count[fx["home_team"]] += 1
            team_count[fx["away_team"]] += 1
            if team_count[fx["home_team"]] > SKIP_FIRST and team_count[fx["away_team"]] > SKIP_FIRST:
                filtered.append(fx)
        _fixture_cache[league] = filtered
    return _fixture_cache[league]


# ---------------------------------------------------------------------------
# Run a single experiment
# ---------------------------------------------------------------------------
def run_experiment(name: str, weight_overrides: Dict) -> Dict:
    """Run backtest with specific weight overrides.

    weight_overrides is a nested dict like:
        {"projection": {"blend_season": 0.70, "blend_recent": 0.30}}
    """
    _reset_weights()

    # Apply overrides
    for section, vals in weight_overrides.items():
        if section in rag.SCORING_WEIGHTS:
            for k, v in vals.items():
                rag.SCORING_WEIGHTS[section][k] = v

    # Run projections across all leagues
    all_results = []
    for league in LEAGUES:
        fixtures = get_fixtures(league)
        for fx in fixtures:
            try:
                proj = project_fixture(fx["home_team"], fx["away_team"], league)
                all_results.append({"league": league, **fx, **proj})
            except Exception:
                pass

    # Compute metrics per stat
    metrics = {}
    for stat in STATS:
        m = compute_metrics(all_results, stat)
        if m.get("n", 0) > 0:
            metrics[stat] = m

    return {"name": name, "n_fixtures": len(all_results), "metrics": metrics}


def summarize(result: Dict) -> str:
    """One-line summary of an experiment."""
    parts = [f"{result['name']:40s}"]
    for stat in STATS:
        m = result["metrics"].get(stat, {})
        if m.get("n", 0) > 0:
            mae = m["mae"]
            skill = m.get("skill_vs_naive", 0) or 0
            # Average hit rate across common lines
            hit_keys = [k for k in m if k.startswith("hit_")]
            avg_hit = sum(m[k] for k in hit_keys if m[k] is not None) / max(len(hit_keys), 1)
            parts.append(f"{stat[:3]}:MAE={mae:.3f} skill={skill:+.1%} hit={avg_hit:.1%}")
        else:
            parts.append(f"{stat[:3]}:N/A")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = []

# ── Baseline (current weights) ──
EXPERIMENTS.append(("BASELINE (current)", {}))

# ── Season/Recent blend sweep ──
for stat in STATS:
    for season_w in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        recent_w = round(1.0 - season_w, 2)
        EXPERIMENTS.append((
            f"blend_{stat}={season_w:.0%}/{recent_w:.0%}",
            {"projection": {
                f"blend_{stat}_season": season_w,
                f"blend_{stat}_recent": recent_w,
            }}
        ))

# ── Home advantage sweep ──
for ha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    EXPERIMENTS.append((
        f"home_adv={ha:.2f}",
        {"projection": {"home_advantage": ha}}
    ))

# ── xG blend sweep ──
for xg in [0.0, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
    EXPERIMENTS.append((
        f"xg_blend={xg:.0%}",
        {"projection": {"xg_blend": xg}}
    ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("  WEIGHT SWEEP — Systematic Backtesting")
    print("=" * 80)
    print(f"  Leagues: {', '.join(LEAGUES)}")
    print(f"  Season: {SEASON_YEAR}")
    print(f"  Skip first: {SKIP_FIRST} fixtures per team")
    print(f"  Experiments: {len(EXPERIMENTS)}")

    # Pre-load all fixture data
    print("\n  Loading fixtures...")
    total_fx = 0
    for league in LEAGUES:
        fx = get_fixtures(league)
        total_fx += len(fx)
        print(f"    {league}: {len(fx)} fixtures")
    print(f"  Total: {total_fx} fixtures\n")

    results = []
    t0 = time.time()

    for i, (name, overrides) in enumerate(EXPERIMENTS):
        t_exp = time.time()
        print(f"  [{i+1}/{len(EXPERIMENTS)}] {name}...", end="", flush=True)
        result = run_experiment(name, overrides)
        elapsed = time.time() - t_exp
        print(f" done ({elapsed:.1f}s)")
        results.append(result)

    total_time = time.time() - t0
    print(f"\n  All experiments done in {total_time:.0f}s\n")

    # Reset to original
    _reset_weights()

    # ── Print results ──
    print("=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)

    # Find baseline
    baseline = results[0]
    bl_metrics = baseline["metrics"]

    print(f"\n  BASELINE:")
    for stat in STATS:
        m = bl_metrics.get(stat, {})
        if m.get("n", 0) > 0:
            hit_keys = [k for k in m if k.startswith("hit_")]
            avg_hit = sum(m[k] for k in hit_keys if m[k] is not None) / max(len(hit_keys), 1)
            print(f"    {stat:8s}: MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  "
                  f"bias={m['bias']:+.3f}  skill={m.get('skill_vs_naive', 0):+.1%}  "
                  f"avg_hit={avg_hit:.1%}")

    # ── Per-stat best configs ──
    for stat in STATS:
        print(f"\n  BEST CONFIGS FOR {stat.upper()}:")
        print(f"  {'Config':40s} {'MAE':>7s} {'RMSE':>7s} {'Bias':>7s} {'Skill':>7s} {'AvgHit':>7s}")
        print(f"  {'-'*75}")

        stat_results = []
        for r in results:
            m = r["metrics"].get(stat, {})
            if m.get("n", 0) > 0:
                hit_keys = [k for k in m if k.startswith("hit_")]
                avg_hit = sum(m[k] for k in hit_keys if m[k] is not None) / max(len(hit_keys), 1)
                stat_results.append((r["name"], m["mae"], m["rmse"], m["bias"],
                                     m.get("skill_vs_naive", 0) or 0, avg_hit))

        # Sort by MAE (lower is better)
        stat_results.sort(key=lambda x: x[1])
        for name, mae, rmse, bias, skill, avg_hit in stat_results[:10]:
            marker = " <-- BEST" if stat_results[0][0] == name else ""
            print(f"  {name:40s} {mae:7.3f} {rmse:7.3f} {bias:+7.3f} {skill:+6.1%} {avg_hit:6.1%}{marker}")

    # ── Per-stat best by hit rate ──
    for stat in STATS:
        print(f"\n  BEST HIT RATES FOR {stat.upper()}:")
        print(f"  {'Config':40s} {'AvgHit':>7s} {'MAE':>7s}")
        print(f"  {'-'*55}")

        stat_results = []
        for r in results:
            m = r["metrics"].get(stat, {})
            if m.get("n", 0) > 0:
                hit_keys = [k for k in m if k.startswith("hit_")]
                avg_hit = sum(m[k] for k in hit_keys if m[k] is not None) / max(len(hit_keys), 1)
                stat_results.append((r["name"], avg_hit, m["mae"]))

        stat_results.sort(key=lambda x: -x[1])  # highest hit rate first
        for name, avg_hit, mae in stat_results[:10]:
            print(f"  {name:40s} {avg_hit:6.1%} {mae:7.3f}")

    # ── Home advantage detail ──
    print(f"\n  HOME ADVANTAGE SWEEP (goals only):")
    print(f"  {'HA Value':>10s} {'MAE':>7s} {'Bias':>7s} {'Skill':>7s} {'Hit2.5':>7s} {'Hit3.5':>7s}")
    print(f"  {'-'*55}")
    for r in results:
        if r["name"].startswith("home_adv="):
            m = r["metrics"].get("goals", {})
            if m.get("n", 0) > 0:
                ha_val = r["name"].split("=")[1]
                h25 = m.get("hit_2.5", 0) or 0
                h35 = m.get("hit_3.5", 0) or 0
                print(f"  {ha_val:>10s} {m['mae']:7.3f} {m['bias']:+7.3f} "
                      f"{m.get('skill_vs_naive', 0):+6.1%} {h25:6.1%} {h35:6.1%}")

    # ── xG blend detail ──
    print(f"\n  xG BLEND SWEEP (goals only):")
    print(f"  {'xG Weight':>10s} {'MAE':>7s} {'Bias':>7s} {'Skill':>7s} {'Hit2.5':>7s}")
    print(f"  {'-'*45}")
    for r in results:
        if r["name"].startswith("xg_blend="):
            m = r["metrics"].get("goals", {})
            if m.get("n", 0) > 0:
                xg_val = r["name"].split("=")[1]
                h25 = m.get("hit_2.5", 0) or 0
                print(f"  {xg_val:>10s} {m['mae']:7.3f} {m['bias']:+7.3f} "
                      f"{m.get('skill_vs_naive', 0):+6.1%} {h25:6.1%}")

    # Save full results to JSON
    output_path = Path(__file__).resolve().parent.parent.parent / "Index" / "weight_sweep_results.json"
    serializable = []
    for r in results:
        entry = {"name": r["name"], "n_fixtures": r["n_fixtures"], "metrics": {}}
        for stat, m in r["metrics"].items():
            entry["metrics"][stat] = {k: v for k, v in m.items() if k != "calibration"}
        serializable.append(entry)

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Full results saved to {output_path}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
