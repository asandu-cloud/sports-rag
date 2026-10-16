"""A3.6 study: does a fitted Dixon–Coles rho beat the fixed -0.10?

The frozen control's goal forecasts (Phase 4 step 2 runs) give each fixture's goal
means; actual scorelines come from the platform database. Rho is fitted on 2022
(pooled, per league, and per league shrunk toward the pooled value) and every
variant is scored on 2023 by the joint scoreline log loss, against the fixed -0.10,
with a calendar-week bootstrap and league slices.

    python -m Scripts.ops.rho_study --out Research/rho-<date>
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import gzip
import json
from pathlib import Path
import sqlite3

import numpy as np

from Scripts.data_platform import dixon_coles_rho as dcr
from Scripts.data_platform.features.bookmaker_accuracy import SEED, _interval
from Scripts.ops.odds_archive import ROOT

RUNS = ROOT / "Research/phase4-step2-backtest-2026-09-29/runs"
FIXED = -0.10
SHRINK_STRENGTH = 300


def _per_row(rows, rho_of):
    return [dcr.nll([r], rho_of(r)) for r in rows]


def evaluate(train, test, *, resamples=2000, seed=SEED):
    pooled = dcr.fit(train)["rho"]
    by_league = defaultdict(list)
    for r in train:
        by_league[r["league"]].append(r)
    league_fit = {lg: dcr.fit(rows) for lg, rows in by_league.items()}
    league = {lg: f["rho"] for lg, f in league_fit.items()}
    shrunk = {lg: dcr.shrink(f["rho"], f["n"], pooled=pooled, strength=SHRINK_STRENGTH) for lg, f in league_fit.items()}
    rho = {"fixed": lambda r: FIXED, "pooled": lambda r: pooled,
           "league": lambda r: league.get(r["league"], pooled), "league_shrunk": lambda r: shrunk.get(r["league"], pooled)}
    losses = {name: _per_row(test, f) for name, f in rho.items()}
    weeks = [r["week"] for r in test]

    def compare(name, idx=None):
        idx = range(len(test)) if idx is None else idx
        a, b = np.array([losses[name][i] for i in idx]), np.array([losses["fixed"][i] for i in idx])
        low, high = _interval([weeks[i] for i in idx], (a - b).tolist(), resamples, seed)
        return {"n": len(a), "mean_difference": float((a - b).mean()), "relative_change": float(a.mean() / b.mean() - 1),
                "interval95": [low, high]}

    leagues = sorted({r["league"] for r in test})
    return {"fitted": {"pooled": pooled, "league": league, "league_shrunk": shrunk},
            "variants": {name: float(np.mean(v)) for name, v in losses.items()},
            "comparisons": {f"{name}_vs_fixed": compare(name) for name in ("pooled", "league", "league_shrunk")},
            "by_league": {lg: {name: compare(name, [i for i, r in enumerate(test) if r["league"] == lg])
                               for name in ("pooled", "league_shrunk")} for lg in leagues},
            "train": len(train), "test": len(test)}


def load():
    db = sqlite3.connect(f"file:{ROOT / 'Index/platform.db'}?mode=ro", uri=True, timeout=60)
    scores = {fid: (hg, ag) for fid, hg, ag in db.execute(
        "SELECT api_football_id, home_goals, away_goals FROM fixtures WHERE home_goals IS NOT NULL")}
    rows = defaultdict(list)
    for path in sorted(glob.glob(str(RUNS / "*/forecasts.jsonl.gz"))):
        with gzip.open(path, "rt") as handle:
            for line in handle:
                r = json.loads(line)
                if r.get("market") != "goals" or r.get("candidate_id") != "control" or int(r["fixture_id"]) not in scores:
                    continue
                h, a = r["scores"]["distribution"]["goal_means"]
                hg, ag = scores[int(r["fixture_id"])]
                rows[str(r["year"])].append({"home_mean": h, "away_mean": a, "home_goals": hg, "away_goals": ag,
                                             "league": r["league"], "week": r["week"]})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    rows = load()
    result = evaluate(rows["2022"], rows["2023"])
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "rho.json").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
