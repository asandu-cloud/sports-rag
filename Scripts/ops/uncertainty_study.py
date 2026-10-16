"""A3.4 study: does evidence-aware variance beat the hand floors and the fixed dispersion?

The frozen control's corners and SoT forecasts (Phase 4 step 2 runs) are joined to
their input snapshots for the matches behind each team's mean. α and κ of
``uncertainty.total_variance`` are fitted on 2022 and evaluated on 2023 against
(a) the control's own variance (with the hand floors) and (b) Phase 4's fixed
dispersion (corners 0.025, SoT 0.01), by negative-binomial log loss of the
actual total, with a calendar-week bootstrap and season-stage slices.

    python -m Scripts.ops.uncertainty_study --out Research/uncertainty-<date>
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import gzip
import json
from pathlib import Path

import numpy as np

from Scripts.data_platform import uncertainty as u
from Scripts.data_platform.features.bookmaker_accuracy import SEED, _interval
from Scripts.ops.odds_archive import ROOT

RUNS = ROOT / "Research/phase4-step2-backtest-2026-09-29/runs"
SNAPSHOTS = ROOT / "Research/phase4-historical-control-2026-09-29/reproduction/snapshots.jsonl"
FIXED_ALPHA = {"corners": 0.025, "sot": 0.01}


def _compare(weeks, a, b, resamples, seed):
    deltas = (np.array(a) - np.array(b)).tolist()
    low, high = _interval(weeks, deltas, resamples, seed)
    return {"n": len(deltas), "mean_difference": float(np.mean(deltas)),
            "relative_change": float(np.mean(a) / np.mean(b) - 1), "interval95": [low, high]}


def evaluate(train, test, *, fixed_alpha, resamples=2000, seed=SEED):
    fit = u.fit(train)
    losses = defaultdict(list)
    for r in test:
        losses["control"].append(-u.log_pmf(r["y"], r["mean"], r["control_variance"]))
        losses["fixed"].append(-u.log_pmf(r["y"], r["mean"], r["mean"] + fixed_alpha * r["mean"] ** 2))
        losses["evidence"].append(-u.log_pmf(r["y"], r["mean"], u.total_variance(r["mean"], fit["alpha"], r["matches"], fit["kappa"])))
    weeks = [r["week"] for r in test]
    stages = {}
    for stage in sorted({r["season_stage"] for r in test}):
        idx = [i for i, r in enumerate(test) if r["season_stage"] == stage]
        pick = lambda name: [losses[name][i] for i in idx]
        stages[stage] = {"n": len(idx), "evidence_vs_control": _compare([weeks[i] for i in idx], pick("evidence"), pick("control"), resamples, seed),
                         "evidence_vs_fixed": _compare([weeks[i] for i in idx], pick("evidence"), pick("fixed"), resamples, seed)}
    return {"fit": fit, "train": len(train), "test": len(test),
            "logloss": {k: float(np.mean(v)) for k, v in losses.items()},
            "evidence_vs_control": _compare(weeks, losses["evidence"], losses["control"], resamples, seed),
            "evidence_vs_fixed": _compare(weeks, losses["evidence"], losses["fixed"], resamples, seed),
            "fixed_vs_control": _compare(weeks, losses["fixed"], losses["control"], resamples, seed),
            "by_season_stage": stages}


def snapshot_matches(path=SNAPSHOTS):
    """{snapshot_id: average effective matches of the two teams}."""
    out = {}
    with open(path) as handle:
        for line in handle:
            snap = json.loads(line)
            fixture, evidence = snap["fixture"], snap.get("history_evidence") or {}
            teams = []
            for team in (fixture["home_team_id"], fixture["away_team_id"]):
                comps = evidence.get(str(team)) or {}
                entry = comps.get(fixture["competition"]) or max(
                    comps.values(), key=lambda c: len((c.get("current") or {}).get("fixture_ids", [])), default=None)
                if entry:
                    teams.append(u.effective_matches({"current": len((entry.get("current") or {}).get("fixture_ids", [])),
                                                      "prior": len((entry.get("prior") or {}).get("fixture_ids", []))}))
            if len(teams) == 2:
                out[snap["snapshot_id"]] = sum(teams) / 2
    return out


def load(market, matches):
    rows = defaultdict(list)
    for path in sorted(glob.glob(str(RUNS / "*/forecasts.jsonl.gz"))):
        with gzip.open(path, "rt") as handle:
            for line in handle:
                r = json.loads(line)
                if r.get("market") != market or r.get("candidate_id") != "control" or r.get("snapshot_id") not in matches:
                    continue
                d = r["scores"]["distribution"]
                rows[str(r["year"])].append({"mean": d["mean"], "control_variance": d["variance"], "y": r["scores"]["target"],
                                             "matches": matches[r["snapshot_id"]], "week": r["week"],
                                             "season_stage": r["season_stage"]})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    matches = snapshot_matches()
    result = {}
    for market, alpha in FIXED_ALPHA.items():
        rows = load(market, matches)
        result[market] = evaluate(rows.get("2022", []), rows.get("2023", []), fixed_alpha=alpha)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "uncertainty.json").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
