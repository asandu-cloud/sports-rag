"""D29 gate study for A3.7 stage 4: M1 against Spix and the market on the audited goal markets.

Per market (goal lines other than 2.5, Asian handicap, BTTS): weighted log loss of Spix's
probability, M1's (solved again from each read's stored prices with the engine's code) and
the bookmakers' consensus; M1 minus Spix and M1 minus the consensus with 95% intervals that
resample fixtures; calibration error and slope. The gate passes in a market when the
M1-minus-Spix interval lies below zero. Handicaps are also shown without the line M1 is
fitted to, where M1 matches the market by construction.

    python -m Scripts.ops.m1_gate_study --out Research/m1-gate-<date>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from Scripts.data_platform import calibration as cal
from Scripts.data_platform import m1_gate
from Scripts.data_platform.services.refresh_coordination import data_access
from Scripts.ops.odds_archive import ROOT


def evaluate(rows, *, resamples=2000):
    if not rows:
        return {"lines": 0}
    w = np.array([r["w"] for r in rows])
    losses = {name: [m1_gate.log_loss(r["y"], r[key]) for r in rows]
              for name, key in (("spix", "p_spix"), ("m1", "p_m1"), ("market", "p_ref"))}
    mean = {name: float((w * v).sum() / w.sum()) for name, v in losses.items()}
    vs_spix = cal.cluster_interval(rows, losses["m1"], losses["spix"], resamples=resamples)
    vs_market = cal.cluster_interval(rows, losses["m1"], losses["market"], resamples=resamples)
    metrics = {name: cal.metrics([r["y"] for r in rows], [r["w"] for r in rows], [r[key] for r in rows])
               for name, key in (("spix", "p_spix"), ("m1", "p_m1"), ("market", "p_ref"))}
    return {"lines": len(rows), "fixtures": len({r["fixture_id"] for r in rows}), "log_loss": mean,
            "m1_vs_spix": {"change": mean["m1"] / mean["spix"] - 1, "interval95": vs_spix},
            "m1_vs_market": {"change": mean["m1"] / mean["market"] - 1, "interval95": vs_market},
            "calibration": {name: {"error": m["calibration_error"], "slope": m["slope"]} for name, m in metrics.items()},
            "gate": vs_spix[1] < 0}


def run(rows, unsolved, *, resamples=2000):
    groups = {"goals (not 2.5)": [r for r in rows if r["market"] == "goals"],
              "spreads": [r for r in rows if r["market"] == "spreads"],
              "spreads, other lines": [r for r in rows if r["market"] == "spreads" and not r["fitted_line"]],
              "btts": [r for r in rows if r["market"] == "btts"]}
    return {"reads_without_m1": unsolved, "lines": len(rows), "fixtures": len({r["fixture_id"] for r in rows}),
            "median_fit_error": float(np.median([r["fit_error"] for r in rows])) if rows else None,
            "markets": {name: evaluate(sub, resamples=resamples) for name, sub in groups.items()},
            "by_stage": {stage: {name: evaluate([r for r in sub if r["stage"] == stage], resamples=resamples)
                                 for name, sub in groups.items()}
                         for stage in sorted({r["stage"] for r in rows})}}


def summary(result):
    lines = [f"{result['lines']} lines, {result['fixtures']} fixtures; reads where M1 could not be solved: "
             f"{result['reads_without_m1']}; median fit error {result['median_fit_error']:.2e}", "",
             "| Market | Lines (fixtures) | Log loss Spix / M1 / market | M1 vs Spix, 95% | M1 vs market, 95% | Gate |",
             "|---|---|---|---|---|---|"]
    for name, m in result["markets"].items():
        if not m.get("lines"):
            lines.append(f"| {name} | 0 | | | | |")
            continue
        ll, a, b = m["log_loss"], m["m1_vs_spix"], m["m1_vs_market"]
        lines.append(f"| {name} | {m['lines']} ({m['fixtures']}) | {ll['spix']:.4f} / {ll['m1']:.4f} / {ll['market']:.4f} | "
                     f"{a['change'] * 100:+.1f}% ({a['interval95'][0]:+.4f} to {a['interval95'][1]:+.4f}) | "
                     f"{b['change'] * 100:+.1f}% ({b['interval95'][0]:+.4f} to {b['interval95'][1]:+.4f}) | "
                     f"{'pass' if m['gate'] else 'not passed'} |")
    lines += ["", "| Market | Calibration error / slope: Spix | M1 | market |", "|---|---|---|---|"]
    for name, m in result["markets"].items():
        if m.get("lines"):
            c = m["calibration"]
            lines.append(f"| {name} | " + " | ".join(
                f"{c[k]['error'] * 100:.1f} / {c[k]['slope']:.2f}" for k in ("spix", "m1", "market")) + " |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--db", type=Path, default=ROOT / "Index/platform.db")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)
    with data_access():  # never read while refresh-season writes
        rows, unsolved = m1_gate.build(args.db)
    result = run(rows, unsolved, resamples=args.resamples)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "m1_gate.json").write_text(json.dumps(result, indent=2))
    print(summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
