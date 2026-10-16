"""A5.1 study: is Spix's disagreement with the market information or error?

Every line the engine evaluated (``data_platform.disagreement``), on the side Spix
prefers, with the market's consensus fair probability at the read (power margin
removal, median across books) and at close (API-Football's final pre-kickoff
snapshot). Reports, per market and pooled:

* by size of disagreement: what Spix said, what the market said, what happened, and
  where the market closed;
* c, the weight the outcomes give Spix's deviation from the market, and k, the share
  of that deviation the market closes before kickoff; 95% intervals resample fixtures.

The return at the best available price is saved per bucket but not summarised: every
line of a fixture settles on the same score, so a week of returns is mostly noise.

    python -m Scripts.ops.disagreement_study --out Research/disagreement-<date>
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from Scripts.data_platform import disagreement as dis
from Scripts.ops.odds_archive import ROOT

BUCKETS = ((0.0, 0.025), (0.025, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 1.01))


def _hours_before(row):
    kickoff = datetime.fromisoformat(row["kickoff"].replace("Z", "+00:00"))
    evaluated = datetime.fromisoformat(row["evaluated_at"].replace(" ", "T").replace("Z", "+00:00"))
    if evaluated.tzinfo is None:
        evaluated = evaluated.replace(tzinfo=kickoff.tzinfo)
    return (kickoff - evaluated).total_seconds() / 3600


def bucket_table(rows):
    out = []
    for low, high in BUCKETS:
        sub = [r for r in rows if low <= r["p_spix"] - r["p_ref"] < high]
        if not sub:
            continue
        w = np.array([r["w"] for r in sub])
        closed = [r for r in sub if r["p_close"] is not None]
        priced = [r for r in sub if r["best_odds"]]
        out.append({"disagreement": [low, high], "rows": len(sub), "fixtures": len({r["fixture_id"] for r in sub}),
                    "spix": float((w * [r["p_spix"] for r in sub]).sum() / w.sum()),
                    "market": float((w * [r["p_ref"] for r in sub]).sum() / w.sum()),
                    "actual": float((w * [r["y"] for r in sub]).sum() / w.sum()),
                    "closing_rows": len(closed),
                    "close": float(np.mean([r["p_close"] for r in closed])) if closed else None,
                    "market_at_read_closed": float(np.mean([r["p_ref"] for r in closed])) if closed else None,
                    "moved_toward_spix": float(np.mean([r["p_close"] > r["p_ref"] for r in closed])) if closed else None,
                    "moved_away": float(np.mean([r["p_close"] < r["p_ref"] for r in closed])) if closed else None,
                    "return_at_best_price": float(np.mean([r["w"] * (r["y"] * r["best_odds"] - 1) for r in priced]))
                    if priced else None,
                    "selected": sum(r["selected"] for r in sub)})
    return out


def tests(rows, *, resamples=1000):
    closed = [r for r in rows if r["p_close"] is not None]
    return {"rows": len(rows), "fixtures": len({r["fixture_id"] for r in rows}),
            "c": dis.fit_outcome_weight(rows),
            "c_interval95": dis.fixture_bootstrap(rows, dis.fit_outcome_weight, resamples=resamples),
            "closing_rows": len(closed), "closing_fixtures": len({r["fixture_id"] for r in closed}),
            "k": dis.fit_closing_move(closed),
            "k_interval95": dis.fixture_bootstrap(closed, dis.fit_closing_move, resamples=resamples) if closed else None}


def run(rows, *, resamples=1000):
    preferred = [r for r in rows if r["p_spix"] >= r["p_ref"]]
    markets = sorted({r["market"] for r in rows})
    hours = {stage: float(np.median([_hours_before(r) for r in rows if r["stage"] == stage]))
             for stage in sorted({r["stage"] for r in rows})}
    return {"lines": len(rows), "fixtures": len({r["fixture_id"] for r in rows}), "median_hours_before_kickoff": hours,
            "pooled": {"tests": tests(rows, resamples=resamples), "buckets": bucket_table(preferred)},
            "by_stage": {s: tests([r for r in rows if r["stage"] == s], resamples=resamples) for s in hours},
            "markets": {m: {"tests": tests([r for r in rows if r["market"] == m], resamples=resamples),
                            "buckets": bucket_table([r for r in preferred if r["market"] == m])} for m in markets},
            "selected": bucket_table([r for r in preferred if r["selected"]])}


def summary(result):
    def fmt(interval):
        return "n/a" if not interval or interval[0] is None else f"{interval[0]:+.2f} to {interval[1]:+.2f}"
    lines = ["| Market | Lines (fixtures) | c (outcomes) | 95% | k (closing move) | 95% |", "|---|---|---|---|---|---|"]
    for name, t in [("all", result["pooled"]["tests"])] + [(m, v["tests"]) for m, v in result["markets"].items()]:
        c = "n/a" if t["c"] is None else f"{t['c']:+.2f}"
        k = "n/a" if t["k"] is None else f"{t['k']:+.2f}"
        lines.append(f"| {name} | {t['rows']} ({t['fixtures']}) | {c} | {fmt(t['c_interval95'])} | {k} | {fmt(t['k_interval95'])} |")
    lines += ["", "| Disagreement (pts) | Lines | Spix | Market | Actual | Closed toward Spix / away |", "|---|---|---|---|---|---|"]
    for b in result["pooled"]["buckets"]:
        moved = "n/a" if b["moved_toward_spix"] is None else f"{b['moved_toward_spix'] * 100:.0f}% / {b['moved_away'] * 100:.0f}%"
        lines.append(f"| {b['disagreement'][0] * 100:.1f}–{min(b['disagreement'][1], 1) * 100:.1f} | {b['rows']} | "
                     f"{b['spix'] * 100:.1f}% | {b['market'] * 100:.1f}% | {b['actual'] * 100:.1f}% | {moved} |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--db", type=Path, default=ROOT / "Index/platform.db")
    parser.add_argument("--backfill", type=Path, default=ROOT / "Index/closing_backfill")
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args(argv)
    rows = dis.build(args.db, backfill_dir=args.backfill)
    result = run(rows, resamples=args.resamples)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "disagreement.json").write_text(json.dumps(result, indent=2))
    print(summary(result))
    for stage, t in result["by_stage"].items():
        print(f"\n{stage} (median {result['median_hours_before_kickoff'][stage]:.1f} h before kickoff): c {t['c']:+.2f} "
              f"({t['c_interval95'][0]:+.2f} to {t['c_interval95'][1]:+.2f}), k {t['k']:+.3f} "
              f"({t['k_interval95'][0]:+.3f} to {t['k_interval95'][1]:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
