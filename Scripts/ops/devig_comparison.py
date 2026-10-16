"""Which margin-removal method gives the most accurate market probability? (rework A3.1)

On the football-data closing history (fixtures linked to ours, individual
bookmakers, suspect Pinnacle prices excluded) each method removes every book's
margin, the books are combined by median per outcome (as the reference price
does), and the result is scored by log loss on the outcome. Each method is
compared with proportional on the same fixtures, with a calendar-week
bootstrap; calibration error (10 equal-width bins) shows the favourite–longshot
correction directly.

    python -m Scripts.ops.devig_comparison --out Research/devig-comparison-<date>
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

import numpy as np

from Scripts.data_platform import devig
from Scripts.data_platform.features.bookmaker_accuracy import EVENTS, SEED, _interval, load
from Scripts.data_platform.features.market_baseline import DEFAULT_DB

EPS = 1e-15


def _consensus(per_book):
    medians = [statistics.median(p[i] for p in per_book) for i in range(len(per_book[0]))]
    total = sum(medians)
    return [m / total for m in medians]


def _calibration_error(points, bins=10):
    buckets = defaultdict(lambda: [0.0, 0.0, 0])
    for p, hit in points:
        b = buckets[min(int(p * bins), bins - 1)]
        b[0] += p
        b[1] += hit
        b[2] += 1
    total = sum(b[2] for b in buckets.values())
    return sum(abs(b[0] - b[1]) for b in buckets.values()) / total if total else None


def compare(db_path, event, *, resamples=2000, seed=SEED):
    fixtures = load(db_path, event)
    selections = EVENTS[event][1]
    losses, weeks, calibration = defaultdict(list), [], defaultdict(list)
    for entry in fixtures.values():
        books = [prices for prices in entry["books"].values() if all(prices.get(s) for s in selections)]
        if not books:
            continue
        consensus = {}
        for method in devig.METHODS:
            per_book = [devig.fair([prices[s] for s in selections], method) for prices in books]
            per_book = [p for p in per_book if p]
            if not per_book:
                break
            consensus[method] = _consensus(per_book)
        if len(consensus) != len(devig.METHODS):
            continue  # compare every method on the same fixtures
        weeks.append(entry["week"])
        for method, p in consensus.items():
            losses[method].append(-math.log(max(p[entry["y"]], EPS)))
            calibration[method] += [(prob, int(i == entry["y"])) for i, prob in enumerate(p)]
    base = np.array(losses["proportional"])
    rows = []
    for method in devig.METHODS:
        values = np.array(losses[method])
        deltas = values - base
        interval = _interval(weeks, deltas.tolist(), resamples, seed) if method != "proportional" else [0.0, 0.0]
        rows.append({"method": method, "logloss": float(values.mean()),
                     "difference_vs_proportional": float(deltas.mean()), "interval95": interval,
                     "relative_change": float(values.mean() / base.mean() - 1),
                     "calibration_error": _calibration_error(calibration[method])})
    better = [r for r in rows if r["method"] != "proportional" and r["interval95"][1] < 0]
    best = min(better, key=lambda r: r["logloss"])["method"] if better else "proportional"
    return {"event": event, "fixtures": len(weeks), "weeks": len(set(weeks)), "methods": rows, "best": best}


def markdown(results):
    lines = ["# Margin-removal methods on closing history (A3.1)", "",
             "Median across bookmakers after each method; log loss on the outcome; difference to proportional "
             "with a 95% calendar-week bootstrap (negative = better than proportional).", ""]
    for event, result in results.items():
        lines += [f"## {event}: {result['fixtures']} fixtures, {result['weeks']} weeks; best: **{result['best']}**", "",
                  "| Method | Log loss | Difference vs proportional | 95% interval | Relative | Calibration error |",
                  "|---|---:|---:|---|---:|---:|"]
        for r in result["methods"]:
            low, high = r["interval95"]
            lines.append(f"| {r['method']} | {r['logloss']:.5f} | {r['difference_vs_proportional']:+.5f} | "
                         f"{low:+.5f} to {high:+.5f} | {r['relative_change'] * 100:+.3f}% | {r['calibration_error']:.4f} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    results = {event: compare(DEFAULT_DB, event) for event in EVENTS}
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "devig-comparison.json").write_text(json.dumps(results, indent=2))
        (args.out / "devig-comparison.md").write_text(markdown(results))
    print(markdown(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
