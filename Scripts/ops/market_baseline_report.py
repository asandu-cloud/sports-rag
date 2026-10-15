"""Compare saved research forecasts with the betting market (protocol amendment market-baseline.v1).

Reads research forecast files (``forecasts.jsonl.gz`` rows with ``candidate_id``,
``market``, ``fixture_id``, ``week`` and ``scores``), keeps the goals forecasts,
and scores each candidate's 1X2 and over/under 2.5 probabilities against the
market's closing fair probabilities on the same fixtures.

    python -m Scripts.ops.market_baseline_report \\
        --forecasts 'Research/phase4-finalists-2026-09-29/runs/2023-Q*/forecasts.jsonl.gz' \\
        --out Research/market-baseline-<date>
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import gzip
import json
from pathlib import Path

from Scripts.data_platform.features import market_baseline as mb


def load_forecasts(paths):
    forecasts = defaultdict(list)
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("market") == "goals":
                        forecasts[row["candidate_id"]].append(row)
    return dict(forecasts)


def build(forecasts, *, db_path=mb.DEFAULT_DB, timing="closing"):
    fixtures = {int(r["fixture_id"]) for rows in forecasts.values() for r in rows}
    market = mb.market_probabilities(db_path, fixtures, timing=timing)
    return {"version": mb.VERSION, "timing": timing, "fixtures": len(fixtures), "with_market": len(market),
            "candidates": {name: {event: mb.compare_with_market(rows, market, event) for event in mb.EVENTS}
                           for name, rows in sorted(forecasts.items())}}


def markdown(result):
    lines = [f"# Research forecasts vs the market ({result['version']}, {result['timing']} prices)", "",
             f"{result['with_market']} of {result['fixtures']} fixtures have a market price. "
             "Gap = forecast log loss minus market log loss (negative: the forecast beat the market); "
             "95% paired calendar-week bootstrap.", "",
             "| Candidate | Event | Paired | Weeks | Forecast | Market | Relative | 95% gap | Verdict |",
             "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for name, events in result["candidates"].items():
        for event, r in events.items():
            if not r["paired"]:
                lines.append(f"| {name} | {event} | 0 | 0 | – | – | – | – | no market prices |")
                continue
            low, high = r["paired_week_delta95"]
            verdict = "beats market" if r["beats_market"] else "worse than market" if r["worse_than_market"] else "inconclusive"
            lines.append(f"| {name} | {event} | {r['paired']} | {r['weeks']} | {r['forecast_logloss']:.4f} | "
                         f"{r['market_logloss']:.4f} | {r['relative_change'] * 100:+.2f}% | {low:+.4f} to {high:+.4f} | {verdict} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--forecasts", action="append", required=True, help="glob of forecasts.jsonl.gz (repeatable)")
    parser.add_argument("--timing", choices=("closing", "best_available"), default="closing")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    paths = sorted({p for pattern in args.forecasts for p in glob.glob(pattern)})
    if not paths:
        parser.error("no forecast files matched")
    result = build(load_forecasts(paths), timing=args.timing)
    result["sources"] = paths
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "market-baseline.json").write_text(json.dumps(result, indent=2))
        (args.out / "market-baseline.md").write_text(markdown(result))
    print(markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
