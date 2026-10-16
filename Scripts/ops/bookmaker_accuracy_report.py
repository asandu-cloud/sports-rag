"""Write the historical bookmaker-accuracy report (rework A1 step 4).

    python -m Scripts.ops.bookmaker_accuracy_report --out Research/bookmaker-accuracy-<date>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from Scripts.data_platform.features import bookmaker_accuracy as acc
from Scripts.data_platform.features.market_baseline import DEFAULT_DB


def build(db_path=DEFAULT_DB, *, resamples=2000):
    return {"version": acc.VERSION, "events": {event: acc.evaluate(db_path, event, resamples=resamples)
                                                for event in acc.EVENTS}}


def markdown(result):
    lines = [f"# Bookmaker accuracy at closing ({result['version']})", "",
             "Each bookmaker's closing fair probability against the consensus of the other bookmakers on the "
             "same fixtures (football-data.co.uk history linked to our fixtures). Difference = bookmaker log loss "
             "minus consensus log loss: negative means the bookmaker beat the rest. 95% calendar-week bootstrap.", ""]
    for event, rows in result["events"].items():
        lines += [f"## {event}", "", "| Bookmaker | Fixtures | Weeks | Seasons | Difference | 95% interval | Relative | Margin | Verdict |",
                  "|---|---:|---:|---|---:|---|---:|---:|---|"]
        for r in rows:
            low, high = r["interval95"]
            verdict = "beats consensus" if r["beats_consensus"] else "worse" if r["worse_than_consensus"] else "no difference"
            seasons = f"{r['seasons'][0]}–{r['seasons'][-1]}" if r["seasons"] else "–"
            lines.append(f"| {r['bookmaker']} | {r['fixtures']} | {r['weeks']} | {seasons} | {r['mean_difference']:+.5f} | "
                         f"{low:+.5f} to {high:+.5f} | {r['relative_change'] * 100:+.2f}% | {r['mean_margin'] * 100:.1f}% | {verdict} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = build()
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "bookmaker-accuracy.json").write_text(json.dumps(result, indent=2))
        (args.out / "bookmaker-accuracy.md").write_text(markdown(result))
    print(markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
