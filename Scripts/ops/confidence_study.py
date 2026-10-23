"""A5.3 checks (tracker D44): the spread cut-off for the market-firmness label, and the per-market Spix check.

On the audited reads (``disagreement.build(..., all_sides=True)``, read behind the refresh gate in short statements),
the market-price policy's picks are split by bookmaker agreement. The cut-off is chosen on the earlier weeks
(``label_checks.choose_spread_cutoff``) and must pass on the last week (``label_checks.passes``) before it ships
(``confidence_labels.CUTOFFS``); the Spix check reports per market whether Spix's agreement keeps more value at the
close. A market enters ``SPIX_VOTE_MARKETS`` only by an owner-approved tracker decision, never from this report.

    python -m Scripts.ops.confidence_study --out Research/confidence-<date>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from Scripts.data_platform import disagreement as dis
from Scripts.data_platform import label_checks as lc
from Scripts.data_platform.services.refresh_coordination import data_access
from Scripts.ops.odds_archive import ROOT


def run(rows, *, resamples=1000, min_books=4, min_share=0.2):
    picks = lc.pick_rows(rows)
    train, holdout = lc.split_by_week(picks)
    cutoff = lc.choose_spread_cutoff(train, min_books=min_books, min_share=min_share)
    out = {"picks": len(picks), "fixtures": len({p["fixture_id"] for p in picks}), "train_picks": len(train),
           "holdout_picks": len(holdout), "min_books": min_books, "min_share": min_share, "cutoff": cutoff,
           "passes": False}
    if cutoff is not None:
        split = lambda ps: lc.split_labels(ps, cutoff, min_books=min_books)  # noqa: E731
        out["train"] = lc.compare(*split(train), resamples=resamples)
        out["holdout"] = lc.compare(*split(holdout), resamples=resamples)
        out["passes"] = lc.passes(out["holdout"])
        out["share_high"] = len(split(picks)[0]) / len(picks)
        out["by_market"] = {m: lc.compare(*split([p for p in picks if p["market"] == m]), resamples=resamples)
                            for m in sorted({p["market"] for p in picks})}
    out["spix"] = lc.spix_check(picks, resamples=resamples)
    return out


def _pct(value):
    return "n/a" if value is None else f"{value * 100:+.1f}%"


def _pts(value):
    return "n/a" if value is None else f"{value * 100:+.1f} pts"


def _interval(pair):
    return "n/a" if None in pair else f"{pair[0] * 100:+.1f} to {pair[1] * 100:+.1f}"


def _row(name, c):
    return (f"| {name} | {c['high']['picks']} / {c['low']['picks']} | {_pct(c['high']['value_at_close'])} / "
            f"{_pct(c['low']['value_at_close'])} | {_pct(c['value_diff'])} ({_interval(c['value_diff_interval95'])}) | "
            f"{_pts(c['calibration_diff'])} ({_interval(c['calibration_diff_interval95'])}) |")


def summary(result):
    lines = [f"Market-price picks: {result['picks']} ({result['fixtures']} fixtures); train {result['train_picks']}, "
             f"holdout {result['holdout_picks']}; at least {result['min_books']} bookmakers."]
    if result["cutoff"] is None:
        lines.append("No spread cut-off keeps a fifth of the picks in each label with a close on both sides.")
    else:
        lines += [f"Spread cut-off {result['cutoff']:.4f}; high share {result['share_high']:.0%}; "
                  f"passes on the holdout: {'yes' if result['passes'] else 'no'}", "",
                  "| Firmness | High / low picks | Value at close high / low | Difference, 95% | "
                  "Happened − stated difference, 95% |", "|---|---|---|---|---|",
                  _row("Train", result["train"]), _row("Holdout", result["holdout"])]
        lines += [_row(m, c) for m, c in result["by_market"].items()]
    lines += ["", "| Spix check | Agree / not | Value at close agree / not | Difference, 95% | "
              "Happened − stated difference, 95% | Passes |", "|---|---|---|---|---|---|"]
    for market, c in result["spix"].items():
        if "high" in c:
            lines.append(_row(market, c)[:-1] + f" {'yes' if c['passes'] else 'no'} |")
        else:
            lines.append(f"| {market} | {c['agree']} / {c['disagree']} | too few picks | | | no |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--db", type=Path, default=ROOT / "Index/platform.db")
    parser.add_argument("--backfill", type=Path, default=ROOT / "Index/closing_backfill")
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args(argv)
    with data_access():  # never read while refresh-season writes
        rows = dis.build(args.db, backfill_dir=args.backfill, all_sides=True)
    result = run(rows, resamples=args.resamples)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "confidence.json").write_text(json.dumps(result, indent=2))
    print(summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
