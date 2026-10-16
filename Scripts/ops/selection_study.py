"""A5.2 study: which selection policy publishes honest probabilities and real value? (D30)

Replays the policies of ``data_platform.selection_policies`` on every side of every
evaluated line (latest pre-kickoff read per fixture and stage) and reports, per
policy and stage: picks, Spix's stated probability vs the market's vs what happened
(actual minus stated, 95% interval over fixtures), the expected value each policy
claimed at the best price, and the expected value left at the close (closing
consensus times the price taken, minus one), with its interval.

For the same reads it reports how the best-price bookmaker's own price stood at the
close: shorter means the price was real and was corrected; the same means stable or a
stale quote (A5.3 checks quote age). Closing values are reported for pre-match reads only: lineup reads come about 20
minutes before kickoff, after API-Football's final pre-kickoff snapshot. Returns are
saved but not summarised (one week; every line of a fixture settles on one score).

    python -m Scripts.ops.selection_study --out Research/selection-<date>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from Scripts.data_platform import disagreement as dis
from Scripts.data_platform import selection_policies as sp
from Scripts.ops.odds_archive import ROOT

MIN_EV = (0.0, 0.02, 0.04)
DISAGREEMENT_FLAG = 0.10


def _interval(picks, statistic, resamples):
    wrapped = [{"fixture_id": c[0]["fixture_id"], "pick": c} for c in picks]
    return dis.fixture_bootstrap(wrapped, lambda s: statistic([x["pick"] for x in s]), resamples=resamples)


def disagreements_to_check(rows, threshold=DISAGREEMENT_FLAG):
    """Per read, markets where Spix is at least ``threshold`` above the market on some side."""
    flagged, reads = {}, {(r["fixture_id"], r["stage"]) for r in rows}
    for r in rows:
        if r["p_spix"] - r["p_ref"] >= threshold:
            flagged.setdefault((r["fixture_id"], r["stage"]), set()).add(r["market"])
    counts = [len(flagged.get(read, ())) for read in reads]
    return {"reads": len(reads), "reads_with_any": sum(c > 0 for c in counts),
            "markets_per_read": sum(counts) / len(counts) if counts else None}


def run(rows, *, resamples=1000):
    out = {"lines": len(rows), "fixtures": len({r["fixture_id"] for r in rows}), "stages": {}}
    for stage in sorted({r["stage"] for r in rows}):
        stage_rows = [r for r in rows if r["stage"] == stage]
        result = {"disagreements_to_check": disagreements_to_check(stage_rows), "policies": {}}
        for policy in sp.policies():
            per_threshold = {}
            for min_ev in MIN_EV:
                picks = sp.pick(stage_rows, policy, min_ev=min_ev)
                m = sp.evaluate(picks)
                if picks:
                    m["gap_interval95"] = _interval(picks, sp.stated_gap, resamples)
                    if stage == "pre_match" and m.get("closing_picks"):
                        m["ev_at_close_interval95"] = _interval(picks, sp.ev_at_close, resamples)
                per_threshold[f"{min_ev:.2f}"] = m
            result["policies"][policy.name] = per_threshold
        out["stages"][stage] = result
    return out


def summary(result, min_ev="0.02"):
    def pct(v):
        return "n/a" if v is None else f"{v * 100:+.1f}"
    lines = []
    for stage, s in result["stages"].items():
        d = s["disagreements_to_check"]
        lines += [f"\n{stage} (min EV {min_ev}); disagreements of 10+ pts: {d['reads_with_any']} of {d['reads']} reads, "
                  f"{d['markets_per_read']:.2f} markets per read",
                  "| Policy | Picks (fixtures) | Stated | Market | Actual | Actual − stated, 95% | Claimed EV | EV at close, 95% |",
                  "|---|---|---|---|---|---|---|---|"]
        lines_moves = []
        for name, by_ev in s["policies"].items():
            m = by_ev[min_ev]
            moves = m.get("best_book_at_close") or {}
            if stage == "pre_match" and sum(moves.values()):
                lines_moves.append(f"{name}: best-price book at close shorter {moves['shorter']}, same {moves['same']}, "
                                   f"longer {moves['longer']}, not quoted {moves['missing']}")
            if not m["picks"]:
                lines.append(f"| {name} | 0 | | | | | | |")
                continue
            gap = m["gap_interval95"]
            close = (f"{pct(m['ev_at_close'])}% ({pct(m['ev_at_close_interval95'][0])} to {pct(m['ev_at_close_interval95'][1])})"
                     if m.get("ev_at_close_interval95") else "n/a")
            lines.append(f"| {name} | {m['picks']} ({m['fixtures']}) | {m['stated'] * 100:.1f}% | {m['market'] * 100:.1f}% | "
                         f"{m['actual'] * 100:.1f}% | {pct(m['actual'] - m['stated'])} ({pct(gap[0])} to {pct(gap[1])}) | "
                         f"{pct(m['claimed_ev'])}% | {close} |")
        lines += lines_moves
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--db", type=Path, default=ROOT / "Index/platform.db")
    parser.add_argument("--backfill", type=Path, default=ROOT / "Index/closing_backfill")
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args(argv)
    rows = dis.build(args.db, backfill_dir=args.backfill, all_sides=True)
    result = run(rows, resamples=args.resamples)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "selection.json").write_text(json.dumps(result, indent=2))
    for min_ev in ("0.02", "0.00", "0.04"):
        print(summary(result, min_ev))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
