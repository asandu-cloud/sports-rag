"""A4.2 study: does per-market Platt scaling improve Spix's own probabilities? (D27)

Data: every evaluated candidate of the latest pre-kickoff match read per fixture and
stage (``calibration_data``, A4.1), turned to one reference side per line
(``calibration.canonical``); corners priced with the stage 1 fixed dispersion. Each
kickoff day is predicted by a fit on the days before it (rolling origin); no
calibration (identity) is the control. Intervals resample whole fixtures.

Gate (D27): out-of-sample calibration error at most 3 points and slope 0.8-1.2.
A calibrator is adopted only if it passes the gate and its log-loss interval
against identity lies below zero.

Two controls: the bookmaker's own de-vigged probability on the same lines (where a
two-sided price exists), and for corners the full count distribution of every
finished fixture's latest read scored under the floors and the fixed dispersion.

    python -m Scripts.ops.calibration_study --out Research/calibration-<date>
"""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import numpy as np
from scipy.special import gammaln

from Scripts.data_platform import calibration as cal
from Scripts.data_platform import calibration_data
from Scripts.ops.odds_archive import ROOT

GATE_ERROR, GATE_SLOPE = 0.03, (0.8, 1.2)
ISOTONIC_MIN_ROWS = 1000


def passes_gate(m):
    return (m["calibration_error"] is not None and m["calibration_error"] <= GATE_ERROR
            and m["slope"] is not None and GATE_SLOPE[0] <= m["slope"] <= GATE_SLOPE[1])


def _metrics(rows, probabilities):
    return cal.metrics([r["y"] for r in rows], [r["w"] for r in rows], probabilities)


def evaluate_market(rows, *, min_train_fixtures=40):
    makers = {"platt": cal.Platt}
    if len(rows) >= ISOTONIC_MIN_ROWS:
        makers["isotonic"] = cal.Isotonic
    out = {"rows": len(rows), "fixtures": len({r["fixture_id"] for r in rows}),
           "full_fit": {"platt": vars(cal.Platt().fit(rows))}}
    test, identity = None, None
    methods = {}
    for name, make in makers.items():
        tested, predicted = cal.rolling_origin(rows, make, min_train_fixtures=min_train_fixtures)
        if not tested:
            return {**out, "tested": 0}
        test, identity = tested, [r["p"] for r in tested]
        methods[name] = predicted
    base_loss = cal.row_log_loss(test, identity)
    out.update({"tested": len(test), "tested_fixtures": len({r["fixture_id"] for r in test}),
                "tested_days": sorted({r["kickoff"][:10] for r in test}),
                "identity": {**_metrics(test, identity), "gate": passes_gate(_metrics(test, identity))}})
    for name, predicted in methods.items():
        m = _metrics(test, predicted)
        low, high = cal.cluster_interval(test, cal.row_log_loss(test, predicted), base_loss)
        m.update({"gate": passes_gate(m), "log_loss_change": m["log_loss"] / out["identity"]["log_loss"] - 1,
                  "difference_interval95": [low, high], "adopt": passes_gate(m) and high < 0})
        out[name] = m
    picked = [i for i, r in enumerate(test) if r["selected"]]
    if picked:
        sub = [test[i] for i in picked]
        out["selected"] = {"rows": len(sub), "identity": _metrics(sub, [identity[i] for i in picked]),
                           "platt": _metrics(sub, [methods["platt"][i] for i in picked])}
    return out


def market_benchmark(rows):
    """Spix against the bookmaker's de-vigged probability on the lines priced two-sided."""
    priced = [r for r in rows if r["market_p"] is not None]
    if not priced:
        return None
    spix, market = [r["p"] for r in priced], [r["market_p"] for r in priced]
    low, high = cal.cluster_interval(priced, cal.row_log_loss(priced, spix), cal.row_log_loss(priced, market))
    return {"rows": len(priced), "fixtures": len({r["fixture_id"] for r in priced}),
            "spix": _metrics(priced, spix), "market": _metrics(priced, market), "difference_interval95": [low, high]}


def _negbin_log_pmf(k, mean, variance):
    variance = np.maximum(variance, mean + 1e-9)
    r = mean * mean / (variance - mean)
    p = r / (r + mean)
    return gammaln(k + r) - gammaln(r) - gammaln(k + 1) + r * np.log(p) + k * np.log(1 - p)


def corner_count_scores(db_path, *, resamples=2000, seed=20260928):
    """Per stage: mean log loss of the actual corner count under the stored (floor) variance and the fixed one."""
    from Scripts.rag_ingest.core.prediction_guardrails import FIXED_DISPERSIONS
    alpha = FIXED_DISPERSIONS["corners"]
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60)) as db:
        reads = [dict(zip(("fixture_api_id", "stage", "evaluated_at", "kickoff", "read_json"), r)) for r in db.execute(
            "SELECT fixture_api_id, stage, evaluated_at, kickoff, read_json FROM match_reads")]
    outcomes = calibration_data.fixture_outcomes(db_path)
    by_stage = {}
    for read in calibration_data.latest_reads(reads):
        corners = (outcomes.get(str(read["fixture_api_id"])) or {}).get("corners")
        if corners is None:
            continue
        for res in json.loads(read["read_json"]).get("canonical_results", []):
            projection = res.get("projection") or {}
            if (res.get("market") or {}).get("group") == "corners" and projection.get("value"):
                mean = float(projection["value"])
                by_stage.setdefault(read["stage"], []).append(
                    (corners, mean, float(projection.get("variance") or mean)))
    out = {}
    for stage, records in sorted(by_stage.items()):
        k, mean, floors = (np.array(v, dtype=float) for v in zip(*records))
        fixed_loss = -_negbin_log_pmf(k, mean, mean + alpha * mean * mean)
        floor_loss = -_negbin_log_pmf(k, mean, floors)
        delta = fixed_loss - floor_loss
        draws = np.random.default_rng(seed).integers(0, len(delta), size=(resamples, len(delta)))
        out[stage] = {"fixtures": len(records), "fixed": float(fixed_loss.mean()), "floors": float(floor_loss.mean()),
                      "difference_interval95": np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist(),
                      "mean_corr": float(np.corrcoef(mean, k)[0, 1])}
    return out


def run(dataset, *, min_train_fixtures=40, db_path=None):
    rows = cal.canonical(dataset)
    by_market = {m: [r for r in rows if r["market"] == m] for m in cal.SCOPE}
    result = {"scope": list(cal.SCOPE), "excluded": "1X2 and goals O/U 2.5 (market-anchored, D21)",
              "min_train_fixtures": min_train_fixtures,
              "markets": {m: evaluate_market(r, min_train_fixtures=min_train_fixtures) for m, r in by_market.items() if r},
              "market_benchmark": {m: market_benchmark(r) for m, r in by_market.items() if r}}
    floors = [r for r in cal.canonical(dataset, live_dispersion=False) if r["market"] == "corners"]
    fixed = by_market.get("corners") or []
    if floors and fixed:
        low, high = cal.cluster_interval(fixed, cal.row_log_loss(fixed, [r["p"] for r in fixed]),
                                         cal.row_log_loss(floors, [r["p"] for r in floors]))
        result["corners_dispersion"] = {"lines": {"floors": _metrics(floors, [r["p"] for r in floors]),
                                                  "fixed": _metrics(fixed, [r["p"] for r in fixed]),
                                                  "difference_interval95": [low, high]}}
        if db_path is not None:
            result["corners_dispersion"]["counts"] = corner_count_scores(db_path)
    return result


def summary(result):
    lines = ["| Market | Tested rows (fixtures) | Identity error / slope | Platt error / slope | Platt log loss | 95% interval | Adopt |",
             "|---|---|---|---|---|---|---|"]
    for market, r in result["markets"].items():
        if not r.get("tested"):
            lines.append(f"| {market} | 0 | | | | | no |")
            continue
        i, p = r["identity"], r["platt"]
        lines.append(f"| {market} | {r['tested']} ({r['tested_fixtures']}) | {i['calibration_error'] * 100:.1f} / {i['slope']:.2f} | "
                     f"{p['calibration_error'] * 100:.1f} / {p['slope']:.2f} | {p['log_loss_change'] * 100:+.2f}% | "
                     f"{p['difference_interval95'][0]:+.4f} to {p['difference_interval95'][1]:+.4f} | {'yes' if p['adopt'] else 'no'} |")
    lines += ["", "| Market | Priced rows (fixtures) | Spix error / slope | Market error / slope | Spix − market log loss, 95% |",
              "|---|---|---|---|---|"]
    for market, b in result["market_benchmark"].items():
        if b:
            lines.append(f"| {market} | {b['rows']} ({b['fixtures']}) | {b['spix']['calibration_error'] * 100:.1f} / "
                         f"{b['spix']['slope']:.2f} | {b['market']['calibration_error'] * 100:.1f} / {b['market']['slope']:.2f} | "
                         f"{b['difference_interval95'][0]:+.4f} to {b['difference_interval95'][1]:+.4f} |")
    for stage, c in (result.get("corners_dispersion", {}).get("counts") or {}).items():
        lines.append(f"\nCorner counts, {stage}: {c['fixtures']} fixtures, fixed {c['fixed']:.4f} vs floors {c['floors']:.4f} "
                     f"({c['difference_interval95'][0]:+.4f} to {c['difference_interval95'][1]:+.4f})")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--db", type=Path, default=ROOT / "Index/platform.db")
    parser.add_argument("--min-train-fixtures", type=int, default=40)
    args = parser.parse_args(argv)
    result = run(calibration_data.build(args.db), min_train_fixtures=args.min_train_fixtures, db_path=args.db)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "calibration.json").write_text(json.dumps(result, indent=2))
    print(summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
