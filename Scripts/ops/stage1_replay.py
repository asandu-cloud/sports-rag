"""Stage 1 review: the corners selector replayed on audited reads with today's variance and with the fixed dispersion.

For every audited read (latest before kickoff, per fixture and stage), the engine's own
``select_best_total_recommendation`` runs twice on the corner prices the decision audit
stored: once with the stored variance (today's floors) and once with the A3.5 fixed
dispersion (variance = mean + 0.025 * mean^2). Each variant's pick is graded and compared
with the bookmakers' consensus at the read (power margin removal, median across books) and
at the close (API-Football's final pre-kickoff snapshot).

    python -m Scripts.ops.stage1_replay --out Research/stage1-replay-<date>
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

from Scripts.data_platform import disagreement as dis
from Scripts.data_platform.calibration_data import _label, fixture_outcomes, latest_read_rows
from Scripts.data_platform.reference_price import audit_books, devig_consensus, norm
from Scripts.data_platform.scorecard import outcome_weight
from Scripts.data_platform.services.refresh_coordination import data_access
from Scripts.ops.odds_archive import ROOT

MARKET_KEY = "totals_corners_over_under"


def _engine():
    path = str(ROOT / "Scripts/rag_ingest")
    if path not in sys.path:
        sys.path.insert(0, path)
    from core import line_selection
    from core.prediction_guardrails import FIXED_DISPERSIONS
    return line_selection, FIXED_DISPERSIONS["corners"]


def replay(result):
    """{"floors": pick or None, "fixed": pick or None} for one stored corners result."""
    line_selection, alpha = _engine()
    projection = result.get("projection") or {}
    mean, stored = projection.get("value"), projection.get("variance")
    options = ((result.get("context") or {}).get("decision_audit") or {}).get("input_options") or []
    if not mean or not options:
        return None
    out = {}
    for name, variance in (("floors", stored), ("fixed", mean + alpha * mean * mean)):
        chosen = line_selection.select_best_total_recommendation(deepcopy(options), mean, variance)["bet_recommendation"]
        out[name] = None if chosen is None else {
            "side": str(chosen["side"]).lower(), "point": float(chosen["point"]), "odds": chosen["odds"],
            "bookmaker": chosen.get("bookmaker"), "p": chosen["_model_prob"]}
    return {**out, "books": audit_books(options, MARKET_KEY)}


def _consensus(books, side, point):
    _, consensus, _ = devig_consensus(books, side, point, spreads=False, method="power")
    return None if consensus is None else consensus["fair_probability"]


def build(db_path, backfill_dir):
    outcomes = fixture_outcomes(db_path)
    snapshots = dis.closing_snapshots(db_path, backfill_dir)
    rows = []
    for read in latest_read_rows(db_path):
        fid = str(read["fixture_api_id"])
        outcome = outcomes.get(fid)
        if not outcome or outcome.get("corners") is None:
            continue
        data = json.loads(read["read_json"])
        for res in data.get("canonical_results", []):
            if (res.get("market") or {}).get("group") != "corners":
                continue
            picks = replay(res)
            if not picks:
                continue
            fixture = res.get("fixture") or {}
            for variant in ("floors", "fixed"):
                pick = picks[variant]
                if pick is None:
                    continue
                y, w = outcome_weight(_label({"market": "corners", "side": pick["side"], "point": pick["point"]}, outcome))
                if w <= 0:
                    continue
                line = {"market": "corners", "market_key": MARKET_KEY, "side": norm(pick["side"]), "point": pick["point"],
                        "home": fixture.get("home_team"), "away": fixture.get("away_team")}
                payload = snapshots.get(fid)
                rows.append({"variant": variant, "fixture_id": fid, "stage": read["stage"], **pick, "y": y, "w": w,
                             "p_ref": _consensus(picks["books"], line["side"], pick["point"]),
                             "p_close": dis.closing_probability(payload, line) if payload else None})
    return rows


def evaluate(rows, *, resamples=2000):
    out = {}
    for variant in ("floors", "fixed"):
        sub = [r for r in rows if r["variant"] == variant]
        if not sub:
            continue
        w = np.array([r["w"] for r in sub])
        priced = [r for r in sub if r["p_ref"] is not None]
        closed = [r for r in sub if r["p_close"] is not None]
        gap = [{"fixture_id": r["fixture_id"], "w": r["w"], "v": r["y"] - r["p"]} for r in sub]
        out[variant] = {
            "picks": len(sub), "fixtures": len({r["fixture_id"] for r in sub}),
            "median_odds": float(np.median([r["odds"] for r in sub])),
            "stated": float((w * [r["p"] for r in sub]).sum() / w.sum()),
            "actual": float((w * [r["y"] for r in sub]).sum() / w.sum()),
            "market": float(np.mean([r["p_ref"] for r in priced])) if priced else None,
            "stated_on_priced": float(np.mean([r["p"] for r in priced])) if priced else None,
            "actual_minus_stated_interval95": dis.fixture_bootstrap(
                gap, lambda s: sum(x["w"] * x["v"] for x in s) / sum(x["w"] for x in s), resamples=resamples),
            "ev_at_close": float(np.mean([r["p_close"] * r["odds"] - 1 for r in closed])) if closed else None,
            "ev_at_close_interval95": dis.fixture_bootstrap(
                closed, lambda s: float(np.mean([r["p_close"] * r["odds"] - 1 for r in s])), resamples=resamples)
            if closed else None,
            "closing_picks": len(closed)}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--db", type=Path, default=ROOT / "Index/platform.db")
    parser.add_argument("--backfill", type=Path, default=ROOT / "Index/closing_backfill")
    args = parser.parse_args(argv)
    with data_access():  # never read while refresh-season writes
        rows = build(args.db, args.backfill)
    result = {"by_stage": {stage: evaluate([r for r in rows if r["stage"] == stage]) for stage in sorted({r["stage"] for r in rows})},
              "pooled": evaluate(rows)}
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "stage1_replay.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
