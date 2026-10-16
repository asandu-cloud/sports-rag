"""M1 study: do the market's goal expectations predict goals better than the engine? (rework A3.2)

For fixtures with the engine's saved goal means (Phase 4 research forecasts),
M1 is solved from the football-data closing prices (1X2, over/under 2.5 and the
main Asian handicap line; power margin removal, median across bookmakers).
Both sets of means are scored by the Poisson log loss of the actual home and
away goals on the same fixtures, with a calendar-week bootstrap.

    python -m Scripts.ops.market_goals_study --out Research/market-goals-<date>
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import glob
import gzip
import json
import math
from pathlib import Path
import sqlite3
import statistics

import numpy as np

from Scripts.data_platform import devig, market_goals as mg
from Scripts.data_platform.features.bookmaker_accuracy import SEED, _aggregate, _interval, _week
from Scripts.data_platform.features.market_baseline import DEFAULT_DB
from Scripts.ops.odds_archive import ROOT

FORECASTS = ROOT / "Research/phase4-finalists-2026-09-29/runs/2023-Q*/forecasts.jsonl.gz"


def _median_fair(per_book):
    medians = [statistics.median(p[i] for p in per_book) for i in range(len(per_book[0]))]
    total = sum(medians)
    return [m / total for m in medians]


def market_targets(db_path=DEFAULT_DB, fixture_ids=None, *, method="power"):
    """{fixture_api_id: {"1x2", "totals", "handicaps", "week", "goals"}} from closing prices."""
    query = """SELECT m.fixture_api_id, m.match_date, m.fthg, m.ftag, o.market, o.bookmaker, o.selection, o.line, o.price
               FROM odds o JOIN matches m ON m.id = o.match_id
               WHERE m.link_status = 'matched' AND o.closing = 1 AND o.suspect = 0
                 AND o.market IN ('1x2', 'ou25', 'ah')"""
    raw = defaultdict(lambda: {"books": defaultdict(lambda: defaultdict(dict))})
    wanted = None if fixture_ids is None else {int(f) for f in fixture_ids}
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as db:
        for fid, day, hg, ag, market, book, side, line, price in db.execute(query):
            if (wanted is not None and fid not in wanted) or _aggregate(book):
                continue
            entry = raw[fid]
            entry.update(week=_week(day), goals=(hg, ag))
            entry["books"][(market, line)][book][side] = price
    targets = {}
    for fid, entry in raw.items():
        out = {"week": entry["week"], "goals": entry["goals"], "totals": {}, "handicaps": {}}
        x12 = [devig.fair([p["home"], p["draw"], p["away"]], method) for p in entry["books"][("1x2", None)].values()
               if {"home", "draw", "away"} <= set(p)]
        if x12:
            out["1x2"] = _median_fair(x12)
        ou = [devig.fair([p["over"], p["under"]], method) for p in entry["books"][("ou25", None)].values()
              if {"over", "under"} <= set(p)]
        if ou:
            out["totals"][2.5] = _median_fair(ou)[0]
        lines = Counter(line for (market, line), books in entry["books"].items() if market == "ah" for _ in books)
        if lines:
            main = lines.most_common(1)[0][0]
            ah = [devig.fair([p["home"], p["away"]], method) for p in entry["books"][("ah", main)].values()
                  if {"home", "away"} <= set(p)]
            if ah:
                out["handicaps"][main] = _median_fair(ah)[0]
        targets[fid] = out
    return targets


def compare(db_path, engine_means, *, resamples=2000, seed=SEED):
    """Poisson log loss of actual goals: M1 means vs the engine's, on the same fixtures."""
    targets = market_targets(db_path, engine_means)
    m1_loss, engine_loss, weeks, fits = [], [], [], []
    for fid, (home, away) in engine_means.items():
        target = targets.get(int(fid))
        fit = mg.solve(target) if target else None
        if not fit:
            continue
        hg, ag = target["goals"]
        nll = lambda h, a: -math.log(max(mg.poisson_pmf(h, hg), 1e-15)) - math.log(max(mg.poisson_pmf(a, ag), 1e-15))
        m1_loss.append(nll(fit["home"], fit["away"]))
        engine_loss.append(nll(home, away))
        weeks.append(target["week"])
        fits.append(fit["error"])
    if not m1_loss:
        return {"compared": 0, "engine_only": len(engine_means)}
    deltas = (np.array(m1_loss) - np.array(engine_loss)).tolist()
    low, high = _interval(weeks, deltas, resamples, seed)
    return {"compared": len(m1_loss), "engine_only": len(engine_means) - len(m1_loss), "weeks": len(set(weeks)),
            "m1_nll": float(np.mean(m1_loss)), "engine_nll": float(np.mean(engine_loss)),
            "mean_difference": float(np.mean(deltas)), "relative_change": float(np.mean(m1_loss) / np.mean(engine_loss) - 1),
            "interval95": [low, high], "m1_beats_engine": high < 0,
            "fit_error_median": float(statistics.median(fits))}


def engine_goal_means(pattern=str(FORECASTS), candidate="control"):
    means = {}
    for path in sorted(glob.glob(pattern)):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("market") == "goals" and row.get("candidate_id") == candidate:
                    home, away = row["scores"]["distribution"]["goal_means"]
                    means[int(row["fixture_id"])] = (home, away)
    return means


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = {"control_2023": compare(DEFAULT_DB, engine_goal_means())}
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "market-goals.json").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
