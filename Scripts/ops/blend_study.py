"""A3.3: does a fitted blend of the market and Spix beat the market alone? (frozen gate, D16)

Spix's saved 2023 goal forecasts (frozen control) give the 1X2 and over/under 2.5
probabilities. The market at publication is football-data's pre-closing prices
(collected before closing; the exact time is not recorded), power margin removal,
median across bookmakers. Time-ordered folds: fit on earlier quarters, predict
the next. Compared on the predicted fixtures with a calendar-week bootstrap:

* blend vs the market alone (the A3 gate);
* blend vs the market recalibrated with Spix's weight fixed at zero (what Spix adds);
* the closing market vs the blend (how much the market still learns after publication).

    python -m Scripts.ops.blend_study --out Research/blend-<date>
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import gzip
import json
import math
from pathlib import Path

import numpy as np

from Scripts.data_platform import blend
from Scripts.data_platform.features.bookmaker_accuracy import SEED, _interval
from Scripts.data_platform.features.market_baseline import DEFAULT_DB
from Scripts.ops.market_goals_study import FORECASTS, market_targets

EPS = 1e-15


def _comparison(weeks, a, b, resamples, seed):
    deltas = (np.array(a) - np.array(b)).tolist()
    low, high = _interval(weeks, deltas, resamples, seed)
    return {"n": len(deltas), "mean_difference": float(np.mean(deltas)),
            "relative_change": float(np.mean(a) / np.mean(b) - 1), "interval95": [low, high]}


def cross_validate(rows, *, resamples=2000, seed=SEED, spix_penalty=0.0):
    quarters = sorted({r["quarter"] for r in rows})
    losses, weeks, folds = defaultdict(list), [], []
    for index in range(1, len(quarters)):
        train = [r for r in rows if r["quarter"] < quarters[index]]
        test = [r for r in rows if r["quarter"] == quarters[index]]
        if not train or not test:
            continue
        stack = lambda items, key: np.array([r[key] for r in items], dtype=float)
        y_train = np.array([r["y"] for r in train])
        recal = blend.fit(stack(train, "market"), stack(train, "spix"), y_train, use_spix=False)
        model = blend.fit(stack(train, "market"), stack(train, "spix"), y_train, spix_penalty=spix_penalty)
        folds.append({"trained_on": sorted({r["quarter"] for r in train}), "tested_on": quarters[index],
                      "b": model["b"], "c": model["c"], "recalibration_b": recal["b"], "train": len(train), "test": len(test)})
        predictions = {"market": stack(test, "market"), "closing": stack(test, "closing"),
                       "recalibrated": blend.predict(recal, stack(test, "market"), stack(test, "spix")),
                       "blend": blend.predict(model, stack(test, "market"), stack(test, "spix")),
                       "spix": stack(test, "spix")}
        for name, p in predictions.items():
            losses[name] += [-math.log(max(p[i][r["y"]], EPS)) for i, r in enumerate(test)]
        weeks += [r["week"] for r in test]
    compare = lambda a, b: _comparison(weeks, losses[a], losses[b], resamples, seed)
    return {"fixtures": len(weeks), "weeks": len(set(weeks)), "folds": folds,
            "logloss": {name: float(np.mean(values)) for name, values in losses.items()},
            "blend_vs_market": compare("blend", "market"),
            "blend_vs_recalibrated": compare("blend", "recalibrated"),
            "recalibrated_vs_market": compare("recalibrated", "market"),
            "spix_vs_market": compare("spix", "market"),
            "closing_vs_blend": compare("closing", "blend")}


def load_rows(event, *, pattern=str(FORECASTS), db_path=DEFAULT_DB):
    forecasts = []
    for path in sorted(glob.glob(pattern)):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("market") == "goals" and row.get("candidate_id") == "control":
                    forecasts.append(row)
    ids = {int(r["fixture_id"]) for r in forecasts}
    publication = market_targets(db_path, ids, closing_prices=False)
    closing = market_targets(db_path, ids, closing_prices=True)
    rows = []
    for r in forecasts:
        fid = int(r["fixture_id"])
        pub, close = publication.get(fid, {}), closing.get(fid, {})
        if event == "winner":
            winner = r["scores"]["derived"]["winner"]
            if "1x2" in pub and "1x2" in close:
                rows.append({"quarter": r["fold"], "week": r["week"], "market": pub["1x2"], "closing": close["1x2"],
                             "spix": winner["p"], "y": winner["y"]})
        else:
            over = r["scores"]["totals"]["2.5"]["binary"]
            if 2.5 in pub.get("totals", {}) and 2.5 in close.get("totals", {}):
                pm, pc = pub["totals"][2.5], close["totals"][2.5]
                rows.append({"quarter": r["fold"], "week": r["week"], "market": [pm, 1 - pm], "closing": [pc, 1 - pc],
                             "spix": [over["p"], 1 - over["p"]], "y": 0 if over["y"] else 1})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = {event: cross_validate(load_rows(event)) for event in ("winner", "over2.5")}
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "blend.json").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
