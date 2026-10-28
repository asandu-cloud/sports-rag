"""Write the Spix scorecard (rework A1 step 5) for review.

Reads published picks from the platform database (the measured cohort: picks
published after the measurement activation boundary) and writes
``Research/scorecards/<as-of>/scorecard.json`` and ``scorecard.md``.

    python -m Scripts.ops.scorecard [--as-of 2026-10-19] [--resamples 2000] [--method power]

``--method`` recomputes the market's probabilities at publication and at close from the prices stored with each
pick (default: ``reference_price.MEASUREMENT_METHOD``); a method other than the default writes
``scorecard-<method>.json`` and ``.md`` beside the default files.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

from Scripts.ops.odds_archive import ROOT

OUT_DIR = ROOT / "Research/scorecards"


def load_rows():
    from Scripts.data_platform.repositories.predictions import PredictionRepository
    from Scripts.data_platform.services.measurement_runtime import CONTROL, MeasurementRuntime
    repo = PredictionRepository()
    boundary = int(MeasurementRuntime(repo._factory).state(CONTROL).get("after_recommendation_id") or 0)
    rows = repo.get_recent(days=3650, published_only=True, publication_scope="all", limit=10 ** 7)
    return [r for r in rows if (r.get("recommendation_id") or 0) > boundary], boundary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    from Scripts.data_platform.reference_price import MEASUREMENT_METHOD, METHOD_VERSIONS
    parser.add_argument("--method", choices=sorted(METHOD_VERSIONS), default=MEASUREMENT_METHOD)
    args = parser.parse_args(argv)
    from Scripts.data_platform.scorecard import build, markdown
    rows, boundary = load_rows()
    card = build(rows, as_of=args.as_of, resamples=args.resamples, after_recommendation_id=boundary,
                 method=args.method)
    folder = args.out / args.as_of
    folder.mkdir(parents=True, exist_ok=True)
    name = "scorecard" if args.method == MEASUREMENT_METHOD else f"scorecard-{args.method}"
    (folder / f"{name}.json").write_text(json.dumps(card, indent=2, default=str))
    (folder / f"{name}.md").write_text(markdown(card))
    print(markdown(card))
    print(f"\nwritten to {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
