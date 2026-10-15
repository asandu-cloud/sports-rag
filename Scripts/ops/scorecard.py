"""Write the Spix scorecard (rework A1 step 5) for review.

Reads published picks from the platform database (the measured cohort: picks
published after the measurement activation boundary) and writes
``Research/scorecards/<as-of>/scorecard.json`` and ``scorecard.md``.

    python -m Scripts.ops.scorecard [--as-of 2026-10-19] [--resamples 2000]
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
    args = parser.parse_args(argv)
    from Scripts.data_platform.scorecard import build, markdown
    rows, boundary = load_rows()
    card = build(rows, as_of=args.as_of, resamples=args.resamples, after_recommendation_id=boundary)
    folder = args.out / args.as_of
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "scorecard.json").write_text(json.dumps(card, indent=2, default=str))
    (folder / "scorecard.md").write_text(markdown(card))
    print(markdown(card))
    print(f"\nwritten to {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
