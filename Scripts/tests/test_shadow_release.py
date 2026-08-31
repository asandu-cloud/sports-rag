"""Tests for durable, non-publishing shadow capture."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

from Scripts.rag_ingest.core.market_result import (
    Decision,
    DecisionStatus,
    FixtureRef,
    MarketRef,
    MarketResult,
    PriceQuote,
    Projection,
    Provenance,
)
from Scripts.rag_ingest.core.release_control import apply_release_policy, resolve_release_policy
from Scripts.rag_ingest.shadow_release import capture_shadow_run


def _shadow_result() -> MarketResult:
    candidate = Decision(
        status=DecisionStatus.RECOMMENDED,
        quote=PriceQuote(side="over", line=2.5, odds=2.0, bookmaker="Book"),
        model_probability=0.62,
        implied_probability=0.50,
        value_edge=0.12,
        expected_value=0.24,
        confidence="medium",
    )
    context = {}
    withheld = apply_release_policy(candidate, context, policy=resolve_release_policy("shadow"))
    return MarketResult(
        fixture=FixtureRef("fixture-1", "EPL", "Home", "Away", "2026-09-01T15:00:00Z"),
        market=MarketRef("totals", "goals", "goals"),
        projection=Projection(3.1, "goals"),
        decision=withheld,
        provenance=Provenance("test", "snapshot-1", "2026-09-01T09:00:00Z"),
        context=context,
    )


class ShadowReleaseTests(unittest.TestCase):
    def test_capture_is_non_publishing_and_idempotently_archives_candidates(self):
        observed_modes = []

        def evaluator(_league, _market, _date):
            observed_modes.append(resolve_release_policy().mode)
            return [_shadow_result()], ["fixture checked"]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = root / "shadow.jsonl"
            db_path = root / "predictions.db"
            first = capture_shadow_run(
                leagues=["EPL"], markets=["totals"], target_date=date(2026, 9, 1),
                output_path=archive, db_path=db_path, evaluator=evaluator,
            )
            second = capture_shadow_run(
                leagues=["EPL"], markets=["totals"], target_date=date(2026, 9, 1),
                output_path=archive, db_path=db_path, evaluator=evaluator,
            )
            rows = [json.loads(line) for line in archive.read_text(encoding="utf-8").splitlines()]
            conn = sqlite3.connect(str(db_path))
            try:
                tracker_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            finally:
                conn.close()

        self.assertEqual(observed_modes, ["shadow", "shadow"])
        self.assertEqual(first["snapshots_written"], 1)
        self.assertEqual(second["snapshots_written"], 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["candidate"]["status"], "recommended")
        self.assertEqual(rows[0]["result"]["decision"]["status"], "no_bet")
        self.assertEqual(tracker_count, 1)


if __name__ == "__main__":
    unittest.main()
