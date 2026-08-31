"""Tests for chronological recorded-decision analysis and release gates."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from walk_forward import RecordedDecision, build_walk_forward_report, release_readiness


def _record(
    identifier: str,
    probability: float,
    outcome: str,
    *,
    generated_at: str = "2026-09-01T09:00:00Z",
    kickoff: str = "2026-09-01T15:00:00Z",
) -> RecordedDecision:
    return RecordedDecision(
        record_id=identifier,
        source="canonical_shadow",
        league="EPL",
        market="goals",
        model_probability=probability,
        odds=2.0,
        outcome=outcome,
        generated_at=generated_at,
        kickoff=kickoff,
        prediction_date="2026-09-01",
    )


class WalkForwardTests(unittest.TestCase):
    def test_asian_half_outcomes_contribute_fractional_roi_and_calibration_weight(self):
        report = build_walk_forward_report([
            _record("one", 0.70, "hit"),
            _record("two", 0.60, "half_miss"),
            _record("three", 0.50, "push"),
        ], bucket_width=0.10)
        overall = report["overall"]

        self.assertEqual(overall["graded_records"], 3)
        self.assertAlmostEqual(overall["stake_units"], 3.0)
        self.assertAlmostEqual(overall["profit_units"], 0.5)
        self.assertAlmostEqual(overall["roi"], 1.0 / 6.0)
        self.assertAlmostEqual(overall["resolved_stake_units"], 1.5)
        self.assertAlmostEqual(overall["win_stake_units"], 1.0)
        self.assertAlmostEqual(overall["hit_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(overall["brier_score"], 0.18)

    def test_release_gate_holds_a_small_or_temporally_invalid_sample(self):
        report = build_walk_forward_report([
            _record(
                "late",
                0.70,
                "hit",
                generated_at="2026-09-01T16:00:00Z",
                kickoff="2026-09-01T15:00:00Z",
            )
        ])
        readiness = release_readiness(
            report,
            minimum_resolved_stake=1.0,
            minimum_bucket_sample=1.0,
            maximum_calibration_gap=0.50,
        )

        self.assertFalse(readiness["eligible"])
        self.assertEqual(len(report["pre_match_violations"]), 1)
        self.assertIn("after kick-off", " ".join(readiness["reasons"]))


if __name__ == "__main__":
    unittest.main()
