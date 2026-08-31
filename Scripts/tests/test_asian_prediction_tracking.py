"""Asian total settlement must stay consistent from selection through tracking."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from prediction_tracker import _compute_roi, _grade_prediction


class AsianPredictionTrackingTests(unittest.TestCase):
    def test_quarter_line_outcomes_are_recorded_as_partial_settlements(self):
        result = {"goals_home": 2, "goals_away": 1}

        half_win, actual = _grade_prediction(
            {"market": "goals", "side": "over", "line": 2.75}, result,
        )
        half_loss, _ = _grade_prediction(
            {"market": "goals", "side": "over", "line": 3.25}, result,
        )

        self.assertEqual(actual, 3)
        self.assertEqual(half_win, "half_hit")
        self.assertEqual(half_loss, "half_miss")

    def test_flat_stake_roi_handles_half_win_and_half_loss(self):
        roi = _compute_roi([
            {"odds": 2.0, "outcome": "half_hit"},
            {"odds": 2.0, "outcome": "half_miss"},
        ])

        # Returns are 1.5 and 0.5 from two one-unit stakes: breakeven.
        self.assertAlmostEqual(roi, 0.0)


if __name__ == "__main__":
    unittest.main()
