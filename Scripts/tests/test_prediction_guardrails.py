"""Unit tests for conservative canonical recommendation guardrails."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from core.prediction_guardrails import assess_market_quality, resolve_total_variance


def _quality(*, current_matches: int, effective_sample_size: float = 10.0) -> dict:
    profile = {
        "temporal_status": "fixture_rows_strictly_before_target_date",
        "current_season_matches": current_matches,
        "effective_sample_size": effective_sample_size,
    }
    return {
        "fixture_date": "2026-08-26T19:30:00Z",
        "profiles": {"status": "available", "home": dict(profile), "away": dict(profile)},
    }


class PredictionGuardrailTests(unittest.TestCase):
    def test_missing_variance_uses_an_explicit_overdispersed_floor(self):
        variance, detail = resolve_total_variance("cards", 5.0, None, None)

        self.assertEqual(detail["source"], "conservative_fallback_floor")
        self.assertGreater(variance, 5.0)
        self.assertEqual(detail["variance"], variance)

    def test_one_current_match_suppresses_a_public_recommendation(self):
        assessment = assess_market_quality(_quality(current_matches=1), "goals")

        self.assertFalse(assessment["eligible"])
        self.assertEqual(assessment["status"], "suppressed")
        self.assertIn("Fewer than", " ".join(assessment["reasons"]))

    def test_early_but_eligible_profile_is_confidence_capped(self):
        assessment = assess_market_quality(_quality(current_matches=3), "corners")

        self.assertTrue(assessment["eligible"])
        self.assertEqual(assessment["status"], "capped")
        self.assertEqual(assessment["confidence_cap"], "low")

    def test_cards_require_a_profiled_referee(self):
        assessment = assess_market_quality(
            _quality(current_matches=10), "cards", referee_source="unavailable",
        )

        self.assertFalse(assessment["eligible"])
        self.assertIn("profiled referee", " ".join(assessment["reasons"]))


if __name__ == "__main__":
    unittest.main()
