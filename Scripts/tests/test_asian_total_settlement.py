"""Tests for Asian total settlement and its selector integration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from core.line_selection import select_best_total_recommendation
from prob_models import (
    asian_total_equivalent_probability,
    asian_total_expected_value,
    asian_total_settlement_profile,
    poisson_cdf,
    poisson_pmf,
)


class AsianTotalSettlementTests(unittest.TestCase):
    def test_over_quarter_line_treats_exactly_three_as_a_half_win(self):
        mean = 3.0
        profile = asian_total_settlement_profile(mean, 2.75, "over")

        self.assertAlmostEqual(profile["half_win"], poisson_pmf(3, mean), places=8)
        self.assertAlmostEqual(profile["full_win"], 1.0 - poisson_cdf(3, mean), places=8)
        self.assertAlmostEqual(profile["full_loss"], poisson_cdf(2, mean), places=8)
        self.assertAlmostEqual(sum(profile.values()), 1.0, places=8)

    def test_whole_line_push_is_excluded_from_equivalent_probability_and_ev(self):
        mean = 3.0
        odds = 1.95
        profile = asian_total_settlement_profile(mean, 3.0, "over")
        win = 1.0 - poisson_cdf(3, mean)
        push = poisson_pmf(3, mean)
        loss = poisson_cdf(2, mean)

        self.assertAlmostEqual(profile["full_win"], win, places=8)
        self.assertAlmostEqual(profile["push"], push, places=8)
        self.assertAlmostEqual(profile["full_loss"], loss, places=8)
        self.assertAlmostEqual(asian_total_equivalent_probability(profile), win / (win + loss), places=8)
        self.assertAlmostEqual(asian_total_expected_value(profile, odds), win * (odds - 1.0) - loss, places=8)

    def test_selector_does_not_turn_a_quarter_line_half_win_into_a_full_win(self):
        options = [
            {"side": "over", "point": 2.75, "odds": 1.95, "bookmaker": "Book", "market_key": "totals"},
            {"side": "under", "point": 2.75, "odds": 1.95, "bookmaker": "Book", "market_key": "totals"},
        ]

        result = select_best_total_recommendation(options, projection_total=3.0, combined_var=None)

        self.assertIsNone(result["bet_recommendation"])
        over = next(line for line in result["all_lines"] if line["side"] == "over")
        self.assertEqual(over["_probability_basis"], "asian_equivalent_non_push")
        self.assertGreater(over["_settlement_profile"]["half_win"], 0.20)
        self.assertLess(over["_ev"], 0.02)

    def test_selector_requires_a_paired_price_for_a_vig_adjusted_recommendation(self):
        result = select_best_total_recommendation(
            [{"side": "over", "point": 2.5, "odds": 2.40, "bookmaker": "Book", "market_key": "totals"}],
            projection_total=4.0,
            combined_var=5.0,
        )

        self.assertIsNone(result["bet_recommendation"])
        self.assertIn("paired opposite price", result["no_bet_reason"])


if __name__ == "__main__":
    unittest.main()
