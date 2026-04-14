from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from core.line_selection import (
    select_best_btts_recommendation,
    select_best_total_recommendation,
)


class MarketSelectionTests(unittest.TestCase):
    def test_totals_returns_no_bet_when_conviction_is_thin(self):
        options = [
            {"side": "over", "point": 2.5, "odds": 1.91, "bookmaker": "Book", "market_key": "totals"},
            {"side": "under", "point": 2.5, "odds": 1.91, "bookmaker": "Book", "market_key": "totals"},
        ]
        result = select_best_total_recommendation(options, projection_total=2.52, combined_var=None)
        self.assertIsNone(result["bet_recommendation"])
        self.assertIsNotNone(result["no_bet_reason"])
        self.assertIsNotNone(result["best_value"])

    def test_totals_recommends_when_model_and_price_align(self):
        options = [
            {"side": "over", "point": 2.5, "odds": 2.05, "bookmaker": "Book", "market_key": "totals"},
            {"side": "under", "point": 2.5, "odds": 1.80, "bookmaker": "Book", "market_key": "totals"},
        ]
        result = select_best_total_recommendation(options, projection_total=3.2, combined_var=None)
        self.assertIsNotNone(result["bet_recommendation"])
        self.assertEqual(result["bet_recommendation"]["side"], "over")

    def test_btts_rejects_opposite_side_juiced_price(self):
        options = [
            {"side": "yes", "odds": 1.42, "bookmaker": "Book", "market_key": "btts"},
            {"side": "no", "odds": 3.40, "bookmaker": "Book", "market_key": "btts"},
        ]
        result = select_best_btts_recommendation(options, p_btts_yes=0.64)
        self.assertIsNone(result["bet_recommendation"])
        self.assertIsNotNone(result["no_bet_reason"])

    def test_btts_recommends_model_preferred_side_when_price_is_good(self):
        options = [
            {"side": "yes", "odds": 1.95, "bookmaker": "Book", "market_key": "btts"},
            {"side": "no", "odds": 1.95, "bookmaker": "Book", "market_key": "btts"},
        ]
        result = select_best_btts_recommendation(options, p_btts_yes=0.64)
        self.assertIsNotNone(result["bet_recommendation"])
        self.assertEqual(result["bet_recommendation"]["side"], "yes")


if __name__ == "__main__":
    unittest.main()
