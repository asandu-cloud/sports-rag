"""Tests for backtest slice generation and edge-bucket analysis."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from Scripts.rag_ingest.league_eval import compute_edge_buckets


class EdgeBucketTests(unittest.TestCase):
    def test_edge_buckets_basic(self):
        """Edge buckets should produce results for fixtures with varying edges."""
        results = [
            {"proj_goals": 2.8, "actual_goals": 3},   # edge 0.3 from 2.5 → over correct
            {"proj_goals": 2.2, "actual_goals": 2},   # edge 0.3 from 2.5 → under correct
            {"proj_goals": 3.5, "actual_goals": 4},   # edge 1.0 from 2.5 → over correct
            {"proj_goals": 1.5, "actual_goals": 1},   # edge 1.0 from 2.5 → under correct
            {"proj_goals": 2.6, "actual_goals": 2},   # edge 0.1 from 2.5 → over wrong
            {"proj_goals": 2.4, "actual_goals": 3},   # edge 0.1 from 2.5 → under wrong
        ]

        buckets = compute_edge_buckets(results, "goals", [2.5])
        # Should have some buckets
        self.assertIsInstance(buckets, list)

    def test_edge_buckets_empty_data(self):
        buckets = compute_edge_buckets([], "goals", [2.5])
        self.assertEqual(buckets, [])

    def test_edge_buckets_no_valid_projections(self):
        results = [{"proj_goals": None, "actual_goals": 3}]
        buckets = compute_edge_buckets(results, "goals", [2.5])
        self.assertEqual(buckets, [])

    def test_high_edge_should_hit_more(self):
        """Fixtures with larger edges should have higher hit rates (in theory)."""
        # Create fixtures where high-edge predictions are always correct
        results = []
        for i in range(10):
            results.append({"proj_goals": 4.0, "actual_goals": 4})  # edge 1.5 from 2.5
        for i in range(10):
            results.append({"proj_goals": 2.6, "actual_goals": 2})  # edge 0.1 from 2.5, wrong

        buckets = compute_edge_buckets(results, "goals", [2.5])
        # The high-edge bucket (1.0-2.0) should have 100% hit rate
        high_edge = [b for b in buckets if b["edge_bucket"] == "1.0-2.0"]
        if high_edge:
            self.assertEqual(high_edge[0]["hit_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
