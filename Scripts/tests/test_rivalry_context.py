"""Tests for rivalry_context.py derby/rivalry detection."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from Scripts.rag_ingest import league_context
from Scripts.rag_ingest.rendering import comparison
from Scripts.rag_ingest.league_context import get_league_context, league_context_evidence_line
from Scripts.rag_ingest.rivalry_context import get_rivalry_context, rivalry_context_evidence_line


class RivalryContextTests(unittest.TestCase):
    def setUp(self):
        league_context._league_context_cache.clear()

    def test_detects_el_clasico(self):
        ctx = get_rivalry_context("Real Madrid", "Barcelona", "LaLiga")

        self.assertEqual(ctx.source, "curated")
        self.assertEqual(ctx.label, "El Clasico")
        self.assertGreater(ctx.cards_modifier, 1.0)
        self.assertGreater(ctx.corners_modifier, 1.0)
        self.assertEqual(ctx.goals_modifier, 1.0)
        self.assertIn("El Clasico", rivalry_context_evidence_line(ctx))

    def test_detection_is_direction_insensitive_and_alias_aware(self):
        ctx = get_rivalry_context("Barca", "Real Madrid", "LaLiga")
        self.assertEqual(ctx.source, "curated")
        self.assertEqual(ctx.label, "El Clasico")

    def test_non_rivalry_returns_neutral_context(self):
        ctx = get_rivalry_context("Arsenal", "Brighton", "EPL")
        self.assertEqual(ctx.source, "none")
        self.assertEqual(ctx.cards_modifier, 1.0)
        self.assertIsNone(rivalry_context_evidence_line(ctx))

    def test_league_context_applies_rivalry_modifier(self):
        with mock.patch("Scripts.rag_ingest.league_context._compute_rest_days", return_value=(None, 0)), \
             mock.patch("Scripts.rag_ingest.league_context._compute_regime_shift", return_value={}), \
             mock.patch("Scripts.rag_ingest.league_context.get_stake_context", None):
            ctx = get_league_context("Real Madrid", "Barcelona", "LaLiga")

        self.assertEqual(ctx.rivalry_label, "El Clasico")
        self.assertGreater(ctx.adjustments["cards"], 1.0)
        self.assertGreater(ctx.adjustments["corners"], 1.0)
        self.assertEqual(ctx.adjustments["goals"], 1.0)
        self.assertIn("Rivalry:", league_context_evidence_line(ctx))

    def test_league_context_evidence_prioritizes_rivalry(self):
        ctx = league_context.LeagueContext()
        ctx.source = "computed"
        ctx.evidence = [
            "Home: short rest (2d)",
            "Away: short rest (2d)",
            "Home: 4 fixtures in 14d",
            "Away: 4 fixtures in 14d",
            "Rivalry: El Clasico detected.",
        ]

        line = league_context_evidence_line(ctx)
        self.assertIn("Rivalry: El Clasico", line)
        self.assertNotIn("Away: 4 fixtures", line)

    def test_comparison_render_mentions_detected_rivalry(self):
        events = [{"home_team": "Real Madrid", "away_team": "Barcelona"}]
        with mock.patch(
            "Scripts.rag_ingest.rendering.comparison.comparison_signals_cards",
            return_value=("Real Madrid", ["Profile projection: Real Madrid 2.4 vs Barcelona 2.1 cards."], 2.4, 2.1),
        ):
            out = comparison.render_comparison_answer(
                "which team gets more cards in LaLiga?",
                "LaLiga",
                events,
            )

        self.assertIn("Rivalry: El Clasico", out)


if __name__ == "__main__":
    unittest.main()
