"""Tests for league_context.py — rest days, regime shift, bounded adjustments."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from Scripts.rag_ingest.league_context import (
    LeagueContext,
    get_league_context,
    _compute_rest_days,
    _compute_regime_shift,
    league_context_evidence_line,
)


class LeagueContextBasicTests(unittest.TestCase):
    def test_default_context_neutral(self):
        """Default LeagueContext should have neutral (1.0) adjustments."""
        ctx = LeagueContext()
        self.assertEqual(ctx.adjustments["goals"], 1.0)
        self.assertEqual(ctx.adjustments["corners"], 1.0)
        self.assertFalse(ctx.home_short_rest)

    def test_adjustments_bounded(self):
        """Adjustments must stay within [0.85, 1.15] bounds."""
        recent = {"n": 6, "corners_for_avg": 8.0, "cards_avg": 3.0,
                  "sot_for_avg": 5.0, "xg_for_avg": 2.0,
                  "control_avg": 0.60, "possession_avg": 0.55}
        profile = {"corners_pm": 5.0, "cards_per_90_team": 2.0,
                   "sot_for_pm": 4.0, "goals_for_pm": 1.5,
                   "control_index": 0.55, "possession": 0.52}

        with mock.patch("Scripts.rag_ingest.league_context._recent_stats", return_value=recent), \
             mock.patch("Scripts.rag_ingest.league_context._profile_meta", return_value=profile), \
             mock.patch("Scripts.rag_ingest.league_context.get_recent_team_fixture_rows", return_value=[]):

            ctx = get_league_context("Home", "Away", "EPL")

        for stat in ["goals", "corners", "cards", "sot"]:
            self.assertGreaterEqual(ctx.adjustments[stat], 0.85)
            self.assertLessEqual(ctx.adjustments[stat], 1.15)

    def test_evidence_line_generation(self):
        ctx = LeagueContext()
        ctx.evidence = ["Arsenal: short rest (2d)", "cards regime: up 20%"]
        ctx.source = "computed"

        line = league_context_evidence_line(ctx)
        self.assertIsNotNone(line)
        self.assertIn("Arsenal", line)

    def test_no_evidence_returns_none(self):
        ctx = LeagueContext()
        line = league_context_evidence_line(ctx)
        self.assertIsNone(line)


class RestDaysTests(unittest.TestCase):
    def test_compute_rest_no_rows(self):
        with mock.patch("Scripts.rag_ingest.league_context.get_recent_team_fixture_rows",
                         return_value=[]):
            rest, cong = _compute_rest_days("Team", "EPL")
        self.assertIsNone(rest)
        self.assertEqual(cong, 0)


class RegimeShiftTests(unittest.TestCase):
    def test_no_data_returns_empty(self):
        with mock.patch("Scripts.rag_ingest.league_context._recent_stats", return_value={}), \
             mock.patch("Scripts.rag_ingest.league_context._profile_meta", return_value={}):
            shift = _compute_regime_shift("Team", "EPL")
        self.assertEqual(shift, {})

    def test_positive_shift_detected(self):
        recent = {"corners_for_avg": 7.0, "cards_avg": 3.0,
                  "sot_for_avg": 5.0, "xg_for_avg": 2.0,
                  "control_avg": 0.60, "possession_avg": 0.55}
        profile = {"corners_pm": 5.0, "cards_per_90_team": 2.0,
                   "sot_for_pm": 4.0, "goals_for_pm": 1.5,
                   "control_index": 0.55, "possession": 0.52}

        with mock.patch("Scripts.rag_ingest.league_context._recent_stats", return_value=recent), \
             mock.patch("Scripts.rag_ingest.league_context._profile_meta", return_value=profile):
            shift = _compute_regime_shift("Team", "EPL")

        self.assertGreater(shift.get("corners", 0), 0)
        self.assertGreater(shift.get("goals", 0), 0)


if __name__ == "__main__":
    unittest.main()
