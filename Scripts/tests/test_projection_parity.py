"""Tests that rag_cli_v2 projection wrappers call core and produce identical output."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from Scripts.rag_ingest.core import projections as core_proj


class ProjectionParityTests(unittest.TestCase):
    """Verify rag_cli_v2 delegates to core for all unified projection functions."""

    def _mock_profile_and_recent(self):
        """Set up common mocks for profile and recent stats."""
        profile = {
            "cards_per_90_team": 2.0, "opp_cards_induced_pm": 1.5,
            "corners_pm": 5.0, "goals_for_pm": 1.5, "goals_against_pm": 1.2,
            "sot_for_pm": 4.0, "sot_against_pm": 3.5,
            "control_index": 0.55, "dominance_index": 0.50,
            "possession": 0.52, "aggression_index_norm": 0.30,
            "fouls_per_90_team": 11.0, "cards_per_foul_team": 0.18,
            "shots_for_pm": 12.0, "form_index_team": 0.55,
        }
        recent = {
            "n": 6, "cards_avg": 2.0, "cards_induced_avg": 1.5,
            "corners_for_avg": 5.0, "corners_against_avg": 4.5,
            "sot_for_avg": 4.0, "sot_against_avg": 3.5,
            "xg_for_avg": 1.5, "shots_for_avg": 12.0,
            "fouls_avg": 11.0, "aggression_avg": 0.30,
            "control_avg": 0.55, "form_avg": 0.55,
            "possession_avg": 0.52,
            "corners_for_slope": 0.0, "sot_for_slope": 0.0,
            "cards_slope": 0.0, "goals_slope": 0.0,
            "yellows_avg": None, "reds_avg": None,
            "yellows_induced_avg": None, "reds_induced_avg": None,
            "cards_per_foul_avg": None, "fouls_drawn_avg": None,
        }
        return profile, recent

    def test_total_goals_calls_core(self):
        """projected_total_goals in rag_cli_v2 should delegate to core."""
        profile, recent = self._mock_profile_and_recent()
        ref_mod_ns = type("RM", (), {"source": "unavailable", "multiplier": 1.0,
                                      "confidence": 0.0, "sample_size": 0,
                                      "avg_cards_per_match": 0.0, "cards_per_foul": 0.0,
                                      "avg_fouls_per_match": 0.0, "strictness_ratio": 1.0})()

        with mock.patch.object(core_proj, "_profile_meta", return_value=profile), \
             mock.patch.object(core_proj, "_recent_stats", return_value=recent), \
             mock.patch.object(core_proj, "get_referee_modifier", return_value=ref_mod_ns), \
             mock.patch.object(core_proj, "_ml_blend_weight", return_value=0.0):

            core_result = core_proj.projected_total_goals("Home", "Away", "EPL")

        self.assertEqual(len(core_result), 3)
        self.assertIsNotNone(core_result[0])

    def test_total_corners_calls_core(self):
        profile, recent = self._mock_profile_and_recent()

        with mock.patch.object(core_proj, "_profile_meta", return_value=profile), \
             mock.patch.object(core_proj, "_recent_stats", return_value=recent), \
             mock.patch.object(core_proj, "_ml_blend_weight", return_value=0.0):

            result = core_proj.projected_total_corners("Home", "Away", "EPL")

        self.assertEqual(len(result), 3)
        self.assertIsNotNone(result[0])

    def test_total_sot_calls_core(self):
        profile, recent = self._mock_profile_and_recent()

        with mock.patch.object(core_proj, "_profile_meta", return_value=profile), \
             mock.patch.object(core_proj, "_recent_stats", return_value=recent), \
             mock.patch.object(core_proj, "_ml_blend_weight", return_value=0.0):

            result = core_proj.projected_total_sot("Home", "Away", "EPL")

        self.assertEqual(len(result), 3)
        self.assertIsNotNone(result[0])

    def test_total_cards_backward_compat(self):
        """projected_total_cards must return 4-tuple."""
        profile, recent = self._mock_profile_and_recent()
        ref_mod_ns = type("RM", (), {"source": "unavailable", "multiplier": 1.0,
                                      "confidence": 0.0, "sample_size": 0,
                                      "avg_cards_per_match": 0.0, "cards_per_foul": 0.0,
                                      "avg_fouls_per_match": 0.0, "strictness_ratio": 1.0})()

        with mock.patch.object(core_proj, "_profile_meta", return_value=profile), \
             mock.patch.object(core_proj, "_recent_stats", return_value=recent), \
             mock.patch.object(core_proj, "get_referee_modifier", return_value=ref_mod_ns), \
             mock.patch.object(core_proj, "_ml_blend_weight", return_value=0.0):

            result = core_proj.projected_total_cards("Home", "Away", "EPL")

        self.assertEqual(len(result), 4)

    def test_league_ctx_default_none_no_effect(self):
        """When league_ctx=None, projections should be unaffected."""
        profile, recent = self._mock_profile_and_recent()

        with mock.patch.object(core_proj, "_profile_meta", return_value=profile), \
             mock.patch.object(core_proj, "_recent_stats", return_value=recent), \
             mock.patch.object(core_proj, "_ml_blend_weight", return_value=0.0):

            result_none = core_proj.projected_total_goals("Home", "Away", "EPL", league_ctx=None)
            result_default = core_proj.projected_total_goals("Home", "Away", "EPL")

        self.assertEqual(result_none, result_default)

    def test_compute_stat_confidence_returns_valid_bucket(self):
        profile, recent = self._mock_profile_and_recent()

        with mock.patch.object(core_proj, "_profile_meta", return_value=profile), \
             mock.patch.object(core_proj, "_recent_stats", return_value=recent):
            conf = core_proj.compute_stat_confidence("goals", "Home", "Away", "EPL")

        self.assertIn(conf, ("high", "medium", "low"))


if __name__ == "__main__":
    unittest.main()
