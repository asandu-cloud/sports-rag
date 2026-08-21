"""Tests that rag_cli_v2 projection wrappers call core and produce identical output."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

import rag_cli_v2 as rag
from core import line_selection as core_selection
from core import projections as core_proj


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

    def test_per_team_primitives_call_core(self):
        home_meta = {"corners_pm": 5.0}
        away_meta = {"corners_pm": 4.0}
        with mock.patch.object(core_proj, "projected_corners", return_value=(5.1, 4.2)) as corners, \
             mock.patch.object(core_proj, "projected_cards", return_value=(1.8, 2.0)) as cards, \
             mock.patch.object(core_proj, "projected_sot", return_value=(4.4, 3.7)) as sot, \
             mock.patch.object(core_proj, "projected_goals", return_value=(1.7, 1.1)) as goals:
            self.assertEqual(rag.projected_corners(home_meta, away_meta), (5.1, 4.2))
            self.assertEqual(rag.projected_cards(home_meta, away_meta, referee_modifier=1.05), (1.8, 2.0))
            self.assertEqual(rag.projected_sot(home_meta, away_meta), (4.4, 3.7))
            self.assertEqual(rag.projected_goals(home_meta, away_meta), (1.7, 1.1))

        corners.assert_called_once_with(home_meta, away_meta)
        cards.assert_called_once_with(home_meta, away_meta, referee_modifier=1.05)
        sot.assert_called_once_with(home_meta, away_meta)
        goals.assert_called_once_with(home_meta, away_meta)

    def test_legacy_selectors_call_core(self):
        options = [{"side": "over", "point": 2.5, "odds": 2.0}]
        score_probs = {(1, 0): 0.2}
        score_odds = {(1, 0): [{"odds": 5.0}]}
        moneyline = [{"side": "home", "odds": 2.0}]
        with mock.patch.object(core_selection, "choose_best_total_line", return_value={"side": "over"}) as totals, \
             mock.patch.object(core_selection, "choose_best_btts_side", return_value={"side": "yes"}) as btts, \
             mock.patch.object(core_selection, "choose_best_correct_scores", return_value=[{"score": (1, 0)}]) as correct, \
             mock.patch.object(core_selection, "choose_best_moneyline_side", return_value={"bet_recommendation": None}) as ml, \
             mock.patch.object(core_selection, "choose_best_spread_line", return_value={"team": "Home"}) as spreads:
            self.assertEqual(rag.choose_best_total_line(options, 3.0, 1.2), {"side": "over"})
            self.assertEqual(rag.choose_best_btts_side(options, 0.62), {"side": "yes"})
            self.assertEqual(rag.choose_best_correct_scores(score_probs, score_odds, top_n=3), [{"score": (1, 0)}])
            self.assertEqual(rag.choose_best_moneyline_side(0.5, 0.25, 0.25, moneyline, "Home", "Away"), {"bet_recommendation": None})
            self.assertEqual(rag.choose_best_spread_line(options, 0.4, "Home"), {"team": "Home"})

        totals.assert_called_once_with(options, 3.0, 1.2)
        btts.assert_called_once_with(options, 0.62)
        correct.assert_called_once_with(score_probs, score_odds, top_n=3)
        ml.assert_called_once_with(0.5, 0.25, 0.25, moneyline, "Home", "Away")
        spreads.assert_called_once_with(options, 0.4, "Home")

    def test_btts_calls_core(self):
        with mock.patch.object(core_proj, "projected_btts_prob", return_value=("ok", 1, 2, 3, 4)) as patched:
            result = rag.projected_btts_prob("Home", "Away", "EPL", fixture_date="2026-04-10")
        patched.assert_called_once_with("Home", "Away", "EPL", league_ctx=None, fixture_date="2026-04-10")
        self.assertEqual(result, ("ok", 1, 2, 3, 4))

    def test_moneyline_calls_core(self):
        with mock.patch.object(core_proj, "projected_moneyline_probs", return_value=(0.5, 0.2, 0.3, 1.4, 0.9)) as patched:
            result = rag.projected_moneyline_probs("Home", "Away", "EPL", fixture_date="2026-04-10")
        patched.assert_called_once_with("Home", "Away", "EPL", league_ctx=None, fixture_date="2026-04-10")
        self.assertEqual(result, (0.5, 0.2, 0.3, 1.4, 0.9))

    def test_goal_difference_calls_core(self):
        with mock.patch.object(core_proj, "projected_goal_difference", return_value=(0.4, 0.2, 0.1)) as patched:
            result = rag.projected_goal_difference("Home", "Away", "EPL", fixture_date="2026-04-10")
        patched.assert_called_once_with("Home", "Away", "EPL", league_ctx=None, fixture_date="2026-04-10")
        self.assertEqual(result, (0.4, 0.2, 0.1))

    def test_total_goals_calls_core_wrapper(self):
        with mock.patch.object(core_proj, "projected_total_goals", return_value=(2.8, 2.6, 3.0)) as patched:
            result = rag.projected_total_goals("Home", "Away", "EPL", fixture_date="2026-04-10")
        patched.assert_called_once_with(
            "Home", "Away", "EPL",
            knockout_ctx=None,
            league_ctx=None,
            fixture_date="2026-04-10",
        )
        self.assertEqual(result, (2.8, 2.6, 3.0))

    def test_total_corners_calls_core_wrapper(self):
        with mock.patch.object(core_proj, "projected_total_corners", return_value=(10.1, 9.8, 10.4)) as patched:
            result = rag.projected_total_corners("Home", "Away", "EPL", fixture_date="2026-04-10")
        patched.assert_called_once_with(
            "Home", "Away", "EPL",
            knockout_ctx=None,
            league_ctx=None,
            fixture_date="2026-04-10",
        )
        self.assertEqual(result, (10.1, 9.8, 10.4))

    def test_total_sot_calls_core_wrapper(self):
        with mock.patch.object(core_proj, "projected_total_sot", return_value=(8.1, 7.8, 8.4)) as patched:
            result = rag.projected_total_sot("Home", "Away", "EPL", fixture_date="2026-04-10")
        patched.assert_called_once_with(
            "Home", "Away", "EPL",
            knockout_ctx=None,
            league_ctx=None,
            fixture_date="2026-04-10",
        )
        self.assertEqual(result, (8.1, 7.8, 8.4))

    def test_team_corners_calls_core_with_fixture_context(self):
        with mock.patch.object(core_proj, "projected_team_corners", return_value=(5.1, 5.0, 5.2)) as patched:
            result = rag.projected_team_corners("Home", "Away", "EPL", True, fixture_date="2026-04-10")
        patched.assert_called_once_with(
            "Home", "Away", "EPL", True,
            league_ctx=None,
            fixture_date="2026-04-10",
        )
        self.assertEqual(result, (5.1, 5.0, 5.2))

    def test_team_cards_calls_core_with_fixture_context(self):
        ref_mod = object()
        with mock.patch.object(core_proj, "projected_team_cards", return_value=(1.8, 1.7, 1.9)) as patched:
            result = rag.projected_team_cards(
                "Home", "Away", "EPL", True,
                ref_mod=ref_mod,
                fixture_date="2026-04-10",
            )
        patched.assert_called_once_with(
            "Home", "Away", "EPL", True,
            ref_mod=ref_mod,
            lineup_ctx=None,
            league_ctx=None,
            fixture_date="2026-04-10",
        )
        self.assertEqual(result, (1.8, 1.7, 1.9))

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
