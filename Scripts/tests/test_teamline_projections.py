from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from Scripts.rag_ingest.core import projections
from Scripts.rag_ingest.core import team_resolution as resolver


class TeamlineResolutionTests(unittest.TestCase):
    def tearDown(self):
        resolver._team_recent_stats_cache.clear()
        resolver._team_fixture_meta_cache.clear()

    def test_recent_stats_include_cards_induced_average(self):
        rows = [
            {"meta": {"opponent": "Chelsea", "fixture": "Arsenal vs Chelsea", "fixture_date": "2026-03-01", "season": "2025/26", "cards_per_90_team": 2.0}},
            {"meta": {"opponent": "Liverpool", "fixture": "Liverpool vs Arsenal", "fixture_date": "2026-02-22", "season": "2025/26", "cards_per_90_team": 1.0}},
        ]
        # _get_team_fixture_meta is called for each opponent in 3 loops:
        # cards_induced (2 calls), yellows/reds_induced (2 calls), fouls_drawn (2 calls) = 6 total
        opp_meta_chelsea = {"cards_per_90_team": 3.0, "yellow_cards": 2.0, "red_cards": 0.0, "fouls_per_90_team": 12.0}
        opp_meta_liverpool = {"cards_per_90_team": 1.0, "yellow_cards": 1.0, "red_cards": 0.0, "fouls_per_90_team": 10.0}
        with mock.patch.object(resolver, "get_recent_team_fixture_rows", return_value=rows), \
             mock.patch.object(
                 resolver,
                 "_get_team_fixture_meta",
                 side_effect=[
                     opp_meta_chelsea, opp_meta_liverpool,  # cards_induced loop
                     opp_meta_chelsea, opp_meta_liverpool,  # fouls_drawn loop
                 ],
             ):
            stats = resolver.get_team_recent_stats("Arsenal", "EPL", last_n=2)

        expected = (3.0 + 0.85 * 1.0) / (1.0 + 0.85)
        self.assertAlmostEqual(stats["cards_induced_avg"], expected, places=4)


class TeamlineProjectionTests(unittest.TestCase):
    def test_team_corners_anchored_to_match_total(self):
        """Team corners should be anchored to match total via share + stabilizer."""
        arsenal_meta = {
            "corners_pm": 5.5, "corners_home_pm": 6.0,
            "corners_against_pm": 4.5, "corners_against_home_pm": 4.2,
            "dominance_index": 0.45, "control_index": 0.68,
            "possession": 0.60, "shots_for_pm": 14.0, "sot_for_pm": 4.5,
        }
        chelsea_meta = {
            "corners_pm": 4.5, "corners_away_pm": 4.0,
            "corners_against_pm": 5.0, "corners_against_away_pm": 5.2,
            "dominance_index": 0.40, "control_index": 0.55,
            "possession": 0.48, "shots_for_pm": 11.0, "sot_for_pm": 3.8,
            "sot_against_away_pm": 4.2,
        }

        def profile_side_effect(team, league):
            return arsenal_meta if team == "Arsenal" else chelsea_meta

        def recent_side_effect(team, league, last_n=6, venue=None):
            if team == "Arsenal":
                return {
                    "corners_for_avg": 6.0, "corners_against_avg": 4.5,
                    "shots_for_avg": 16.0, "sot_for_avg": 5.0,
                    "control_avg": 0.70, "possession_avg": 0.62,
                    "corners_for_slope": 0.0,
                }
            if team == "Chelsea":
                return {
                    "corners_for_avg": 4.2, "corners_against_avg": 5.0,
                    "shots_for_avg": 12.0, "sot_for_avg": 3.5,
                    "control_avg": 0.52, "possession_avg": 0.47,
                    "corners_for_slope": 0.0,
                }
            return {}

        with mock.patch.object(projections, "_profile_meta", side_effect=profile_side_effect), \
             mock.patch.object(projections, "projected_corners", return_value=(5.2, 3.8)), \
             mock.patch.object(projections, "_recent_stats", side_effect=recent_side_effect), \
             mock.patch.object(projections, "_ml_blend_weight", return_value=0.0), \
             mock.patch.object(
                 projections,
                 "projected_total_corners",
                 return_value=(9.5, 9.0, 10.0),
             ):
            blended, season_proj, recent_proj = projections.projected_team_corners(
                "Arsenal", "Chelsea", "EPL", is_home=True
            )

        self.assertIsNotNone(blended)
        # Blended should be in a reasonable range for one team's corners
        self.assertGreater(blended, 3.0)
        self.assertLess(blended, 8.0)
        # Season/recent projections should exist
        self.assertIsNotNone(season_proj)
        self.assertIsNotNone(recent_proj)

    def test_team_corners_sum_near_match_total(self):
        """Sum of team-corner projections should be close to match total."""
        arsenal_meta = {
            "corners_pm": 5.5, "corners_home_pm": 6.0,
            "corners_against_pm": 4.5,
            "dominance_index": 0.50, "control_index": 0.60,
            "possession": 0.55,
        }
        chelsea_meta = {
            "corners_pm": 4.5, "corners_away_pm": 4.0,
            "corners_against_pm": 5.0,
            "dominance_index": 0.42, "control_index": 0.52,
            "possession": 0.48,
        }

        def profile_side_effect(team, league):
            return arsenal_meta if team == "Arsenal" else chelsea_meta

        recent_data = {
            "Arsenal": {"corners_for_avg": 6.0, "corners_against_avg": 4.5,
                        "shots_for_avg": 15.0, "sot_for_avg": 4.5,
                        "control_avg": 0.62, "possession_avg": 0.56,
                        "corners_for_slope": 0.0},
            "Chelsea": {"corners_for_avg": 4.2, "corners_against_avg": 5.2,
                        "shots_for_avg": 12.0, "sot_for_avg": 3.5,
                        "control_avg": 0.50, "possession_avg": 0.47,
                        "corners_for_slope": 0.0},
        }

        def recent_side_effect(team, league, last_n=6, venue=None):
            return recent_data.get(team, {})

        match_total = 9.8

        with mock.patch.object(projections, "_profile_meta", side_effect=profile_side_effect), \
             mock.patch.object(projections, "projected_corners", return_value=(5.2, 3.8)), \
             mock.patch.object(projections, "_recent_stats", side_effect=recent_side_effect), \
             mock.patch.object(projections, "_ml_blend_weight", return_value=0.0), \
             mock.patch.object(
                 projections,
                 "projected_total_corners",
                 return_value=(match_total, 9.0, 10.2),
             ):
            home_bl, _, _ = projections.projected_team_corners(
                "Arsenal", "Chelsea", "EPL", is_home=True
            )
            away_bl, _, _ = projections.projected_team_corners(
                "Chelsea", "Arsenal", "EPL", is_home=False
            )

        team_sum = home_bl + away_bl
        # Sum should be within 15% of match total (stabilizer causes some divergence)
        self.assertAlmostEqual(team_sum, match_total, delta=match_total * 0.15)

    def test_team_cards_use_match_total_share_and_stabilizer(self):
        arsenal_meta = {
            "cards_per_90_team": 1.8,
            "cards_home_pm": 2.0,
            "fouls_per_90_team": 10.0,
            "fouls_home_pm": 11.0,
            "cards_per_foul_team": 0.20,
            "aggression_index_norm": 0.50,
            "control_index": 0.62,
            "cards_induced_home_pm": 1.90,
            "opp_cards_induced_pm": 1.80,
        }
        chelsea_meta = {
            "cards_per_90_team": 1.4,
            "cards_away_pm": 1.3,
            "fouls_per_90_team": 9.0,
            "fouls_away_pm": 8.5,
            "cards_per_foul_team": 0.12,
            "aggression_index_norm": 0.30,
            "control_index": 0.68,
            "cards_induced_away_pm": 1.5,
            "opp_cards_induced_pm": 1.7,
        }

        def profile_side_effect(team, league):
            return arsenal_meta if team == "Arsenal" else chelsea_meta

        def recent_side_effect(team, league, last_n=6, venue=None):
            if team == "Arsenal":
                return {
                    "cards_avg": 2.2,
                    "cards_induced_avg": 1.6,
                    "fouls_avg": 10.5,
                    "aggression_avg": 0.55,
                    "control_avg": 0.61,
                }
            if team == "Chelsea":
                return {
                    "cards_avg": 1.4,
                    "cards_induced_avg": 1.4,
                    "fouls_avg": 9.2,
                    "aggression_avg": 0.28,
                    "control_avg": 0.69,
                }
            return {}

        ref_mod = SimpleNamespace(
            source="unavailable",
            avg_cards_per_match=0.0,
            sample_size=0,
            confidence=0.0,
            cards_per_foul=0.0,
        )

        with mock.patch.object(projections, "_profile_meta", side_effect=profile_side_effect), \
             mock.patch.object(projections, "_recent_stats", side_effect=recent_side_effect), \
             mock.patch.object(
                 projections,
                 "projected_total_cards",
                 return_value=(3.6, 3.4, 3.8, ref_mod),
             ):
            blended, season_proj, recent_proj = projections.projected_team_cards(
                "Arsenal", "Chelsea", "EPL", is_home=True, ref_mod=ref_mod
            )

        self.assertAlmostEqual(season_proj, 1.7795, places=4)
        self.assertAlmostEqual(recent_proj, 2.0502, places=4)
        # Output blend changed from 0.50/0.50 to 0.55/0.45 (cards_share_model weights)
        # share_anchor * 0.55 + stabilizer * 0.45
        self.assertIsNotNone(blended)
        self.assertGreater(blended, 1.5)
        self.assertLess(blended, 2.5)

    def test_team_cards_returns_three_tuple(self):
        """Backward compat: projected_team_cards() must return 3-tuple."""
        arsenal_meta = {"cards_per_90_team": 1.8, "opp_cards_induced_pm": 1.80}
        chelsea_meta = {"cards_per_90_team": 1.4, "opp_cards_induced_pm": 1.70}

        ref_mod = SimpleNamespace(
            source="unavailable", avg_cards_per_match=0.0,
            sample_size=0, confidence=0.0, cards_per_foul=0.0, multiplier=1.0,
        )

        with mock.patch.object(projections, "_profile_meta",
                               side_effect=lambda t, l: arsenal_meta if t == "Arsenal" else chelsea_meta), \
             mock.patch.object(projections, "_recent_stats", return_value={
                 "cards_avg": 1.8, "cards_induced_avg": 1.5,
                 "fouls_avg": 10.0, "aggression_avg": 0.4, "control_avg": 0.6,
             }), \
             mock.patch.object(projections, "projected_total_cards",
                               return_value=(3.6, 3.4, 3.8, ref_mod)):
            result = projections.projected_team_cards("Arsenal", "Chelsea", "EPL",
                                                       is_home=True, ref_mod=ref_mod)

        self.assertEqual(len(result), 3)
        blended, season_proj, recent_proj = result
        self.assertIsNotNone(blended)


if __name__ == "__main__":
    unittest.main()
