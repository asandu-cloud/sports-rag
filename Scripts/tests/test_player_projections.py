"""Tests for player_projections.py — gating, role heuristics, line feasibility."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from Scripts.rag_ingest.player_projections import (
    PLAYER_PROP_WEIGHTS,
    infer_player_roles,
    projected_player_stat,
)


class PlayerRoleHeuristicTests(unittest.TestCase):
    def test_penalty_taker_high_conversion(self):
        meta = {"starter_goals_per_90": 0.45, "shots_per_90": 1.5}
        roles = infer_player_roles(meta)
        self.assertTrue(roles["probable_penalty_taker"])

    def test_normal_conversion_not_penalty(self):
        meta = {"starter_goals_per_90": 0.15, "shots_per_90": 2.0}
        roles = infer_player_roles(meta)
        self.assertFalse(roles["probable_penalty_taker"])

    def test_set_piece_creator(self):
        meta = {"starter_assists_per_90": 0.30, "position": "Midfielder"}
        roles = infer_player_roles(meta)
        self.assertTrue(roles["probable_set_piece"])

    def test_high_usage_creator(self):
        meta = {"starter_goals_per_90": 0.35, "starter_assists_per_90": 0.30}
        roles = infer_player_roles(meta)
        self.assertTrue(roles["high_usage_creator"])

    def test_text_penalty_detection(self):
        meta = {"starter_goals_per_90": 0.10, "shots_per_90": 1.0}
        roles = infer_player_roles(meta, player_text="Penalty taker for the team")
        self.assertTrue(roles["probable_penalty_taker"])

    def test_empty_meta_no_crash(self):
        roles = infer_player_roles({})
        self.assertFalse(roles["probable_penalty_taker"])
        self.assertFalse(roles["probable_set_piece"])


class PlayerGatingTests(unittest.TestCase):
    def test_bench_skip_default(self):
        """Bench skip should be enabled by default."""
        self.assertTrue(PLAYER_PROP_WEIGHTS.get("bench_skip_entirely", True))

    def test_min_minutes_for_rec(self):
        """Minimum expected minutes for recommendation should be set."""
        self.assertGreaterEqual(PLAYER_PROP_WEIGHTS.get("min_expected_minutes_for_rec", 0), 30)

    def test_rotation_max_line(self):
        """Rotation players should be capped at 0.5 lines."""
        self.assertEqual(PLAYER_PROP_WEIGHTS.get("rotation_max_line", 0.5), 0.5)


class PlayerProjectionRoleUsageTests(unittest.TestCase):
    def test_penalty_role_increases_goal_projection(self):
        player_meta = {
            "player_name": "Test Forward",
            "player_id": 10,
            "team": "Home",
            "position": "F",
            "minutes_risk": "starter",
            "start_rate": 0.82,
            "total_minutes": 900,
            "qualified_appearances": 10,
            "appearances_total": 12,
            "avg_minutes_per_appearance": 81,
            "starter_goals_per_90": 0.24,
            "shots_per_90": 1.2,
        }
        team_meta = {"goals_for_pm": 1.7, "xg_home_pm": 1.8, "xg_away_pm": 1.4}
        opp_meta = {"goals_against_pm": 1.3, "sot_against_pm": 4.1}

        base = projected_player_stat(
            player_meta, team_meta, opp_meta,
            stat="goals", league="EPL", is_home=True,
            player_text="",
        )
        with_penalties = projected_player_stat(
            player_meta, team_meta, opp_meta,
            stat="goals", league="EPL", is_home=True,
            player_text="Primary penalty taker",
        )

        self.assertIsNotNone(base)
        self.assertIsNotNone(with_penalties)
        self.assertGreater(with_penalties.projection, base.projection)
        self.assertTrue(with_penalties.evidence["probable_penalty_taker"])


if __name__ == "__main__":
    unittest.main()
