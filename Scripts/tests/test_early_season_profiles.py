"""Regression tests for time-safe, sparse-season team profiles."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from core import team_resolution


def _row(season: str, fixture_date: str, corners: float, *, fixture: str) -> dict:
    return {
        "meta": {
            "season": season,
            "fixture_date": fixture_date,
            "fixture": fixture,
            "team": "Home",
            "home_away": "home",
            "final_score": "2-1",
            "corners_for": corners,
            "corners_against": 4.0,
            "shots_for": 12.0,
            "shots_against": 10.0,
            "sot_for": 5.0,
            "sot_against": 3.0,
            "cards_per_90_team": 2.0,
            "cards_total": 2.0,
            "yellow_cards": 2.0,
            "red_cards": 0.0,
            "fouls_per_90_team": 11.0,
            "fouls_committed": 11.0,
            "xg_for": 1.6,
            "possession": 0.52,
            "control_index": 0.55,
            "aggression_index_norm": 0.25,
            "form_index_team": 0.50,
        },
        "text": "",
    }


class EarlySeasonProfileTests(unittest.TestCase):
    def setUp(self):
        team_resolution._team_profile_context_cache.clear()
        team_resolution._team_fixture_rows_cache.clear()

    def test_dated_profile_uses_only_fixture_rows_before_the_target_date(self):
        rows = [
            _row("2025/26", "2026-05-10", 4.0, fixture="prior-1"),
            _row("2025/26", "2026-05-17", 4.0, fixture="prior-2"),
            _row("2026/27", "2026-08-10", 10.0, fixture="current-before"),
            # These rows must not influence a prediction for 20 August.
            _row("2026/27", "2026-08-20", 100.0, fixture="target-fixture"),
            _row("2026/27", "2026-08-27", 100.0, fixture="future-fixture"),
        ]
        documents = [
            {"meta": {"season": "2026/27", "matches_played": 3, "corners_pm": 100.0}},
            {"meta": {"season": "2025/26", "matches_played": 38, "corners_pm": 4.0}},
        ]

        with mock.patch.object(team_resolution, "get_team_profile_docs", return_value=documents), \
             mock.patch.object(team_resolution, "_get_all_team_fixture_rows", return_value=rows):
            profile, audit = team_resolution.get_team_profile_context(
                "Home", "EPL", target_date="2026-08-20T15:00:00Z",
            )

        self.assertEqual(audit["temporal_status"], "fixture_rows_strictly_before_target_date")
        self.assertEqual(audit["current_season_matches"], 1)
        self.assertEqual(audit["prior_season_matches"], 2)
        self.assertAlmostEqual(audit["prior_weight"], 8 / 9, places=4)
        # 1 current match at 10 corners shrunk toward a prior rate of 4.
        self.assertAlmostEqual(profile["corners_pm"], 4.6666666667, places=6)
        self.assertLess(profile["corners_pm"], 10.0)
        self.assertEqual(profile["_profile_source"], "fixture_rows_as_of")

    def test_recent_fixture_window_excludes_the_target_fixture_and_same_day_rows(self):
        rows = [
            _row("2026/27", "2026-08-10", 6.0, fixture="before"),
            _row("2026/27", "2026-08-20", 7.0, fixture="same-day"),
            _row("2026/27", "2026-08-27", 8.0, fixture="after"),
            _row("2025/26", "2026-05-10", 4.0, fixture="prior-season"),
        ]

        with mock.patch.object(team_resolution, "_get_all_team_fixture_rows", return_value=rows):
            recent = team_resolution.get_recent_team_fixture_rows(
                "Home", "EPL", limit=6, target_date="2026-08-20T15:00:00Z",
            )

        self.assertEqual([row["meta"]["fixture"] for row in recent], ["before"])

    def test_sparse_current_profile_is_shrunk_toward_the_prior_and_preserves_current_sample_count(self):
        current = {"season": "2026/27", "matches_played": 1, "corners_pm": 9.0}
        prior = {"season": "2025/26", "matches_played": 38, "corners_pm": 3.0}

        profile, prior_weight, mode = team_resolution._blend_sparse_current_profile(current, prior)

        self.assertEqual(mode, "current_plus_prior")
        self.assertEqual(profile["matches_played"], 1)
        self.assertAlmostEqual(prior_weight, 8 / 9)
        self.assertAlmostEqual(profile["corners_pm"], 3.6666666667, places=6)


if __name__ == "__main__":
    unittest.main()
