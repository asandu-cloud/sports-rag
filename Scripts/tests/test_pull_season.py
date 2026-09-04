"""Regression coverage for the legacy season-refresh bridge."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "Scripts" / "pull_season.py"


def _load_pull_season_module():
    """Load the executable script without relying on package layout."""
    spec = importlib.util.spec_from_file_location("pull_season_under_test", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlayerProfileRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pull_season = _load_pull_season_module()

    def test_domestic_player_profile_uses_fresh_feature_file_and_season_label(self):
        """A domestic refresh invokes its profile builder with explicit inputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            script = temp_root / "Scripts" / "Premier_League" / "Prem_player_profile.py"
            script.parent.mkdir(parents=True)
            script.touch()
            feature_file = (
                temp_root
                / "Output"
                / "Prem_feature_engineering"
                / "player_engineered_features_2026.json"
            )
            feature_file.parent.mkdir(parents=True)
            feature_file.write_text("[]", encoding="utf-8")

            with (
                mock.patch.object(self.pull_season, "ROOT", temp_root),
                mock.patch.object(self.pull_season, "run_command", return_value=True) as run_command,
            ):
                result = self.pull_season.run_player_profile("EPL", 2026)

        self.assertTrue(result)
        run_command.assert_called_once_with(
            [
                sys.executable,
                str(script),
                "--input",
                str(feature_file),
                "--league",
                "EPL",
                "--season",
                "2026/27",
                "--profiles_basename",
                "player_profiles_2026",
            ],
            "EPL player profiles",
        )

    def test_missing_feature_output_fails_before_profile_builder_runs(self):
        """Profile files must never be rebuilt from an absent/stale input path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            script = temp_root / "Scripts" / "Premier_League" / "Prem_player_profile.py"
            script.parent.mkdir(parents=True)
            script.touch()

            with (
                mock.patch.object(self.pull_season, "ROOT", temp_root),
                mock.patch.object(self.pull_season, "run_command") as run_command,
            ):
                result = self.pull_season.run_player_profile("EPL", 2026)

        self.assertFalse(result)
        run_command.assert_not_called()


class PullSeasonFailFastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pull_season = _load_pull_season_module()

    def test_profile_failure_stops_pipeline_before_later_stages(self):
        """A failed profile aggregation must block normalize/embed/retraining."""
        with (
            mock.patch.object(self.pull_season, "run_script", side_effect=[True, True, True]) as run_script,
            mock.patch.object(self.pull_season, "run_player_profile", return_value=False) as run_profiles,
            mock.patch.object(self.pull_season, "run_command") as run_command,
        ):
            result = self.pull_season.main(
                [
                    "--season",
                    "2026",
                    "--league",
                    "EPL",
                    "--normalize",
                    "--embed",
                    "--retrain-ml",
                ]
            )

        self.assertEqual(result, 1)
        self.assertEqual(run_script.call_count, 3)
        run_profiles.assert_called_once_with("EPL", 2026)
        run_command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
