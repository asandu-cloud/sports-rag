"""Tests for referee_data.py — profile building, modifier computation, fixture detail."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from Scripts.rag_ingest.referee_data import (
    RefereeModifier,
    RefereeProfile,
    compute_referee_modifier,
    _normalize_ref_name,
    _save_fixture_referee_detail,
    load_fixture_referee_detail,
    get_fixture_referee_detail,
    FIXTURE_REFEREE_DETAIL_PATH,
)


class NormalizeRefNameTests(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(_normalize_ref_name("Michael Oliver, England"), "michael oliver")

    def test_empty(self):
        self.assertIsNone(_normalize_ref_name(""))
        self.assertIsNone(_normalize_ref_name(None))

    def test_no_country(self):
        self.assertEqual(_normalize_ref_name("Michael Oliver"), "michael oliver")


class ComputeRefereeModifierTests(unittest.TestCase):
    def test_unavailable_when_no_name(self):
        mod = compute_referee_modifier(None)
        self.assertEqual(mod.source, "unavailable")
        self.assertEqual(mod.multiplier, 1.0)

    def test_profile_found(self):
        profile = RefereeProfile(
            name="test ref",
            total_matches=25,
            avg_cards_per_match=5.0,
            avg_fouls_per_match=22.0,
            strictness_ratio=1.15,
            cards_per_foul=0.23,
        )
        profiles = {"test ref": profile}

        with mock.patch("Scripts.rag_ingest.referee_data._ensure_profiles_loaded",
                         return_value=profiles):
            mod = compute_referee_modifier("Test Ref", weight=0.35)

        self.assertEqual(mod.source, "profile")
        self.assertGreater(mod.multiplier, 1.0)
        self.assertEqual(mod.sample_size, 25)
        self.assertAlmostEqual(mod.strictness_ratio, 1.15)

    def test_low_sample_no_modifier(self):
        profile = RefereeProfile(
            name="new ref",
            total_matches=3,
            avg_cards_per_match=6.0,
            strictness_ratio=1.5,
        )
        profiles = {"new ref": profile}

        with mock.patch("Scripts.rag_ingest.referee_data._ensure_profiles_loaded",
                         return_value=profiles):
            mod = compute_referee_modifier("New Ref")

        self.assertEqual(mod.source, "profile")
        self.assertEqual(mod.multiplier, 1.0)  # below MIN_SAMPLE → no multiplier


class FixtureRefDetailTests(unittest.TestCase):
    def test_save_and_load(self):
        """Test that fixture detail can be saved and loaded back."""
        detail_rows = [
            {"fixture_id": 123, "league": "EPL", "season": 2025,
             "referee_name": "test ref", "total_cards": 4, "total_yellows": 4,
             "total_reds": 0, "total_fouls": 20},
        ]
        profiles = {
            "test ref": RefereeProfile(
                name="test ref", total_matches=10,
                avg_cards_per_match=4.2, avg_fouls_per_match=22.0,
                avg_yellows_per_match=3.8, avg_reds_per_match=0.4,
                strictness_ratio=1.05, cards_per_foul=0.19,
            )
        }

        import Scripts.rag_ingest.referee_data as rd
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            tmp_path = Path(f.name)

        original_path = rd.FIXTURE_REFEREE_DETAIL_PATH
        try:
            rd.FIXTURE_REFEREE_DETAIL_PATH = tmp_path
            _save_fixture_referee_detail(detail_rows, profiles)

            # Reset cache
            rd._fixture_referee_detail_cache = None

            loaded = load_fixture_referee_detail()
            self.assertIn(123, loaded)
            row = loaded[123]
            self.assertEqual(row["referee_name"], "test ref")
            self.assertAlmostEqual(row["avg_cards_per_match"], 4.2)
            self.assertAlmostEqual(row["strictness_ratio"], 1.05)

            # Test individual lookup
            rd._fixture_referee_detail_cache = None
            single = get_fixture_referee_detail(123)
            self.assertIsNotNone(single)
            self.assertEqual(single["league"], "EPL")
        finally:
            rd.FIXTURE_REFEREE_DETAIL_PATH = original_path
            rd._fixture_referee_detail_cache = None
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
