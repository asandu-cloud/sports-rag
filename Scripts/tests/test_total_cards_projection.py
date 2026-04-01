"""Tests for the cards-workflow rebuild: projected_total_cards_detail()."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from Scripts.rag_ingest.core import projections


class TotalCardsDetailTests(unittest.TestCase):
    """Tests for the new projected_total_cards_detail() function."""

    def _make_ref_mod(self, **kwargs):
        defaults = dict(
            multiplier=1.0, referee_name=None, sample_size=0,
            confidence=0.0, strictness_ratio=1.0, avg_cards_per_match=0.0,
            avg_fouls_per_match=0.0, cards_per_foul=0.0, source="unavailable",
        )
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_detail_returns_required_keys(self):
        """Detail helper must return all required keys."""
        home_meta = {"cards_per_90_team": 2.0, "opp_cards_induced_pm": 1.5}
        away_meta = {"cards_per_90_team": 1.5, "opp_cards_induced_pm": 2.0}

        ref_mod = self._make_ref_mod()

        with mock.patch.object(projections, "_profile_meta", side_effect=[home_meta, away_meta]), \
             mock.patch.object(projections, "_recent_stats", return_value={
                 "n": 10, "cards_avg": 2.0, "cards_induced_avg": 1.5,
                 "cards_slope": 0.0, "fouls_avg": 12.0,
                 "yellows_avg": 1.8, "reds_avg": 0.1,
             }), \
             mock.patch.object(projections, "get_referee_modifier", return_value=ref_mod), \
             mock.patch.object(projections, "_ml_blend_weight", return_value=0.0):
            detail = projections.projected_total_cards_detail("Home", "Away", "EPL")

        required_keys = {
            "total_cards", "total_yellows", "total_reds", "season_total",
            "recent_total", "referee_modifier", "lineup_card_context",
            "uncertainty", "confidence_bucket", "warnings",
        }
        self.assertTrue(required_keys.issubset(detail.keys()))

    def test_detail_total_equals_yellows_plus_reds(self):
        """total_cards should approximately equal total_yellows + total_reds."""
        home_meta = {"cards_per_90_team": 2.0, "opp_cards_induced_pm": 1.5}
        away_meta = {"cards_per_90_team": 1.5, "opp_cards_induced_pm": 2.0}

        ref_mod = self._make_ref_mod()

        with mock.patch.object(projections, "_profile_meta", side_effect=[home_meta, away_meta]), \
             mock.patch.object(projections, "_recent_stats", return_value={
                 "n": 10, "cards_avg": 2.0, "cards_induced_avg": 1.5,
                 "cards_slope": 0.0, "fouls_avg": 12.0,
                 "yellows_avg": None, "reds_avg": None,
             }), \
             mock.patch.object(projections, "get_referee_modifier", return_value=ref_mod), \
             mock.patch.object(projections, "_ml_blend_weight", return_value=0.0):
            detail = projections.projected_total_cards_detail("Home", "Away", "EPL")

        tc = detail["total_cards"]
        ty = detail["total_yellows"]
        tr = detail["total_reds"]
        self.assertIsNotNone(tc)
        self.assertAlmostEqual(tc, ty + tr, places=2)

    def test_backward_compatible_wrapper(self):
        """projected_total_cards() should still return the 4-tuple."""
        home_meta = {"cards_per_90_team": 2.0, "opp_cards_induced_pm": 1.5}
        away_meta = {"cards_per_90_team": 1.5, "opp_cards_induced_pm": 2.0}
        ref_mod = self._make_ref_mod()

        with mock.patch.object(projections, "_profile_meta", side_effect=[home_meta, away_meta]), \
             mock.patch.object(projections, "_recent_stats", return_value={
                 "n": 10, "cards_avg": 2.0, "cards_induced_avg": 1.5,
                 "cards_slope": 0.0, "fouls_avg": 12.0,
                 "yellows_avg": None, "reds_avg": None,
             }), \
             mock.patch.object(projections, "get_referee_modifier", return_value=ref_mod), \
             mock.patch.object(projections, "_ml_blend_weight", return_value=0.0):
            result = projections.projected_total_cards("Home", "Away", "EPL")

        self.assertEqual(len(result), 4)
        blended, season, recent, rm = result
        self.assertIsNotNone(blended)
        self.assertIsInstance(blended, float)

    def test_no_referee_increases_uncertainty(self):
        """Without referee, uncertainty should be higher."""
        home_meta = {"cards_per_90_team": 2.0, "opp_cards_induced_pm": 1.5}
        away_meta = {"cards_per_90_team": 1.5, "opp_cards_induced_pm": 2.0}
        ref_unavailable = self._make_ref_mod(source="unavailable")
        ref_available = self._make_ref_mod(
            source="profile", confidence=0.8, sample_size=20,
            avg_cards_per_match=4.0, strictness_ratio=1.1,
        )

        recent = {"n": 10, "cards_avg": 2.0, "cards_induced_avg": 1.5,
                  "cards_slope": 0.0, "fouls_avg": 12.0,
                  "yellows_avg": None, "reds_avg": None}

        with mock.patch.object(projections, "_profile_meta", side_effect=[home_meta, away_meta]), \
             mock.patch.object(projections, "_recent_stats", return_value=recent), \
             mock.patch.object(projections, "get_referee_modifier", return_value=ref_unavailable), \
             mock.patch.object(projections, "_ml_blend_weight", return_value=0.0):
            d_no_ref = projections.projected_total_cards_detail("Home", "Away", "EPL")

        with mock.patch.object(projections, "_profile_meta", side_effect=[home_meta, away_meta]), \
             mock.patch.object(projections, "_recent_stats", return_value=recent), \
             mock.patch.object(projections, "get_referee_modifier", return_value=ref_available), \
             mock.patch.object(projections, "_ml_blend_weight", return_value=0.0):
            d_with_ref = projections.projected_total_cards_detail("Home", "Away", "EPL",
                                                                   ref_mod=ref_available)

        self.assertGreater(d_no_ref["uncertainty"], d_with_ref["uncertainty"])

    def test_insufficient_data_returns_none(self):
        """When no profile data, total_cards should be None."""
        with mock.patch.object(projections, "_profile_meta", return_value={}), \
             mock.patch.object(projections, "_recent_stats", return_value={}), \
             mock.patch.object(projections, "get_referee_modifier",
                               return_value=self._make_ref_mod()), \
             mock.patch.object(projections, "_ml_blend_weight", return_value=0.0):
            detail = projections.projected_total_cards_detail("Home", "Away", "EPL")

        self.assertIsNone(detail["total_cards"])
        self.assertEqual(detail["confidence_bucket"], "low")


if __name__ == "__main__":
    unittest.main()
