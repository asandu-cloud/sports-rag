"""Regression coverage for current-season fixture discovery."""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Scripts" / "rag_ingest"))
sys.path.insert(0, str(ROOT / "Scripts"))

import odds_provider  # noqa: E402
import rag_cli_v2  # noqa: E402


class CurrentSeasonEventRoutingTests(unittest.TestCase):
    def test_season_year_rolls_over_at_the_start_of_a_new_campaign(self):
        self.assertEqual(odds_provider.api_football_season_for_date(date(2026, 6, 30)), 2025)
        self.assertEqual(odds_provider.api_football_season_for_date(date(2026, 7, 1)), 2026)
        self.assertEqual(odds_provider.api_football_season_for_date(date(2027, 1, 1)), 2026)

    def test_fixture_lookup_uses_season_derived_from_target_date(self):
        with mock.patch.object(odds_provider, "_api_get", return_value=([], None)) as api_get:
            rows, error = odds_provider._fetch_fixtures("LaLiga", date(2026, 8, 26))

        self.assertEqual(rows, [])
        self.assertIsNone(error)
        self.assertEqual(api_get.call_args.args[0], "/fixtures")
        self.assertEqual(api_get.call_args.args[1]["season"], 2026)
        self.assertEqual(api_get.call_args.args[1]["date"], "2026-08-26")

    def test_legacy_rag_helper_routes_date_aware_discovery_to_canonical_events(self):
        fixture_day = date(2026, 8, 26)
        expected = [{"id": "fixture-1", "home_team": "A", "away_team": "B"}]
        with mock.patch("core.events.fetch_events", return_value=(expected, ["provider note"])) as fetch:
            events, notes = rag_cli_v2.fetch_events("LaLiga", {"totals"}, target_date=fixture_day)

        fetch.assert_called_once_with("LaLiga", {"totals"}, target_date=fixture_day)
        self.assertEqual(events, expected)
        self.assertEqual(notes, ["provider note"])


if __name__ == "__main__":
    unittest.main()
