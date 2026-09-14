"""Regression coverage for team-name resolution shared by model and bot paths."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

import rag_cli_v2 as rag  # noqa: E402
from core import team_resolution  # noqa: E402


class TeamResolutionAliasTests(unittest.TestCase):
    def test_como_never_fuzzy_resolves_to_marseille(self):
        """A short club name must not be captured by Marseille's ``om`` alias."""
        for resolver in (team_resolution.canonical_team_name, rag.canonical_team_name):
            with self.subTest(resolver=resolver.__module__):
                self.assertEqual(resolver("Como"), "Como")
                self.assertEqual(resolver("Como 1907"), "Como")
                self.assertNotEqual(resolver("Como"), "Marseille")

    def test_short_aliases_remain_valid_when_explicitly_entered(self):
        """The safety guard applies only to fuzzy matching, not known acronyms."""
        for resolver in (team_resolution.canonical_team_name, rag.canonical_team_name):
            with self.subTest(resolver=resolver.__module__):
                self.assertEqual(resolver("OM"), "Marseille")
                self.assertEqual(resolver("PSG"), "Paris Saint Germain")

    def test_normal_typo_fuzzy_matching_is_preserved(self):
        for resolver in (team_resolution.canonical_team_name, rag.canonical_team_name):
            with self.subTest(resolver=resolver.__module__):
                self.assertEqual(resolver("Bournemoth"), "Bournemouth")


if __name__ == "__main__":
    unittest.main()
