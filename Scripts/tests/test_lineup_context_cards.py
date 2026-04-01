"""Tests for lineup_context.py card-risk profiling."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from Scripts.rag_ingest.lineup_context import (
    CardRiskProfile,
    LineupContext,
    compute_card_risk_profile,
    card_risk_evidence_line,
    get_card_risk_profiles,
)


class CardRiskProfileTests(unittest.TestCase):
    def test_default_values(self):
        p = CardRiskProfile()
        self.assertEqual(p.team_starter_cards_per_90, 0.0)
        self.assertEqual(p.high_card_risk_players, 0)
        self.assertEqual(p.n_players, 0)

    def test_compute_with_mocked_chroma(self):
        """Test card risk computation with mocked Chroma data."""
        mock_players = [
            {"player_name": "Player A", "position": "Midfielder",
             "starter_cards_per_90": 0.30, "starter_fouls_per_90": 2.0,
             "minutes_risk": 0.9, "total_minutes": 2000},
            {"player_name": "Player B", "position": "Defender",
             "starter_cards_per_90": 0.28, "starter_fouls_per_90": 1.8,
             "minutes_risk": 0.85, "total_minutes": 1800},
            {"player_name": "Player C", "position": "Attacker",
             "starter_cards_per_90": 0.05, "starter_fouls_per_90": 0.3,
             "minutes_risk": 0.8, "total_minutes": 1600},
        ]

        mock_col = mock.MagicMock()
        mock_col.get.return_value = {"metadatas": mock_players}

        with mock.patch(
            "Scripts.rag_ingest.lineup_context._get_chroma_collection",
            return_value=mock_col,
        ):
            profile = compute_card_risk_profile("TestTeam", "EPL")

        self.assertEqual(profile.n_players, 3)
        self.assertGreater(profile.team_starter_cards_per_90, 0)
        self.assertEqual(profile.high_card_risk_players, 2)  # A and B >= 0.25
        self.assertEqual(profile.high_foul_players, 2)  # A and B >= 1.5

    def test_compute_no_chroma_returns_empty(self):
        """When Chroma is unavailable, should return empty profile."""
        with mock.patch(
            "Scripts.rag_ingest.lineup_context._get_chroma_collection",
            side_effect=Exception("not available"),
        ):
            profile = compute_card_risk_profile("TestTeam", "EPL")

        self.assertEqual(profile.n_players, 0)


class CardRiskEvidenceTests(unittest.TestCase):
    def test_evidence_line_with_risk(self):
        ctx = LineupContext()
        ctx.home_card_risk = CardRiskProfile(
            team_starter_cards_per_90=2.5,
            high_card_risk_players=3,
            n_players=11,
        )
        ctx.away_card_risk = CardRiskProfile(
            team_starter_cards_per_90=1.2,
            high_card_risk_players=1,
            n_players=11,
        )

        line = card_risk_evidence_line(ctx)
        self.assertIsNotNone(line)
        self.assertIn("Home", line)
        self.assertIn("3 high-card-risk", line)

    def test_evidence_line_no_risk(self):
        ctx = LineupContext()
        ctx.home_card_risk = CardRiskProfile(n_players=11, high_card_risk_players=0)
        ctx.away_card_risk = CardRiskProfile(n_players=11, high_card_risk_players=1)

        line = card_risk_evidence_line(ctx)
        self.assertIsNone(line)  # Neither side has >= 2 high-risk

    def test_get_card_risk_profiles_no_chroma(self):
        with mock.patch(
            "Scripts.rag_ingest.lineup_context._get_chroma_collection",
            side_effect=Exception("not available"),
        ):
            h, a = get_card_risk_profiles("Home", "Away", "EPL")
        self.assertEqual(h.n_players, 0)
        self.assertEqual(a.n_players, 0)


if __name__ == "__main__":
    unittest.main()
