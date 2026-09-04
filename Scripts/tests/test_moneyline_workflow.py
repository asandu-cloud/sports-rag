from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "Scripts" / "rag_ingest") not in sys.path:
    sys.path.insert(0, str(ROOT / "Scripts" / "rag_ingest"))
if str(ROOT / "Scripts" / "discord_bot") not in sys.path:
    sys.path.insert(0, str(ROOT / "Scripts" / "discord_bot"))

if "discord" not in sys.modules:
    sys.modules["discord"] = SimpleNamespace(Embed=object, File=object)

from Scripts.discord_bot.embeds import parse_fixture_picks
import rag_cli_v2 as rag
from Scripts.rag_ingest.core.line_selection import choose_best_moneyline_side


class MoneylineWorkflowTests(unittest.TestCase):
    def _render_legacy_moneyline(self, odds):
        """Render one fixture without needing local profiles or provider access."""
        event = {"home_team": "Home", "away_team": "Away"}
        with mock.patch.object(
            rag,
            "projected_moneyline_probs",
            return_value=(0.55, 0.25, 0.20, 1.64, 0.92),
        ), mock.patch.object(rag, "_profile_meta", return_value={}), mock.patch.object(
            rag, "extract_moneyline_odds", return_value=odds
        ):
            return rag.render_moneyline_answer("moneyline", "EPL", [event])

    def test_selector_separates_likely_winner_from_value_side(self):
        result = choose_best_moneyline_side(
            0.527,
            0.232,
            0.240,
            [
                {"side": "home", "odds": 1.22, "bookmaker": "Bet365", "team": "Paris Saint Germain"},
                {"side": "draw", "odds": 6.50, "bookmaker": "Bet365", "team": "Draw"},
                {"side": "away", "odds": 11.90, "bookmaker": "Bet365", "team": "Toulouse"},
            ],
            "Paris Saint Germain",
            "Toulouse",
        )

        self.assertEqual(result["most_likely"]["team"], "Paris Saint Germain")
        self.assertEqual(result["best_value"]["team"], "Toulouse")
        self.assertIs(result["recommended"], result["bet_recommendation"])

    def test_compact_parser_uses_most_likely_winner_for_moneyline(self):
        text = """Moneyline (1X2) advice by fixture:

=== Paris Saint Germain vs Toulouse ===
  Goal projections: Paris Saint Germain 1.82 — Toulouse 0.94
  Model: Paris Saint Germain 52.7% | Draw 23.2% | Toulouse 24.0%
  Odds comparison:
    Paris Saint Germain: 52.7% model @ 1.22 (Bet365) | Edge: -24.8% | EV: -0.357
    Draw: 23.2% model @ 6.50 (Bet365) | Edge: +8.7% | EV: +0.508
    Toulouse: 24.0% model @ 11.90 (Bet365) | Edge: +16.1% | EV: +1.856
  Most likely winner: Paris Saint Germain (52.7%)
  Best value side: Toulouse @ 11.90 (Bet365) | Edge: +16.1% | EV: +1.856
  Verdict: Recommended: Toulouse @ 11.90 (low confidence)

Method: Poisson/NegBin scoreline matrix from local KB → 3-way outcome probabilities.
"""

        picks = parse_fixture_picks(text)
        line = picks["Paris Saint Germain vs Toulouse"]

        self.assertIn("**Paris Saint Germain**", line)
        self.assertIn("52.7% model", line)
        self.assertIn("value Toulouse at 11.90", line)
        self.assertNotIn("**Toulouse** @ 11.90", line)

    def test_legacy_renderer_renders_no_bet_for_incomplete_1x2_prices(self):
        text = self._render_legacy_moneyline(
            [
                {"side": "home", "odds": 2.30, "bookmaker": "Book A", "team": "Home"},
                {"side": "draw", "odds": 3.50, "bookmaker": "Book B", "team": "Draw"},
                {"side": "away", "odds": 4.20, "bookmaker": "Book B", "team": "Away"},
            ]
        )

        self.assertIn("Verdict: No Bet", text)
        self.assertIn("complete same-bookmaker 1X2", text)
        self.assertIn("Most likely winner: Home (55.0%)", text)
        self.assertNotIn(">>> Recommended:", text)

    def test_legacy_renderer_renders_no_bet_when_valid_prices_fail_thresholds(self):
        text = self._render_legacy_moneyline(
            [
                {"side": "home", "odds": 1.90, "bookmaker": "Book", "team": "Home"},
                {"side": "draw", "odds": 3.40, "bookmaker": "Book", "team": "Draw"},
                {"side": "away", "odds": 4.50, "bookmaker": "Book", "team": "Away"},
            ]
        )

        self.assertIn("Verdict: No Bet", text)
        self.assertIn("minimum EV threshold", text)
        self.assertIn("Best value side: Home @ 1.90 (Book)", text)
        self.assertNotIn(">>> Recommended:", text)


if __name__ == "__main__":
    unittest.main()
