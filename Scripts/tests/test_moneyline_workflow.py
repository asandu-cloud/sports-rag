from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

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
from Scripts.rag_ingest.core.line_selection import choose_best_moneyline_side


class MoneylineWorkflowTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
