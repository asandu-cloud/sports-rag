from __future__ import annotations

import sys
import tempfile
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
from Scripts.rag_ingest.core.line_selection import select_best_spread_recommendation
from Scripts.rag_ingest.prediction_tracker import log_prediction


class SpreadWorkflowTests(unittest.TestCase):
    def test_selector_uses_asian_handicap_settlement_profile(self):
        matrix = {
            (1, 0): 0.30,  # away +0.75 => half loss
            (0, 0): 0.25,
            (0, 1): 0.20,
            (2, 0): 0.15,
            (1, 1): 0.10,
        }
        with mock.patch(
            "Scripts.rag_ingest.core.line_selection.projected_correct_score_probs",
            return_value=(matrix, 1.2, 0.9),
        ):
            result = select_best_spread_recommendation(
                [
                    {"team": "Home FC", "point": -0.75, "odds": 1.95, "bookmaker": "Bet365", "market_key": "spreads"},
                    {"team": "Away FC", "point": 0.75, "odds": 1.90, "bookmaker": "Bet365", "market_key": "spreads"},
                ],
                projected_diff=0.35,
                home_team="Home FC",
                away_team="Away FC",
                league="EPL",
            )

        best = result["bet_recommendation"]
        self.assertIsNotNone(best)
        self.assertEqual(best["team"], "Away FC")
        self.assertGreater(best["_ev"], 0.15)
        self.assertGreater(best["_model_prob"], 0.60)

    def test_selector_can_return_no_bet_when_spread_is_thin(self):
        matrix = {
            (1, 0): 0.25,
            (0, 0): 0.25,
            (0, 1): 0.25,
            (2, 1): 0.25,
        }
        with mock.patch(
            "Scripts.rag_ingest.core.line_selection.projected_correct_score_probs",
            return_value=(matrix, 1.0, 0.9),
        ):
            result = select_best_spread_recommendation(
                [
                    {"team": "Home FC", "point": -0.5, "odds": 1.95, "bookmaker": "Bet365", "market_key": "spreads"},
                    {"team": "Away FC", "point": 0.5, "odds": 1.95, "bookmaker": "Bet365", "market_key": "spreads"},
                ],
                projected_diff=0.10,
                home_team="Home FC",
                away_team="Away FC",
                league="EPL",
            )

        self.assertIsNone(result["bet_recommendation"])
        self.assertIsNotNone(result["best_value"])
        self.assertIn("threshold", (result["no_bet_reason"] or "").lower())

    def test_parser_skips_no_bet_spread_blocks(self):
        text = """Score Handicap advice by fixture:
- Home FC vs Away FC: projected goal difference +0.18 (Home FC favored).
  Reasoning: Season model diff +0.10.
  Best value line: Away FC +0.5 @ 1.91 (Bet365, spreads).
  Pricing check: 54.8% equivalent cover | Value edge: +0.7% | EV: +0.009
  Verdict: No spread bet — Best spread line does not clear the minimum EV threshold.
  Model lean: Away FC +0.5 (model-only, P(cover) = 54.9%).

Method: scoreline matrix + Asian handicap settlement model from local KB projections and available spread lines.
"""
        picks = parse_fixture_picks(text)
        self.assertEqual(picks, {})

    def test_prediction_tracker_dedupes_same_day_identical_pick(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "predictions.db"
            first = log_prediction(
                home_team="Home FC",
                away_team="Away FC",
                league="EPL",
                market="spreads",
                pick="Away FC +0.5",
                side="away",
                line=0.5,
                odds=1.91,
                bookmaker="Bet365",
                source="auto_push",
                db_path=db_path,
            )
            second = log_prediction(
                home_team="Home FC",
                away_team="Away FC",
                league="EPL",
                market="spreads",
                pick="Away FC +0.5",
                side="away",
                line=0.5,
                odds=1.91,
                bookmaker="Bet365",
                source="auto_push",
                db_path=db_path,
            )

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
