from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace

if "discord" not in sys.modules:
    class _DummyEmbed:
        def __init__(self, *args, **kwargs):
            self.fields = []

        def add_field(self, *args, **kwargs):
            self.fields.append((args, kwargs))

        def set_footer(self, *args, **kwargs):
            return None

    sys.modules["discord"] = SimpleNamespace(Embed=_DummyEmbed)

if "config" not in sys.modules:
    sys.modules["config"] = SimpleNamespace(
        COLOR_BLUE=1,
        COLOR_GREEN=2,
        COLOR_ORANGE=3,
        COLOR_RED=4,
        COLOR_YELLOW=5,
    )

from Scripts.live_discord_bot.embeds import extract_live_bet_picks


class LiveEmbedsTests(unittest.TestCase):
    def test_extract_live_bet_picks_prefers_priced_alerts(self):
        snapshot = SimpleNamespace(
            alerts=[
                {
                    "type": "live_bet",
                    "pick": "⚽ Under 3.5 Goals",
                    "prob": 0.77,
                    "confidence": "Strong Edge",
                    "reason": "Model 77% vs fair 68%",
                    "mode": "odds_aware",
                    "pricing_source": "inplay",
                    "bookmaker": "Bet365",
                    "odds": 1.95,
                    "edge": 0.09,
                    "ev": 0.12,
                },
                {
                    "type": "live_bet",
                    "pick": "🤝 BTTS No",
                    "prob": 0.68,
                    "confidence": "Leaning",
                    "reason": "P(BTTS No) = 68%",
                    "mode": "model_only",
                    "pricing_source": "prematch_fallback",
                    "mode_note": "Live odds unavailable; using model-only projections while pre-match prices are stale.",
                },
            ]
        )

        picks = extract_live_bet_picks(snapshot)
        self.assertEqual(len(picks), 2)
        self.assertEqual(picks[0]["mode"], "odds_aware")
        self.assertIn("Bet365", picks[0]["reason"])
        self.assertIn("Live odds unavailable", picks[1]["reason"])


if __name__ == "__main__":
    unittest.main()
