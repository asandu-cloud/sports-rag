from __future__ import annotations

import unittest
from pathlib import Path

from Scripts.discord_bot.cogs.whale_chat_policy import (
    BETTING_REDIRECT_INSTRUCTION,
    BASE_SYSTEM_PROMPT,
    is_betting_request,
    runtime_instruction_for_message,
)


class WhaleChatPolicyTests(unittest.TestCase):
    def test_detects_betting_requests(self):
        self.assertTrue(is_betting_request("Build me a 3-leg parlay for Saturday"))
        self.assertTrue(is_betting_request("What odds are out for Arsenal vs Chelsea?"))
        self.assertTrue(is_betting_request("What line should I take on Bayern vs Dortmund?"))

    def test_allows_stats_questions(self):
        self.assertFalse(is_betting_request("Compare Arsenal and Chelsea recent form."))
        self.assertFalse(is_betting_request("How many corners do you project for Liverpool vs Brighton?"))
        self.assertFalse(is_betting_request("Which games are on this weekend in LaLiga?"))

    def test_runtime_instruction_redirects_betting_requests(self):
        self.assertEqual(
            runtime_instruction_for_message("Can you build me an acca?"),
            BETTING_REDIRECT_INSTRUCTION,
        )
        self.assertIsNone(runtime_instruction_for_message("Give me the stats on Inter vs Milan."))

    def test_base_prompt_is_stats_first(self):
        self.assertIn("statistics desk", BASE_SYSTEM_PROMPT)
        self.assertIn("do not quote bookmaker odds", BASE_SYSTEM_PROMPT)
        self.assertIn("`/parlay`", BASE_SYSTEM_PROMPT)

    def test_whale_chat_tool_surface_is_stats_only(self):
        whale_chat = Path(__file__).resolve().parents[1] / "discord_bot" / "cogs" / "whale_chat.py"
        source = whale_chat.read_text(encoding="utf-8")
        self.assertIn('"name": "get_fixture_snapshot"', source)
        self.assertNotIn('"name": "get_odds"', source)
        self.assertNotIn('"name": "recommend_line"', source)
        self.assertNotIn('"name": "build_parlay"', source)
        self.assertNotIn('"name": "recommend_spread"', source)


if __name__ == "__main__":
    unittest.main()
