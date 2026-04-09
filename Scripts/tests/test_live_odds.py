from __future__ import annotations

import unittest

from Scripts.live.live_alerts import clear_cooldowns, detect_alerts
from Scripts.live.live_odds import best_total_offer, summarize_live_odds
from Scripts.live.live_state import LiveMatchState, LiveSnapshot


class LiveOddsTests(unittest.TestCase):
    def test_summarize_live_odds_builds_two_way_line_summary(self):
        bookmakers = [
            {
                "title": "Bet365",
                "markets": [
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": 2.5, "price": 2.05},
                            {"name": "Under", "point": 2.5, "price": 1.80},
                        ],
                    }
                ],
            }
        ]
        snapshot = summarize_live_odds(bookmakers, "PSG", "Toulouse", captured_at="2026-04-04T18:00:00+00:00")
        offer = best_total_offer(snapshot, "goals", 2.5, "over")
        self.assertIsNotNone(offer)
        self.assertEqual(offer["bookmaker"], "Bet365")
        self.assertAlmostEqual(offer["odds"], 2.05)
        self.assertGreater(offer["fair_prob"], 0.0)

    def test_detect_alerts_prefers_odds_aware_edge(self):
        clear_cooldowns()
        live_odds = summarize_live_odds(
            [
                {
                    "title": "Bet365",
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "point": 2.5, "price": 2.10},
                                {"name": "Under", "point": 2.5, "price": 1.74},
                            ],
                        }
                    ],
                }
            ],
            "PSG",
            "Toulouse",
            captured_at="2026-04-04T18:00:00+00:00",
        )
        live_odds["source"] = "inplay"
        state = LiveMatchState(
            fixture_id=77,
            league="Ligue1",
            home_team="PSG",
            away_team="Toulouse",
            status="2H",
            elapsed_min=63.0,
            goals_home=1,
            goals_away=0,
            prior_goals=2.6,
            prior_goals_home=1.8,
            prior_goals_away=0.8,
            live_odds=live_odds,
            live_odds_source="inplay",
        )
        snapshot = LiveSnapshot(
            fixture_id=77,
            home_team="PSG",
            away_team="Toulouse",
            league="Ligue1",
            elapsed_min=63.0,
            status="2H",
            score="1 - 0",
            remaining_goals=1.2,
            remaining_corners=2.0,
            remaining_cards=1.0,
            remaining_sot=1.5,
            projected_total_goals=3.1,
            projected_total_corners=9.0,
            projected_total_cards=4.0,
            projected_total_sot=7.0,
            prob_over_goals={"2.5": 0.66},
            prob_over_corners={},
            prob_over_cards={},
            prob_over_sot={},
            btts_prob=0.42,
            moneyline={"home_win": 0.71, "draw": 0.19, "away_win": 0.10},
            live_odds=live_odds,
            live_odds_source="inplay",
        )

        alerts = detect_alerts(snapshot, None, state)
        self.assertTrue(alerts)
        self.assertIn("edge", alerts[0]["message"].lower())
        self.assertIn("2.10", alerts[0]["message"])
        self.assertEqual(alerts[0]["mode"], "odds_aware")
        self.assertEqual(alerts[0]["pricing_source"], "inplay")

    def test_detect_alerts_marks_prematch_fallback_as_model_only(self):
        clear_cooldowns()
        live_odds = summarize_live_odds(
            [
                {
                    "title": "Bet365",
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "point": 2.5, "price": 2.10},
                                {"name": "Under", "point": 2.5, "price": 1.74},
                            ],
                        }
                    ],
                }
            ],
            "PSG",
            "Toulouse",
            captured_at="2026-04-04T18:00:00+00:00",
        )
        live_odds["source"] = "prematch_fallback"
        state = LiveMatchState(
            fixture_id=78,
            league="Ligue1",
            home_team="PSG",
            away_team="Toulouse",
            status="2H",
            elapsed_min=63.0,
            goals_home=1,
            goals_away=0,
            prior_goals=2.6,
            prior_goals_home=1.8,
            prior_goals_away=0.8,
            live_odds=live_odds,
            live_odds_source="prematch_fallback",
        )
        snapshot = LiveSnapshot(
            fixture_id=78,
            home_team="PSG",
            away_team="Toulouse",
            league="Ligue1",
            elapsed_min=63.0,
            status="2H",
            score="1 - 0",
            remaining_goals=1.2,
            remaining_corners=2.0,
            remaining_cards=1.0,
            remaining_sot=1.5,
            projected_total_goals=3.1,
            projected_total_corners=9.0,
            projected_total_cards=4.0,
            projected_total_sot=7.0,
            prob_over_goals={"2.5": 0.66},
            prob_over_corners={},
            prob_over_cards={},
            prob_over_sot={},
            btts_prob=0.42,
            moneyline={"home_win": 0.71, "draw": 0.19, "away_win": 0.10},
            live_odds=live_odds,
            live_odds_source="prematch_fallback",
        )

        alerts = detect_alerts(snapshot, None, state)
        self.assertTrue(alerts)
        self.assertEqual(alerts[0]["mode"], "model_only")
        self.assertEqual(alerts[0]["pricing_source"], "prematch_fallback")
        self.assertIn("Live odds unavailable", alerts[0]["message"])
        self.assertNotIn("EV", alerts[0]["message"])
        self.assertNotIn("2.10", alerts[0]["message"])


if __name__ == "__main__":
    unittest.main()
