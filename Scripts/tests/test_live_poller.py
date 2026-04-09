from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from Scripts.live import live_poller


class LivePollerTests(unittest.TestCase):
    def test_compute_priors_populates_moneyline(self):
        state = live_poller.LiveMatchState(
            fixture_id=101,
            league="Ligue1",
            league_id=61,
            home_team="PSG",
            away_team="Toulouse",
        )
        with mock.patch.object(live_poller, "_HAS_RAG", True), \
             mock.patch.object(live_poller, "_profile_meta", side_effect=[{}, {}]), \
             mock.patch.object(live_poller, "projected_total_goals", return_value=(3.1, 2.9, 3.3)), \
             mock.patch.object(live_poller, "projected_goals", return_value=(2.2, 0.9)), \
             mock.patch.object(live_poller, "projected_total_corners", return_value=(10.4, 10.0, 10.8)), \
             mock.patch.object(live_poller, "projected_total_cards", return_value=(4.1, 4.0, 4.2, 1.0)), \
             mock.patch.object(live_poller, "projected_total_sot", return_value=(9.3, 9.0, 9.6)), \
             mock.patch.object(live_poller, "projected_btts_prob", return_value=(0.46,)), \
             mock.patch.object(live_poller, "projected_moneyline_probs", return_value=(0.73, 0.17, 0.10, 2.2, 0.9)):
            live_poller._compute_priors("PSG", "Toulouse", "Ligue1", state)

        self.assertEqual(state.prior_total_goals, 3.1)
        self.assertEqual(state.prior_home_goals, 2.2)
        self.assertEqual(state.prior_away_goals, 0.9)
        self.assertEqual(
            state.prior_moneyline,
            {"home_win": 0.73, "draw": 0.17, "away_win": 0.10},
        )

    def test_to_engine_state_carries_moneyline(self):
        poller = live_poller.LivePoller(auto_subscribe=False)
        state = live_poller.LiveMatchState(
            fixture_id=202,
            league="EPL",
            league_id=39,
            home_team="Arsenal",
            away_team="Chelsea",
            prior_total_goals=2.8,
            prior_home_goals=1.6,
            prior_away_goals=1.2,
            prior_total_corners=10.2,
            prior_total_cards=4.0,
            prior_total_sot=8.9,
            prior_btts_prob=0.58,
            prior_moneyline={"home_win": 0.49, "draw": 0.26, "away_win": 0.25},
        )
        state.home_stats.goals = 1
        state.away_stats.goals = 0
        engine_state = poller._to_engine_state(state)
        self.assertEqual(engine_state.prior_moneyline, state.prior_moneyline)
        self.assertEqual(engine_state.prior_goals, 2.8)
        self.assertEqual(engine_state.goals_home, 1)

    def test_attach_snapshot_alerts_uses_detector(self):
        poller = live_poller.LivePoller(auto_subscribe=False)
        state = live_poller.LiveMatchState(
            fixture_id=303,
            league="SerieA",
            league_id=135,
            home_team="Inter",
            away_team="Milan",
            prior_total_goals=2.7,
            prior_home_goals=1.4,
            prior_away_goals=1.3,
            live_odds_source="inplay",
        )
        snapshot = SimpleNamespace(alerts=[], timestamp="2026-04-04T12:00:00+00:00")
        with mock.patch.object(live_poller, "_HAS_ALERTS", True), \
             mock.patch.object(live_poller, "_HAS_LIVE_ENGINE", True), \
             mock.patch.object(
                 live_poller,
                 "detect_alerts",
                 return_value=[{"type": "live_bet", "message": "Bet now", "mode": "odds_aware", "pricing_source": "inplay"}],
             ):
            enriched = poller._attach_snapshot_alerts(303, snapshot, state)
        self.assertEqual(enriched.alerts, [{"type": "live_bet", "message": "Bet now", "mode": "odds_aware", "pricing_source": "inplay"}])
        self.assertEqual(enriched.live_odds_source, "inplay")
        self.assertEqual(enriched.recommendation_mode, "odds_aware")
        self.assertTrue(enriched.has_priced_recommendation)


if __name__ == "__main__":
    unittest.main()
