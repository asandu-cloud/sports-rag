from __future__ import annotations

import sys
import os
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from core.market_result import DecisionStatus
from core import market_service


def _event() -> dict:
    return {
        "id": "fixture-123",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "commence_time": "2026-05-01T15:00:00Z",
        "bookmakers": [
            {
                "title": "Book",
                "markets": [
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": 2.5, "price": 2.05},
                            {"name": "Under", "point": 2.5, "price": 1.80},
                        ],
                    },
                    {
                        "key": "btts",
                        "outcomes": [
                            {"name": "Yes", "price": 1.95},
                            {"name": "No", "price": 1.95},
                        ],
                    },
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Arsenal", "price": 2.10},
                            {"name": "Draw", "price": 3.60},
                            {"name": "Chelsea", "price": 3.80},
                        ],
                    },
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Arsenal", "point": -0.5, "price": 2.05},
                            {"name": "Chelsea", "point": 0.5, "price": 1.85},
                        ],
                    },
                ],
            }
        ],
    }


class MarketServiceTests(unittest.TestCase):
    def test_goals_total_returns_a_reproducible_structured_recommendation(self):
        with mock.patch.object(market_service, "projected_total_goals", return_value=(3.2, 3.0, 3.4)):
            result = market_service.evaluate_market(
                _event(), "EPL", "goals",
                input_snapshot_id="recorded:fixture-123",
                generated_at="2026-08-21T12:00:00Z",
            )

        self.assertEqual(result.fixture.event_id, "fixture-123")
        self.assertEqual(result.market.key, "totals")
        self.assertEqual(result.market.group, "goals")
        self.assertEqual(result.projection.value, 3.2)
        self.assertEqual(result.decision.status, DecisionStatus.RECOMMENDED)
        self.assertEqual(result.decision.quote.side, "over")
        self.assertEqual(result.decision.quote.line, 2.5)
        self.assertEqual(result.provenance.input_snapshot_id, "recorded:fixture-123")
        self.assertEqual(result.provenance.generated_at, "2026-08-21T12:00:00Z")

    def test_missing_profile_data_is_unavailable_without_running_selection(self):
        with mock.patch.object(market_service, "projected_total_corners", return_value=(None, None, None)):
            result = market_service.evaluate_market(
                _event(), "EPL", "corners", generated_at="2026-08-21T12:00:00Z"
            )

        self.assertEqual(result.decision.status, DecisionStatus.UNAVAILABLE)
        self.assertIn("Insufficient profile data", result.decision.reason)
        self.assertIsNone(result.decision.quote)

    def test_result_carries_auditable_input_quality(self):
        home_audit = {
            "team": "Arsenal",
            "temporal_status": "fixture_rows_strictly_before_target_date",
            "current_season_matches": 1,
            "prior_weight": 0.8889,
        }
        away_audit = {
            "team": "Chelsea",
            "temporal_status": "fixture_rows_strictly_before_target_date",
            "current_season_matches": 1,
            "prior_weight": 0.8889,
        }
        with mock.patch.object(market_service, "projected_total_goals", return_value=(3.2, 3.0, 3.4)), \
             mock.patch.object(
                 market_service,
                 "get_team_profile_context",
                 side_effect=[({}, home_audit), ({}, away_audit)],
             ):
            result = market_service.evaluate_market(
                _event(), "EPL", "goals", generated_at="2026-08-21T12:00:00Z"
            )

        quality = result.context["data_quality"]
        self.assertEqual(quality["schema_version"], "prediction-input-quality.v1")
        self.assertEqual(quality["profiles"]["home"], home_audit)
        self.assertEqual(quality["profiles"]["away"], away_audit)
        self.assertEqual(quality["odds"]["bookmaker_count"], 1)
        self.assertGreater(quality["odds"]["priced_option_count"], 0)
        self.assertEqual(quality["odds"]["freshness"], "unknown")

    def test_one_current_match_suppresses_an_otherwise_qualified_total(self):
        audit = {
            "temporal_status": "fixture_rows_strictly_before_target_date",
            "current_season_matches": 1,
            "effective_sample_size": 9.0,
        }
        with mock.patch.object(market_service, "projected_total_goals", return_value=(4.0, 3.8, 4.1)), \
             mock.patch.object(market_service, "get_team_profile_context", side_effect=[({}, audit), ({}, audit)]), \
             mock.patch.object(market_service, "get_blended_variance", return_value=4.0):
            result = market_service.evaluate_market(
                _event(), "EPL", "goals", generated_at="2026-08-21T12:00:00Z"
            )

        self.assertEqual(result.decision.status, DecisionStatus.NO_BET)
        self.assertIn("Data-quality guardrail", result.decision.reason)
        guardrail = result.context["data_quality"]["recommendation_guardrail"]
        self.assertEqual(guardrail["status"], "suppressed")
        self.assertGreater(result.projection.variance, result.projection.value)

    def test_shadow_mode_withholds_a_canonical_recommendation_at_the_service_boundary(self):
        with mock.patch.object(market_service, "projected_total_goals", return_value=(3.2, 3.0, 3.4)), \
             mock.patch.object(market_service, "_total_variance", return_value=(None, {"source": "test"})), \
             mock.patch.object(market_service, "_apply_quality_guardrails", side_effect=lambda decision, *_args, **_kwargs: decision), \
             mock.patch.dict(os.environ, {"PREDICTION_RELEASE_MODE": "shadow"}, clear=False):
            result = market_service.evaluate_market(
                _event(), "EPL", "goals", generated_at="2026-08-21T12:00:00Z"
            )

        self.assertEqual(result.decision.status, DecisionStatus.NO_BET)
        self.assertIn("Shadow mode", result.decision.reason)
        self.assertEqual(result.context["release_policy"]["mode"], "shadow")
        self.assertEqual(result.context["release_candidate"]["status"], "recommended")

    def test_btts_uses_the_shared_selector_and_keeps_goal_split_as_evidence(self):
        with mock.patch.object(market_service, "projected_btts_prob", return_value=(0.64, 1.8, 1.4, 1.7, 1.3)):
            result = market_service.evaluate_market(
                _event(), "EPL", "btts", generated_at="2026-08-21T12:00:00Z"
            )

        self.assertEqual(result.market.key, "btts")
        self.assertEqual(result.projection.value, 0.64)
        self.assertEqual(result.decision.status, DecisionStatus.RECOMMENDED)
        self.assertEqual(result.decision.quote.side, "yes")
        self.assertEqual(result.evidence[0].code, "btts_goal_split")
        self.assertEqual(result.evidence[0].values["home_goals"], 1.8)

    def test_moneyline_without_prices_is_unavailable_not_a_no_bet(self):
        no_price_event = _event()
        no_price_event["bookmakers"] = []
        with mock.patch.object(market_service, "projected_moneyline_probs", return_value=(0.55, 0.24, 0.21, 1.8, 1.0)):
            result = market_service.evaluate_market(
                no_price_event, "EPL", "moneyline", generated_at="2026-08-21T12:00:00Z"
            )

        self.assertEqual(result.decision.status, DecisionStatus.UNAVAILABLE)
        self.assertIn("No market prices", result.decision.reason)

    def test_event_evaluation_preserves_requested_market_order(self):
        with mock.patch.object(market_service, "projected_total_goals", return_value=(3.2, 3.0, 3.4)), \
             mock.patch.object(market_service, "projected_btts_prob", return_value=(0.64, 1.8, 1.4, 1.7, 1.3)):
            results = market_service.evaluate_event(
                _event(), "EPL", markets=("btts", "goals"), generated_at="2026-08-21T12:00:00Z"
            )

        self.assertEqual([result.market.key for result in results], ["btts", "totals"])

    def test_unknown_market_is_rejected_before_data_access(self):
        with self.assertRaisesRegex(ValueError, "Unsupported canonical market"):
            market_service.evaluate_market(_event(), "EPL", "correct_score")


if __name__ == "__main__":
    unittest.main()
