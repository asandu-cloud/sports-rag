from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from core.line_selection import select_best_total_recommendation
from core.market_result import (
    MARKET_RESULT_SCHEMA_VERSION,
    Decision,
    DecisionStatus,
    EvidenceItem,
    FixtureRef,
    MarketRef,
    MarketResult,
    MarketResultError,
    PriceQuote,
    Projection,
    Provenance,
    decision_from_selector,
)


def _provenance() -> Provenance:
    return Provenance(
        pipeline_version="legacy-shadow",
        input_snapshot_id="fixture-123:odds-456:profiles-789",
        generated_at="2026-08-20T12:00:00Z",
        model_version="core-projections-2026-05-11",
        as_of="2026-04-30T12:00:00Z",
        sources=("profile_snapshot", "odds_snapshot"),
    )


class MarketResultContractTests(unittest.TestCase):
    def _recommended_result(self) -> MarketResult:
        return MarketResult(
            fixture=FixtureRef(
                event_id="fixture-123",
                league="EPL",
                home_team="Arsenal",
                away_team="Chelsea",
                kickoff="2026-05-01T15:00:00Z",
            ),
            market=MarketRef(key="totals", group="goals", unit="goals"),
            projection=Projection(
                value=3.2,
                unit="goals",
                season_component=3.0,
                recent_component=3.4,
                variance=1.2,
                components={"league_adjustment": 1.0},
            ),
            decision=Decision(
                status=DecisionStatus.RECOMMENDED,
                quote=PriceQuote(
                    side="over", line=2.5, odds=2.05,
                    bookmaker="Book", market_key="totals",
                ),
                model_probability=0.66,
                implied_probability=0.49,
                value_edge=0.17,
                expected_value=0.35,
                confidence="high",
            ),
            provenance=_provenance(),
            evidence=(EvidenceItem("season_recent_agree", {"difference": 0.4}),),
            context={"rivalry": False, "referee_source": "profile"},
        )

    def test_round_trip_preserves_the_versioned_contract(self):
        result = self._recommended_result()

        restored = MarketResult.from_json(result.to_json())

        self.assertEqual(restored, result)
        self.assertEqual(restored.schema_version, MARKET_RESULT_SCHEMA_VERSION)
        self.assertTrue(restored.is_recommended)
        self.assertEqual(restored.decision.quote.side, "over")

    def test_recommended_decision_requires_a_priced_quote_and_probability(self):
        with self.assertRaises(MarketResultError):
            Decision(status=DecisionStatus.RECOMMENDED, model_probability=0.60)

        with self.assertRaises(MarketResultError):
            Decision(
                status=DecisionStatus.RECOMMENDED,
                quote=PriceQuote(side="over", line=2.5, odds=1.95),
            )

    def test_no_bet_and_unavailable_require_explicit_reasons(self):
        with self.assertRaises(MarketResultError):
            Decision(status=DecisionStatus.NO_BET)

        no_bet = Decision(
            status=DecisionStatus.NO_BET,
            reason="Model edge is below the configured threshold.",
        )
        unavailable = Decision(
            status=DecisionStatus.UNAVAILABLE,
            reason="No priced totals lines are available.",
        )

        self.assertEqual(no_bet.status, DecisionStatus.NO_BET)
        self.assertEqual(unavailable.status, DecisionStatus.UNAVAILABLE)

    def test_market_and_projection_units_must_match(self):
        with self.assertRaises(MarketResultError):
            MarketResult(
                fixture=FixtureRef("fixture-123", "EPL", "Arsenal", "Chelsea"),
                market=MarketRef("totals", "goals", "goals"),
                projection=Projection(3.2, "corners"),
                decision=Decision(
                    status=DecisionStatus.UNAVAILABLE,
                    reason="Invalid baseline fixture.",
                ),
                provenance=_provenance(),
            )

    def test_selector_adapter_distinguishes_recommendation_no_bet_and_unavailable(self):
        recommended_selector = select_best_total_recommendation(
            [
                {"side": "over", "point": 2.5, "odds": 2.05, "bookmaker": "Book", "market_key": "totals"},
                {"side": "under", "point": 2.5, "odds": 1.80, "bookmaker": "Book", "market_key": "totals"},
            ],
            projection_total=3.2,
        )
        no_bet_selector = select_best_total_recommendation(
            [
                {"side": "over", "point": 2.5, "odds": 1.91, "bookmaker": "Book", "market_key": "totals"},
                {"side": "under", "point": 2.5, "odds": 1.91, "bookmaker": "Book", "market_key": "totals"},
            ],
            projection_total=2.52,
        )
        unavailable_selector = select_best_total_recommendation([], projection_total=2.8)

        recommended = decision_from_selector(recommended_selector)
        no_bet = decision_from_selector(no_bet_selector)
        unavailable = decision_from_selector(unavailable_selector)

        self.assertEqual(recommended.status, DecisionStatus.RECOMMENDED)
        self.assertEqual(recommended.quote.side, "over")
        self.assertIsNotNone(recommended.model_probability)
        self.assertEqual(no_bet.status, DecisionStatus.NO_BET)
        self.assertIsNotNone(no_bet.reason)
        self.assertEqual(unavailable.status, DecisionStatus.UNAVAILABLE)
        self.assertIsNotNone(unavailable.reason)

    def test_unknown_schema_is_rejected(self):
        payload = self._recommended_result().to_dict()
        payload["schema_version"] = "market_result.v0"

        with self.assertRaises(MarketResultError):
            MarketResult.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
