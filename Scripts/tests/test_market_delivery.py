"""Regression tests for Phase 4 structured market delivery adapters."""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Scripts" / "rag_ingest"))
sys.path.insert(0, str(ROOT / "Scripts"))

from core.market_result import (  # noqa: E402
    Decision,
    DecisionStatus,
    FixtureRef,
    MarketRef,
    MarketResult,
    PriceQuote,
    Projection,
    Provenance,
)
from rendering import market_dispatch  # noqa: E402
from rendering.market_results import render_market_results_text  # noqa: E402


def _result(group: str = "goals", *, status: DecisionStatus = DecisionStatus.RECOMMENDED) -> MarketResult:
    unit = "probability" if group in {"btts", "moneyline"} else "goals"
    value = 0.62 if unit == "probability" else 2.9
    if status is DecisionStatus.RECOMMENDED:
        decision = Decision(
            status=status,
            quote=PriceQuote(side="over", line=2.5, odds=1.95, bookmaker="Book"),
            model_probability=0.62,
            implied_probability=1 / 1.95,
            value_edge=0.107,
            expected_value=0.107,
            confidence="medium",
        )
    else:
        decision = Decision(status=status, reason="No price clears the guardrails.")
    return MarketResult(
        fixture=FixtureRef("fixture-1", "EPL", "Arsenal", "Chelsea", "2026-09-01T15:00:00Z"),
        market=MarketRef("totals", group, unit),
        projection=Projection(value, unit),
        decision=decision,
        provenance=Provenance("test", "snapshot-1", "2026-08-21T12:00:00Z"),
    )


class MarketDeliveryTests(unittest.TestCase):
    def test_canonical_dispatch_fetches_and_evaluates_structured_results(self):
        event = {"id": "fixture-1", "home_team": "Arsenal", "away_team": "Chelsea"}
        structured = _result()
        with mock.patch.object(market_dispatch, "fetch_events", return_value=([event], ["provider note"])) as fetch, \
             mock.patch.object(market_dispatch, "filter_events_by_exact_date", return_value=[event]), \
             mock.patch.object(market_dispatch, "enrich_events_for_groups", return_value=([event], ["odds refreshed"])) as enrich, \
             mock.patch.object(market_dispatch, "evaluate_market", return_value=structured) as evaluate:
            results, notes = market_dispatch.evaluate_market_results_sync(
                "EPL", "totals", date(2026, 9, 1)
            )

        fetch.assert_called_once()
        enrich.assert_called_once_with([event], "EPL", {"totals"})
        evaluate.assert_called_once()
        self.assertEqual(evaluate.call_args.args[2], "goals")
        self.assertEqual(results, [structured])
        self.assertEqual(notes, ["provider note", "odds refreshed"])

    def test_legacy_text_adapter_only_formats_the_structured_decision(self):
        text = render_market_results_text([_result(), _result(status=DecisionStatus.NO_BET)])

        self.assertIn("Recommended: Over 2.5 @ 1.95", text)
        self.assertIn("Verdict: No bet", text)
        self.assertIn("Value edge: +10.7%", text)

    def test_public_market_mapping_does_not_claim_specialist_contracts(self):
        self.assertEqual(market_dispatch.canonical_market_name("totals"), "goals")
        self.assertEqual(market_dispatch.canonical_market_name("moneyline"), "moneyline")
        self.assertIsNone(market_dispatch.canonical_market_name("correctscore"))
        self.assertIsNone(market_dispatch.canonical_market_name("teamlines"))

    def test_web_fixture_adapter_exposes_canonical_results_without_recomputing(self):
        # Import here so the market-delivery tests remain useful without the
        # FastAPI stack in lightweight core-only environments.
        from web_app import api

        goals = _result("goals")
        moneyline = _result("moneyline", status=DecisionStatus.NO_BET)
        event = {
            "id": "fixture-1", "home_team": "Arsenal", "away_team": "Chelsea",
            "commence_time": "2026-09-01T15:00:00Z",
        }
        with mock.patch.object(api, "evaluate_event", return_value=[goals, moneyline]) as evaluate:
            prediction = api._build_fixture_prediction(event, "EPL")

        evaluate.assert_called_once()
        self.assertEqual(prediction.goals.projected_total, 2.9)
        self.assertEqual(prediction.goals.recommended_side, "Over")
        self.assertEqual(prediction.goals.status, "recommended")
        self.assertEqual(prediction.moneyline.status, "no_bet")
        self.assertIsNone(prediction.moneyline.recommended)
        self.assertEqual(prediction.market_results[0]["schema_version"], "market_result.v1")

    def test_discord_bet_cards_read_result_fields_directly(self):
        # The direct renderer must not need the legacy text parser to build a
        # recommendation card.  Disable logo/composite I/O for this pure test.
        from Scripts.discord_bot import embeds

        class _Embed:
            def __init__(self, **kwargs):
                self.title = kwargs.get("title")
                self.fields = []

            def set_author(self, **_kwargs):
                pass

            def set_thumbnail(self, **_kwargs):
                pass

            def set_footer(self, **_kwargs):
                pass

            def add_field(self, **kwargs):
                self.fields.append(SimpleNamespace(**kwargs))

        with mock.patch.object(embeds, "get_matchup_file", return_value=None), \
             mock.patch.object(embeds, "_get_league_logo", return_value=None), \
             mock.patch.object(embeds.discord, "Embed", _Embed):
            cards, files = embeds.market_results_to_embeds("Goals", [_result()], league="EPL")

        self.assertEqual(files, [])
        self.assertEqual(len(cards), 2)  # one header plus the fixture card
        fields = {field.name: field.value for field in cards[1].fields}
        self.assertEqual(fields["📋 Pick"], "**Over 2.5**")
        self.assertEqual(fields["📊 Model"], "62.0%")
        self.assertEqual(fields["📈 Edge"], "+10.7%")


if __name__ == "__main__":
    unittest.main()
