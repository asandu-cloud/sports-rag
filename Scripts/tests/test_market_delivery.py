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
    probability_source,
    published_projection,
    published_result_probabilities,
    stored_probability_source,
)
from rendering import market_dispatch  # noqa: E402
from rendering.market_results import projection_label, render_market_results_text  # noqa: E402


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


SPIX_1X2 = {"home": 0.42, "draw": 0.29, "away": 0.29}
MARKET_1X2 = {"home": 0.30, "draw": 0.26, "away": 0.44}


def _moneyline(*, anchored: bool, recommended: bool = False) -> MarketResult:
    """A 1X2 result; anchored as A3.7 stage 3 records it: the market's numbers beside Spix's own."""
    if recommended:
        decision = Decision(status=DecisionStatus.RECOMMENDED, quote=PriceQuote(side="away", odds=2.40, bookmaker="Book"),
                            model_probability=0.44 if anchored else 0.29, implied_probability=1 / 2.40,
                            value_edge=0.023, expected_value=0.056, confidence="low")
    else:
        decision = Decision(status=DecisionStatus.NO_BET, reason="No side clears the moneyline betting thresholds.")
    context = ({"market_anchor": {"version": "market-anchor.v1", "scope": "strict", "method": "power_devig_median",
                                  "books": 10, "market": MARKET_1X2, "spix": SPIX_1X2}} if anchored else {})
    return MarketResult(
        fixture=FixtureRef("fixture-1", "EPL", "Arsenal", "Chelsea", "2026-09-01T15:00:00Z"),
        market=MarketRef("moneyline", "moneyline", "probability"),
        projection=Projection(0.42, "probability", components={f"{k}_probability": v for k, v in SPIX_1X2.items()}),
        decision=decision,
        provenance=Provenance("test", "snapshot-1", "2026-08-21T12:00:00Z"),
        context=context,
    )


def _with_anchor(result: MarketResult, anchor: dict) -> MarketResult:
    return MarketResult.from_dict({**result.to_dict(), "context": {"market_anchor": anchor}})


M1_FIGURES = {"home_goals": 1.49, "away_goals": 1.04, "total_goals": 2.53, "goal_difference": 0.45, "btts_yes": 0.523}
STAGE_4 = {"scope": "goals", "goal_model": {"home": 1.49, "away": 1.04, "error": 0.0}, "goal_figures": M1_FIGURES}


def _spread() -> MarketResult:
    result = _result("spreads")
    return MarketResult.from_dict({**result.to_dict(), "projection": {"value": 0.3, "unit": "goals"},
                                   "decision": {**result.decision.to_dict(),
                                                "quote": {"side": "Arsenal", "line": -0.5, "odds": 1.9}}})


class MarketAnchorPublicationTests(unittest.TestCase):
    """Under the market anchor (D21) a decision prices on the market's numbers; the projection keeps Spix's own."""

    def test_an_anchored_1x2_publishes_the_markets_probabilities(self):
        self.assertEqual(published_result_probabilities(_moneyline(anchored=True)), (MARKET_1X2, "market"))
        self.assertEqual(published_result_probabilities(_moneyline(anchored=False)), (SPIX_1X2, "model"))
        self.assertEqual(published_result_probabilities(_result("goals")), (None, "model"))

    def test_the_probability_source_follows_each_scope_of_the_anchor(self):
        self.assertEqual(probability_source(_moneyline(anchored=True)), "market")
        self.assertEqual(probability_source(_moneyline(anchored=False)), "model")
        strict = {"scope": "strict", "lines": {"2.5": {"market_over": 0.57, "books": 8, "spix_over": 0.50}}}
        self.assertEqual(probability_source(_with_anchor(_result("goals"), strict)), "market")  # over 2.5
        other_line = MarketResult.from_dict({**_result("goals").to_dict(), "context": {"market_anchor": strict},
                                             "decision": {**_result("goals").decision.to_dict(),
                                                          "quote": {"side": "over", "line": 3.5, "odds": 2.1}}})
        self.assertEqual(probability_source(other_line), "model")  # strict scope anchors 2.5 only
        fitted = {"scope": "goals", "goal_model": {"home": 1.5, "away": 1.1, "error": 0.0}}
        self.assertEqual(probability_source(_with_anchor(_result("btts"), fitted)), "market")  # stage 4: M1
        unfitted = {"scope": "goals", "goal_model": None}
        self.assertEqual(probability_source(_with_anchor(_result("btts"), unfitted)), "model")

    def test_text_and_discord_cards_show_the_anchored_1x2_as_the_markets(self):
        self.assertEqual(projection_label(_moneyline(anchored=True)), "1X2 (market): 30.0% / 26.0% / 44.0%")
        self.assertEqual(projection_label(_moneyline(anchored=False)), "1X2: 42.0% / 29.0% / 29.0%")
        text = render_market_results_text([_moneyline(anchored=True, recommended=True)])
        self.assertIn("Market probability: 44.0%", text)
        from Scripts.discord_bot import embeds

        class _Embed:
            def __init__(self, **kwargs):
                self.fields = []

            def set_author(self, **_kwargs):
                pass

            set_thumbnail = set_footer = set_author

            def add_field(self, **kwargs):
                self.fields.append(SimpleNamespace(**kwargs))

        with mock.patch.object(embeds, "get_matchup_file", return_value=None), \
             mock.patch.object(embeds, "_get_league_logo", return_value=None), \
             mock.patch.object(embeds.discord, "Embed", _Embed):
            cards, _ = embeds.market_results_to_embeds("1X2", [_moneyline(anchored=True, recommended=True)], league="EPL")
        fields = {field.name: field.value for field in cards[1].fields}
        self.assertEqual(fields["📊 Projection"], "1X2 (market): 30.0% / 26.0% / 44.0%")
        self.assertEqual(fields["📊 Market"], "44.0%")
        self.assertNotIn("📊 Model", fields)

    def test_the_web_moneyline_carries_the_anchored_probabilities_and_their_source(self):
        from web_app import api

        anchored = api._moneyline_from_result(_moneyline(anchored=True))
        self.assertEqual((anchored.home_prob, anchored.draw_prob, anchored.away_prob), (0.30, 0.26, 0.44))
        self.assertEqual(anchored.probability_source, "market")
        own = api._moneyline_from_result(_moneyline(anchored=False))
        self.assertEqual((own.home_prob, own.probability_source), (0.42, "model"))


class GoalModelPublicationTests(unittest.TestCase):
    """Under stage 4 (D29) the goal markets price from M1; surfaces show M1's figures, Spix's stay in the projection."""

    def test_the_goal_markets_publish_m1s_figures(self):
        self.assertEqual(published_projection(_with_anchor(_result("goals"), STAGE_4)), (2.53, "market"))
        self.assertEqual(published_projection(_with_anchor(_result("btts"), STAGE_4)), (0.523, "market"))
        self.assertEqual(published_projection(_with_anchor(_spread(), STAGE_4)), (0.45, "market"))

    def test_without_m1s_figures_the_projection_is_spixs(self):
        self.assertEqual(published_projection(_result("goals")), (2.9, "model"))
        strict = {"scope": "strict", "lines": {"2.5": {"market_over": 0.57, "books": 8, "spix_over": 0.50}}}
        self.assertEqual(published_projection(_with_anchor(_result("goals"), strict)), (2.9, "model"))
        unfitted = {"scope": "goals", "goal_model": None}
        self.assertEqual(published_projection(_with_anchor(_result("btts"), unfitted)), (0.62, "model"))
        self.assertEqual(published_projection(_with_anchor(_moneyline(anchored=False), STAGE_4)), (0.42, "model"))

    def test_a_stored_result_reports_the_same_probability_source(self):
        strict = {"scope": "strict", "lines": {"2.5": {"market_over": 0.57, "books": 8, "spix_over": 0.50}}}
        for result in (_moneyline(anchored=True), _moneyline(anchored=False), _result("goals"),
                       _with_anchor(_result("goals"), strict), _with_anchor(_result("btts"), STAGE_4),
                       _with_anchor(_spread(), STAGE_4), _with_anchor(_result("btts"), {"goal_model": None})):
            self.assertEqual(stored_probability_source(result.to_dict()), probability_source(result))
        self.assertEqual(stored_probability_source({}), "model")

    def test_text_and_discord_cards_show_m1s_figures_as_the_markets(self):
        goals, btts, spread = (_with_anchor(r, STAGE_4) for r in (_result("goals"), _result("btts"), _spread()))
        self.assertEqual(projection_label(goals), "Projected total (market): 2.53 goals")
        self.assertEqual(projection_label(btts), "BTTS Yes (market): 52.3%")
        self.assertEqual(projection_label(spread), "Goal difference (market): +0.45")
        self.assertEqual(projection_label(_result("goals")), "Projected total: 2.90 goals")
        self.assertIn("Market probability: 62.0%", render_market_results_text([goals]))
        from Scripts.discord_bot import embeds

        class _Embed:
            def __init__(self, **kwargs):
                self.fields = []

            def set_author(self, **_kwargs):
                pass

            set_thumbnail = set_footer = set_author

            def add_field(self, **kwargs):
                self.fields.append(SimpleNamespace(**kwargs))

        with mock.patch.object(embeds, "get_matchup_file", return_value=None), \
             mock.patch.object(embeds, "_get_league_logo", return_value=None), \
             mock.patch.object(embeds.discord, "Embed", _Embed):
            cards, _ = embeds.market_results_to_embeds("Goals", [goals, btts, spread, _result("btts")], league="EPL")
        shown = [{field.name: field.value for field in card.fields}["📊 Projection"] for card in cards[1:]]
        self.assertEqual(shown, ["Projected total (market): 2.53 goals", "BTTS Yes (market): 52.3%",
                                 "Goal difference (market): +0.45", "62.0%"])

    def test_the_web_goal_markets_carry_m1s_figures_and_their_source(self):
        from web_app import api

        totals = api._totals_from_result(_with_anchor(_result("goals"), STAGE_4))
        self.assertEqual((totals.projected_total, totals.projection_source, totals.probability_source),
                         (2.53, "market", "market"))
        btts = api._btts_from_result(_with_anchor(_result("btts"), STAGE_4))
        self.assertEqual((btts.probability, btts.probability_source), (0.523, "market"))
        spread = api._spread_from_result(_with_anchor(_spread(), STAGE_4))
        self.assertEqual((spread.projected_diff, spread.projection_source, spread.probability_source),
                         (0.45, "market", "market"))
        own = api._totals_from_result(_result("goals"))
        self.assertEqual((own.projected_total, own.projection_source, own.probability_source), (2.9, "model", "model"))


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
