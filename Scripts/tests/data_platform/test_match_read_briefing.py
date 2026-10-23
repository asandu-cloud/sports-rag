"""Tests for the editorial layer above the deterministic Match Read compiler."""

from __future__ import annotations


def _result(group: str, projection: dict) -> dict:
    return {
        "fixture": {
            "event_id": "fixture-briefing",
            "league": "EPL",
            "home_team": "Leeds United",
            "away_team": "Newcastle United",
            "kickoff": "2026-09-20T15:00:00Z",
        },
        "market": {"group": group, "key": group, "unit": "test", "participant": None},
        "projection": projection,
        "decision": {"status": "no_bet", "reason": "No current price qualified."},
        "provenance": {
            "pipeline_version": "canonical-market-service.v1",
            "input_snapshot_id": f"briefing:{group}",
            "generated_at": "2026-09-19T10:00:00Z",
        },
        "context": {"data_quality": {"lineup": {"state": "confirmed"}}},
    }


def _draft():
    from data_platform.services.match_read_compiler import compile_match_read

    return compile_match_read([
        _result("goals", {"value": 2.42, "season_component": 2.61, "recent_component": 2.17}),
        _result("corners", {"value": 9.1, "season_component": 9.3, "recent_component": 8.8}),
        _result("btts", {"value": 0.49, "components": {"home_goals": 1.31, "away_goals": 1.11, "yes_probability": 0.49}}),
        _result("moneyline", {"value": 0.42, "components": {"home_probability": 0.42, "draw_probability": 0.29, "away_probability": 0.29}}),
    ])


def test_briefing_enrichment_uses_only_factual_inputs_and_persists_visuals():
    from data_platform.services.match_read_briefing import (
        build_match_read_facts,
        enrich_match_read_draft,
    )

    draft = _draft()
    facts = build_match_read_facts(draft)
    assert "decision" not in str(facts)
    assert "odds" not in str(facts)
    assert facts["metrics"]["goals"]["projected"] == 2.42

    generated = {
        "summary": (
            "Leeds United and Newcastle United project as a fairly even fixture, with 2.42 total goals "
            "and a 1.31 to 1.11 scoring split. The home side has the highest win probability, while 9.1 "
            "projected corners point to steady rather than extreme match volume."
        ),
        "bullets": [
            "2.42 projected goals, split 1.31 for Leeds and 1.11 for Newcastle.",
            "Leeds hold a 42% win chance in a balanced result model.",
            "9.1 projected corners sit close to recent match volume.",
        ],
    }
    enriched, note = enrich_match_read_draft(
        draft,
        {
            "visuals": {
                "home_team_logo": "https://cdn.example.test/leeds.png",
                "away_team_logo": "https://cdn.example.test/newcastle.png",
                "league_logo": "https://cdn.example.test/epl.png",
            },
        },
        generator=lambda supplied_facts: generated,
    )

    assert note is None
    assert enriched.thesis == generated["summary"]
    assert enriched.game_script["briefing"]["source"] == "gpt-5.6"
    assert enriched.game_script["briefing"]["bullets"] == generated["bullets"]
    assert enriched.game_script["visuals"]["league_logo"] == "https://cdn.example.test/epl.png"
    assert enriched.game_script["schema_version"] == "match-read-game-script.v2"


def test_briefing_uses_cached_facts_before_calling_a_generator():
    from data_platform.services.match_read_briefing import enrich_match_read_draft

    first, _ = enrich_match_read_draft(
        _draft(),
        {},
        generator=lambda _facts: {
            "summary": (
                "Leeds United and Newcastle United project as a fairly even fixture, with 2.42 total goals "
                "and a 1.31 to 1.11 scoring split. The home side has the highest win probability, while 9.1 "
                "projected corners point to steady rather than extreme match volume."
            ),
            "bullets": [
                "2.42 projected goals, split 1.31 for Leeds and 1.11 for Newcastle.",
                "Leeds hold a 42% win chance in a balanced result model.",
                "9.1 projected corners sit close to recent match volume.",
            ],
        },
    )
    reused, note = enrich_match_read_draft(
        _draft(),
        {},
        cached_briefing=first.game_script["briefing"],
        generator=lambda _facts: (_ for _ in ()).throw(AssertionError("generator must not run")),
    )

    assert note is None
    assert reused.game_script["briefing"] == first.game_script["briefing"]


def test_briefing_falls_back_without_an_api_response():
    from data_platform.services.match_read_briefing import enrich_match_read_draft

    enriched, note = enrich_match_read_draft(
        _draft(),
        {},
        generator=lambda _facts: {"summary": "too short", "bullets": []},
    )

    assert note is not None
    assert enriched.game_script["briefing"]["source"] == "deterministic_fallback"
    assert len(enriched.game_script["briefing"]["bullets"]) == 3


def _anchored_moneyline():
    """A 1X2 result under the market anchor (A3.7 stage 3): the decision prices on the market's numbers."""
    result = _result("moneyline", {"value": 0.42, "components": {
        "home_probability": 0.42, "draw_probability": 0.29, "away_probability": 0.29}})
    result["context"]["market_anchor"] = {"version": "market-anchor.v1", "scope": "strict",
                                          "market": {"home": 0.30, "draw": 0.26, "away": 0.44},
                                          "spix": {"home": 0.42, "draw": 0.29, "away": 0.29}}
    return result


def test_an_anchored_result_balance_is_the_markets_and_says_so():
    from data_platform.services.match_read_cards import _metrics_card
    from data_platform.services.match_read_metrics import match_metrics

    assert match_metrics([_anchored_moneyline()])["result_probabilities"] == {
        "home": 0.3, "draw": 0.26, "away": 0.44, "source": "market"}
    assert _metrics_card([_anchored_moneyline()])["result_probabilities"] == {"home": 0.3, "draw": 0.26, "away": 0.44}
    # Without the anchor the facts are unchanged, so cached briefings stay valid
    own = _result("moneyline", {"value": 0.42, "components": {
        "home_probability": 0.42, "draw_probability": 0.29, "away_probability": 0.29}})
    assert match_metrics([own])["result_probabilities"] == {"home": 0.42, "draw": 0.29, "away": 0.29}


def test_the_fallback_briefing_attributes_an_anchored_result_balance_to_the_market():
    from data_platform.services.match_read_briefing import enrich_match_read_draft
    from data_platform.services.match_read_compiler import compile_match_read

    draft = compile_match_read([
        _result("goals", {"value": 2.42, "season_component": 2.61, "recent_component": 2.17}),
        _anchored_moneyline(),
    ])
    enriched, _ = enrich_match_read_draft(draft, {}, generator=lambda _facts: {"summary": "too short", "bullets": []})
    summary = enriched.game_script["briefing"]["summary"]
    assert "The market's consensus gives Leeds United a 30% win chance" in summary
    assert "result model" not in summary
