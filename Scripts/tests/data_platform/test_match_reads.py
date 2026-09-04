"""Phase 0 Match Read contract tests."""

from __future__ import annotations

import pytest


def _market_result(
    *,
    key: str = "totals",
    group: str = "goals",
    side: str = "over",
    line: float = 2.5,
    odds: float = 2.0,
    status: str = "recommended",
) -> dict:
    decision = {
        "status": status,
        "quote": {
            "side": side,
            "line": line,
            "odds": odds,
            "bookmaker": "Book",
            "market_key": key,
        } if status == "recommended" else None,
        "model_probability": 0.60 if status == "recommended" else None,
        "implied_probability": 1 / odds if status == "recommended" else None,
        "value_edge": 0.10 if status == "recommended" else None,
        "expected_value": 0.20 if status == "recommended" else None,
        "confidence": "high" if status == "recommended" else "low",
        "reason": None if status == "recommended" else "No qualifying market price.",
    }
    return {
        "schema_version": "market_result.v1",
        "fixture": {
            "event_id": "fixture-999",
            "league": "EPL",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "kickoff": "2026-09-06T15:00:00Z",
        },
        "market": {"key": key, "group": group, "unit": group, "participant": None},
        "projection": {"value": 3.1, "unit": group, "components": {}},
        "decision": decision,
        "provenance": {
            "pipeline_version": "canonical-market-service.v1",
            "input_snapshot_id": f"snapshot:fixture-999:{key}",
            "generated_at": "2026-09-05T09:00:00Z",
            "model_version": "2026.09.1",
        },
        "evidence": [],
        "context": {},
    }


@pytest.fixture()
def match_read_service(settings, engine, session_factory):
    from data_platform.repositories.match_reads import MatchReadRepository
    from data_platform.services.match_reads import MatchReadService
    return MatchReadService(repo=MatchReadRepository(session_factory=session_factory))


def test_match_read_is_versioned_idempotent_and_supports_a_real_quote_build(
    match_read_service, settings, engine, session_factory,
):
    goals = _market_result()
    btts = _market_result(key="btts", group="btts", side="yes", line=None, odds=1.8)
    requested = dict(
        canonical_results=[goals, btts],
        thesis="An open game is the central read; both selections express that same view.",
        status="recommended",
        selections=[
            {"result_index": 0, "role": "core"},
            {"result_index": 1, "role": "supporting"},
        ],
        packages=[{
            "kind": "same_game_build",
            "label": "Open-game build",
            "rationale": "Only use this exact one-book quote.",
            "selection_positions": [1, 2],
            "quote": {"odds": 3.4, "bookmaker": "Book"},
        }],
        game_script={"tags": ["open_game", "both_teams_score"], "confidence": "medium"},
        evaluated_at="2026-09-05T09:00:00Z",
    )
    first = match_read_service.create(**requested)
    retry = match_read_service.create(**requested)

    assert first["created"] is True
    assert retry["created"] is False
    assert retry["id"] == first["id"]
    assert first["version"] == 1
    assert [item["role"] for item in first["selections"]] == ["core", "supporting"]
    package = first["packages"][0]
    assert package["status"] == "coherent_build"
    assert package["selection_positions"] == [1, 2]
    assert package["joint_model_probability"] is None
    assert package["expected_value"] is None

    amended = match_read_service.create(
        **{**requested, "thesis": "Confirmed context strengthens the open-game read."}
    )
    assert amended["created"] is True
    assert amended["version"] == 2
    assert amended["supersedes_id"] == first["id"]

    # Leaf tracking remains separate, and an unchanged published leaf may be
    # linked to both versions without turning either Match Read mutable.
    from data_platform.repositories.publications import PublicationRepository
    from data_platform.services.publications import PublicationService
    published = PublicationService(repo=PublicationRepository(session_factory=session_factory)).publish(
        goals,
        surface="discord",
        external_reference="discord:1:2:3",
        published_at="2026-09-05T10:00:00Z",
    )
    assert match_read_service.link_published_recommendation(
        first["id"], selection_position=1,
        published_recommendation_id=published["recommendation_id"],
    )
    assert match_read_service.link_published_recommendation(
        amended["id"], selection_position=1,
        published_recommendation_id=published["recommendation_id"],
    )

    assert match_read_service.settle_package(
        package["id"], outcome="win", profit_units=2.4,
    )
    settled = match_read_service.get(first["id"])
    assert settled is not None
    assert settled["packages"][0]["outcome"] == "hit"
    assert settled["packages"][0]["profit_units"] == pytest.approx(2.4)
    assert settled["selections"][0]["published_recommendation_id"] == published["recommendation_id"]


def test_no_bet_match_read_can_be_delivered_without_creating_a_bet(
    match_read_service, settings, engine,
):
    no_bet = _market_result(status="no_bet")
    read = match_read_service.create(
        canonical_results=[no_bet],
        thesis="The fixture was assessed, but the available price does not qualify.",
        status="no_bet",
        evaluated_at="2026-09-05T09:00:00Z",
    )
    delivery = match_read_service.record_delivery(
        read["id"],
        surface="website",
        external_reference="/match/EPL/fixture-999",
        delivered_at="2026-09-05T09:05:00Z",
        metadata={"state": "no_bet"},
    )

    assert read["status"] == "no_bet"
    assert read["selections"] == []
    assert read["packages"] == []
    assert delivery["created"] is True


def test_match_read_rejects_unverified_or_overfull_same_game_construction(
    match_read_service, settings, engine,
):
    from data_platform.services.match_reads import MatchReadValidationError

    results = [
        _market_result(),
        _market_result(key="btts", group="btts", side="yes", line=None),
    ]
    base = dict(
        canonical_results=results,
        thesis="A coherent game-state thesis.",
        status="recommended",
        selections=[
            {"result_index": 0, "role": "core"},
            {"result_index": 1, "role": "supporting"},
        ],
    )

    with pytest.raises(MatchReadValidationError, match="package.quote.bookmaker"):
        match_read_service.create(**{
            **base,
            "packages": [{
                "kind": "same_game_build",
                "selection_positions": [1, 2],
                "quote": {"odds": 3.2},
            }],
        })

    with pytest.raises(MatchReadValidationError, match="joint probability or expected value"):
        match_read_service.create(**{
            **base,
            "packages": [{
                "kind": "same_game_build",
                "selection_positions": [1, 2],
                "quote": {"odds": 3.2, "bookmaker": "Book"},
                "joint_model_probability": 0.45,
            }],
        })

    three = [_market_result(), _market_result(key="btts", group="btts", side="yes", line=None), _market_result(key="cards", group="cards", line=4.5)]
    with pytest.raises(MatchReadValidationError, match="exactly one core"):
        match_read_service.create(
            canonical_results=three,
            thesis="Too many cores should be rejected.",
            status="recommended",
            selections=[
                {"result_index": 0, "role": "core"},
                {"result_index": 1, "role": "core"},
                {"result_index": 2, "role": "supporting"},
            ],
        )
