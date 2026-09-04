"""Phase 0 Match Read contract tests."""

from __future__ import annotations

from datetime import date, datetime

import pytest


def _market_result(
    *,
    key: str = "totals",
    group: str = "goals",
    side: str = "over",
    line: float = 2.5,
    odds: float = 2.0,
    status: str = "recommended",
    fixture_id: str = "fixture-999",
    league: str = "EPL",
    kickoff: str = "2026-09-06T15:00:00Z",
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
            "event_id": fixture_id,
            "league": league,
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "kickoff": kickoff,
        },
        "market": {"key": key, "group": group, "unit": group, "participant": None},
        "projection": {"value": 3.1, "unit": group, "components": {}},
        "decision": decision,
        "provenance": {
            "pipeline_version": "canonical-market-service.v1",
            "input_snapshot_id": f"snapshot:{fixture_id}:{key}",
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


def test_list_for_matchday_returns_all_versions_and_stages_for_explicit_league(
    match_read_service, settings, engine,
):
    """The raw query must never silently choose an effective card version."""
    results = [_market_result()]
    pre_match_v1 = match_read_service.create(
        canonical_results=results,
        thesis="Initial pre-match read.",
        status="recommended",
        selections=[{"result_index": 0, "role": "core"}],
        stage="pre_match",
        evaluated_at="2026-09-05T08:00:00Z",
    )
    pre_match_v2 = match_read_service.create(
        canonical_results=results,
        thesis="Amended pre-match read.",
        status="recommended",
        selections=[{"result_index": 0, "role": "core"}],
        stage="pre_match",
        evaluated_at="2026-09-05T09:00:00Z",
    )
    confirmed_lineups = match_read_service.create(
        canonical_results=results,
        thesis="Confirmed-lineups read.",
        status="recommended",
        selections=[{"result_index": 0, "role": "core"}],
        stage="confirmed_lineups",
        evaluated_at="2026-09-06T13:00:00Z",
    )

    # Same league, but a different source ISO calendar date.
    match_read_service.create(
        canonical_results=[_market_result(fixture_id="fixture-tomorrow", kickoff="2026-09-07T15:00:00Z")],
        thesis="Tomorrow's read.",
        status="recommended",
        selections=[{"result_index": 0, "role": "core"}],
    )
    # Same date, but a different explicitly requested league.
    match_read_service.create(
        canonical_results=[_market_result(fixture_id="fixture-laliga", league="LaLiga")],
        thesis="Other league read.",
        status="recommended",
        selections=[{"result_index": 0, "role": "core"}],
    )

    reads = match_read_service.list_for_matchday(
        league="EPL",
        target_date=date(2026, 9, 6),
    )

    assert [read["id"] for read in reads] == [
        confirmed_lineups["id"],
        pre_match_v1["id"],
        pre_match_v2["id"],
    ]
    assert [(read["stage"], read["version"]) for read in reads] == [
        ("confirmed_lineups", 1),
        ("pre_match", 1),
        ("pre_match", 2),
    ]


def test_list_for_matchday_uses_the_calendar_date_carried_by_iso_kickoff(
    match_read_service, settings, engine,
):
    # This is September 6 in UTC, but September 7 in the ISO offset stored
    # with the fixture.  The raw historical lookup must preserve that fact
    # until a delivery surface declares a timezone policy of its own.
    offset_kickoff = _market_result(
        fixture_id="fixture-offset",
        kickoff="2026-09-07T00:30:00+02:00",
    )
    match_read_service.create(
        canonical_results=[offset_kickoff],
        thesis="Offset kickoff read.",
        status="recommended",
        selections=[{"result_index": 0, "role": "core"}],
    )
    malformed_kickoff = _market_result(
        fixture_id="fixture-malformed",
        kickoff="2026-09-07-not-a-kickoff",
    )
    match_read_service.create(
        canonical_results=[malformed_kickoff],
        thesis="Malformed legacy kickoff should not break the board.",
        status="recommended",
        selections=[{"result_index": 0, "role": "core"}],
    )

    september_sixth = match_read_service.list_for_matchday(
        league="EPL",
        target_date="2026-09-06",
    )
    september_seventh = match_read_service.list_for_matchday(
        league="EPL",
        target_date="2026-09-07",
    )

    assert september_sixth == []
    assert [read["fixture"]["event_id"] for read in september_seventh] == ["fixture-offset"]


def test_list_for_matchday_requires_an_explicit_iso_date(
    match_read_service, settings, engine,
):
    from data_platform.services.match_reads import MatchReadValidationError

    with pytest.raises(MatchReadValidationError, match="YYYY-MM-DD"):
        match_read_service.list_for_matchday(league="EPL", target_date="2026/09/06")

    with pytest.raises(MatchReadValidationError, match="not a datetime"):
        match_read_service.list_for_matchday(
            league="EPL",
            target_date=datetime(2026, 9, 6, 12, 0),
        )


def test_delivered_read_queries_exclude_shadow_rows_and_dedupe_retries(
    match_read_service, settings, engine,
):
    """Only a delivery adapter may move a read into a public surface query."""
    source = [_market_result(status="no_bet")]
    shadow_only = match_read_service.create(
        canonical_results=source,
        thesis="Shadow-only candidate.",
        status="no_bet",
    )
    discord_read = match_read_service.create(
        canonical_results=source,
        thesis="Public Discord card.",
        status="no_bet",
    )
    website_read = match_read_service.create(
        canonical_results=source,
        thesis="Public website card.",
        status="no_bet",
    )

    # Two independently recorded renders of the same immutable Discord card
    # must still appear once in a public board/history query.
    match_read_service.record_delivery(
        discord_read["id"],
        surface="discord",
        external_reference="discord:1:2:3",
        metadata={"post_type": "match_read_league_hub"},
    )
    match_read_service.record_delivery(
        discord_read["id"],
        surface="discord",
        external_reference="discord:1:2:4",
        metadata={"post_type": "match_read_league_hub"},
    )
    match_read_service.record_delivery(
        website_read["id"],
        surface="website",
        external_reference="/matches/EPL/fixture-999",
    )

    discord_board = match_read_service.list_delivered_for_matchday(
        league="EPL",
        target_date="2026-09-06",
        surface="discord",
    )
    website_board = match_read_service.list_delivered_for_matchday(
        league="EPL",
        target_date="2026-09-06",
        surface="website",
    )
    discord_history = match_read_service.list_delivered_for_fixture(
        "fixture-999",
        surface="discord",
    )

    assert [read["id"] for read in discord_board] == [discord_read["id"]]
    assert [read["id"] for read in website_board] == [website_read["id"]]
    assert [read["id"] for read in discord_history] == [discord_read["id"]]
    assert shadow_only["id"] not in {read["id"] for read in discord_board + website_board}


def test_delivery_lookup_recovers_latest_matching_hub_reference(
    match_read_service, settings, engine,
):
    source = [_market_result(status="no_bet")]
    first = match_read_service.create(
        canonical_results=source,
        thesis="First hub card.",
        status="no_bet",
    )
    second = match_read_service.create(
        canonical_results=source,
        thesis="Second hub card.",
        status="no_bet",
    )
    hub_key = "discord-match-read-hub:1:2:EPL:2026-09-06"
    match_read_service.record_delivery(
        first["id"],
        surface="discord",
        external_reference="discord:1:2:3",
        delivered_at="2026-09-05T08:00:00Z",
        metadata={"post_type": "match_read_league_hub", "hub_key": hub_key},
    )
    match_read_service.record_delivery(
        second["id"],
        surface="discord",
        external_reference="discord:1:2:4",
        delivered_at="2026-09-05T09:00:00Z",
        metadata={"post_type": "match_read_league_hub", "hub_key": hub_key},
    )

    recovered = match_read_service.find_latest_delivery(
        surface="discord",
        metadata={"post_type": "match_read_league_hub", "hub_key": hub_key},
    )

    assert recovered is not None
    assert recovered["match_read_id"] == second["id"]
    assert recovered["external_reference"] == "discord:1:2:4"
