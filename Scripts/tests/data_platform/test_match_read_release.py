"""Release-gate tests for public Match Read delivery.

These tests deliberately exercise the gate separately from the compiler: a
valid historical Match Read is still not a valid *current* recommendation.
"""

from __future__ import annotations


def _market_result(
    fixture_id: str,
    *,
    kickoff: str = "2026-09-06T18:00:00Z",
    side: str = "over",
    line: float = 2.5,
) -> dict:
    return {
        "schema_version": "market_result.v1",
        "fixture": {
            "event_id": fixture_id,
            "league": "EPL",
            "home_team": f"Home {fixture_id}",
            "away_team": f"Away {fixture_id}",
            "kickoff": kickoff,
        },
        "market": {"key": "totals", "group": "goals", "unit": "goals", "participant": None},
        "projection": {"value": 2.8, "unit": "goals", "components": {}},
        "decision": {
            "status": "recommended",
            "quote": {
                "side": side,
                "line": line,
                "odds": 1.95,
                "bookmaker": "Book",
                "market_key": "totals",
            },
            "model_probability": 0.60,
            "implied_probability": 1 / 1.95,
            "value_edge": 0.10,
            "expected_value": 0.10,
            "confidence": "high",
            "reason": "Qualified test price.",
        },
        "provenance": {
            "pipeline_version": "canonical-market-service.v1",
            "input_snapshot_id": f"snapshot:{fixture_id}:{side}:{line}",
            "generated_at": "2026-09-06T12:00:00Z",
            "model_version": "test.v1",
        },
        "evidence": [],
        "context": {},
    }


def _services(session_factory):
    from data_platform.repositories.match_reads import MatchReadRepository
    from data_platform.repositories.publications import PublicationRepository
    from data_platform.services.match_read_delivery import MatchReadDeliveryService
    from data_platform.services.match_read_release import MatchReadReleaseService
    from data_platform.services.match_reads import MatchReadService
    from data_platform.services.publications import PublicationService

    reads = MatchReadService(repo=MatchReadRepository(session_factory=session_factory))
    publications = PublicationService(repo=PublicationRepository(session_factory=session_factory))
    delivery = MatchReadDeliveryService(match_reads=reads, publications=publications)
    return reads, MatchReadReleaseService(match_reads=reads, deliveries=delivery)


def _save(reads, result: dict, *, thesis: str, stage: str = "pre_match", evaluated_at: str = "2026-09-06T12:00:00Z") -> dict:
    return reads.create(
        canonical_results=[result],
        thesis=thesis,
        status="recommended",
        selections=[{"result_index": 0, "role": "core"}],
        game_script={"tags": ["test"]},
        stage=stage,
        evaluated_at=evaluated_at,
    )


def test_release_promotes_only_the_current_effective_fixture_read(settings, engine, session_factory):
    reads, release = _services(session_factory)
    source = _market_result("fixture-effective")
    pre = _save(reads, source, thesis="Initial pre-match read.")
    confirmed = _save(
        reads,
        source,
        thesis="Confirmed-lineups amendment.",
        stage="confirmed_lineups",
        evaluated_at="2026-09-06T12:15:00Z",
    )
    started = _save(
        reads,
        _market_result("fixture-started", kickoff="2026-09-06T12:00:00Z"),
        thesis="Already underway.",
        evaluated_at="2026-09-06T11:50:00Z",
    )
    stale = _save(
        reads,
        _market_result("fixture-stale", kickoff="2026-09-06T19:00:00Z"),
        thesis="Old price.",
        evaluated_at="2026-09-06T09:00:00Z",
    )

    result = release.release_matchday_to_website(
        league="EPL",
        target_date="2026-09-06",
        now="2026-09-06T12:30:00Z",
    )

    assert [item["match_read_id"] for item in result["released"]] == [confirmed["id"]]
    assert pre["id"] not in [item["match_read_id"] for item in result["released"]]
    assert {item["match_read_id"] for item in result["skipped"]} == {started["id"], stale["id"]}
    public = reads.list_delivered_for_matchday(
        league="EPL",
        target_date="2026-09-06",
        surface="website",
    )
    assert [read["id"] for read in public] == [confirmed["id"]]
    assert public[0]["selections"][0]["published_recommendation_id"] is not None

    # A retry is a no-op, rather than duplicating a website card or a tracked
    # recommendation.
    retry = release.release_matchday_to_website(
        league="EPL",
        target_date="2026-09-06",
        now="2026-09-06T12:31:00Z",
    )
    assert retry["released"][0]["delivery_created"] is False
    assert retry["released"][0]["recommendations_created"] == 0


def test_release_dry_run_has_no_delivery_side_effect(settings, engine, session_factory):
    reads, release = _services(session_factory)
    saved = _save(reads, _market_result("fixture-dry-run"), thesis="Dry-run read.")

    result = release.release_matchday_to_website(
        league="EPL",
        target_date="2026-09-06",
        now="2026-09-06T12:30:00Z",
        dry_run=True,
    )

    assert result["released"] == [{
        "fixture_id": "fixture-dry-run",
        "match_read_id": saved["id"],
        "would_release": True,
    }]
    assert reads.list_delivered_for_matchday(
        league="EPL",
        target_date="2026-09-06",
        surface="website",
    ) == []

