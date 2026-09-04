"""Tests for the deterministic fixture-level Match Read compiler."""

from __future__ import annotations

from copy import deepcopy

import pytest


def _result(
    *,
    group: str,
    key: str = "totals",
    side: str = "over",
    line: float | None = 2.5,
    confidence: str = "medium",
    edge: float = 0.08,
    probability: float = 0.58,
    status: str = "recommended",
    generated_at: str = "2026-09-05T09:00:00Z",
) -> dict:
    quote = None
    if status == "recommended":
        quote = {
            "side": side,
            "line": line,
            "odds": 2.0,
            "bookmaker": "Book",
            "market_key": key,
        }
    return {
        "schema_version": "market_result.v1",
        "fixture": {
            "event_id": "fixture-123",
            "league": "EPL",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "kickoff": "2026-09-06T15:00:00Z",
        },
        "market": {"group": group, "key": key, "unit": group, "participant": None},
        "projection": {"value": 3.0, "unit": group, "components": {}},
        "decision": {
            "status": status,
            "quote": quote,
            "model_probability": probability if status == "recommended" else None,
            "implied_probability": 0.5 if status == "recommended" else None,
            "value_edge": edge if status == "recommended" else None,
            "expected_value": edge * 2 if status == "recommended" else None,
            "confidence": confidence if status == "recommended" else "low",
            "reason": None if status == "recommended" else "Current price did not qualify.",
        },
        "provenance": {
            "pipeline_version": "canonical-market-service.v1",
            "input_snapshot_id": f"snapshot:fixture-123:{group}",
            "generated_at": generated_at,
            "model_version": "2026.09.1",
        },
        "evidence": [],
        "context": {},
    }


def test_compiler_selects_a_core_and_compatible_individual_angles():
    from data_platform.services.match_read_compiler import compile_match_read

    goals = _result(group="goals", side="over", confidence="high", edge=0.12)
    btts = _result(group="btts", key="btts", side="yes", line=None, edge=0.10)
    corners = _result(group="corners", side="under", line=10.5, edge=0.07)
    cards = _result(group="cards", side="over", line=4.5, confidence="low", edge=0.02)

    draft = compile_match_read([goals, btts, corners, cards])

    assert draft.status == "recommended"
    assert draft.selections == (
        {"result_index": 0, "role": "core"},
        {"result_index": 1, "role": "supporting"},
        {"result_index": 2, "role": "supporting"},
    )
    assert draft.game_script["tags"] == ["open_game", "low_corners"]
    assert draft.game_script["alternative_candidates"][0]["result_index"] == 3
    assert "not a combined-bet recommendation" in draft.game_script["selection_relationship"]
    assert "open-game profile" in draft.thesis
    assert draft.create_kwargs()["packages"] == []


def test_compiler_never_mixes_opposite_goal_or_team_side_stories():
    from data_platform.services.match_read_compiler import compile_match_read

    btts_no = _result(group="btts", key="btts", side="no", line=None, confidence="high", edge=0.12)
    goals_over = _result(group="goals", side="over", confidence="medium", edge=0.10)
    home_win = _result(group="moneyline", key="moneyline", side="Arsenal", line=None, edge=0.09)
    away_spread = _result(group="spreads", key="spreads", side="Chelsea", line=0.5, edge=0.08)

    draft = compile_match_read([btts_no, goals_over, home_win, away_spread])

    selected = {item["result_index"] for item in draft.selections}
    assert 0 in selected
    assert 1 not in selected  # BTTS No and Over goals tell opposite stories.
    assert 2 in selected
    assert 3 not in selected  # Home moneyline and away spread conflict.
    assert draft.game_script["tags"] == ["controlled_game", "home_edge"]


@pytest.mark.parametrize(
    ("statuses", "expected_status"),
    [(("no_bet", "unavailable"), "no_bet"), (("unavailable", "unavailable"), "unavailable")],
)
def test_compiler_has_a_clear_non_recommendation_state(statuses, expected_status):
    from data_platform.services.match_read_compiler import compile_match_read

    draft = compile_match_read([
        _result(group="goals", status=statuses[0]),
        _result(group="btts", key="btts", line=None, status=statuses[1]),
    ])

    assert draft.status == expected_status
    assert draft.selections == ()
    assert draft.game_script["tags"] == [expected_status]
    assert "assessed" in draft.thesis if expected_status == "no_bet" else "cannot be assessed" in draft.thesis


def test_persisted_draft_is_idempotent_when_only_evaluation_clock_changes(
    settings, engine, session_factory,
):
    from data_platform.repositories.match_reads import MatchReadRepository
    from data_platform.services.match_read_compiler import compile_match_read, persist_match_read
    from data_platform.services.match_reads import MatchReadService

    service = MatchReadService(repo=MatchReadRepository(session_factory=session_factory))
    first = compile_match_read([_result(group="goals", confidence="high")])
    changed_clock_result = deepcopy(_result(group="goals", confidence="high"))
    changed_clock_result["provenance"]["generated_at"] = "2026-09-05T09:04:00Z"
    second = compile_match_read([changed_clock_result])

    saved = persist_match_read(first, service=service)
    replay = persist_match_read(second, service=service)

    assert saved["created"] is True
    assert replay["created"] is False
    assert replay["id"] == saved["id"]
    assert replay["version"] == 1


def test_visible_match_read_links_each_selected_leaf_to_official_tracking(
    settings, engine, session_factory,
):
    from data_platform.models import PublishedRecommendation, RecommendationDelivery
    from data_platform.repositories.match_reads import MatchReadRepository
    from data_platform.repositories.publications import PublicationRepository
    from data_platform.services.match_read_compiler import compile_match_read, persist_match_read
    from data_platform.services.match_read_delivery import MatchReadDeliveryService
    from data_platform.services.match_reads import MatchReadService
    from data_platform.services.publications import PublicationService

    reads = MatchReadService(repo=MatchReadRepository(session_factory=session_factory))
    publications = PublicationService(repo=PublicationRepository(session_factory=session_factory))
    delivery_service = MatchReadDeliveryService(match_reads=reads, publications=publications)
    draft = compile_match_read([
        _result(group="goals", confidence="high", edge=0.12),
        _result(group="btts", key="btts", side="yes", line=None, edge=0.10),
    ])
    saved = persist_match_read(draft, service=reads)

    first = delivery_service.record_visible(
        saved["id"],
        surface="discord",
        external_reference="discord:1:2:3",
        delivered_at="2026-09-05T10:00:00Z",
        metadata={"post_type": "match_read"},
    )
    retry = delivery_service.record_visible(
        saved["id"],
        surface="discord",
        external_reference="discord:1:2:3",
        delivered_at="2026-09-05T10:00:00Z",
    )

    assert first["match_read_id"] == saved["id"]
    assert first["match_read_delivery_id"] > 0
    assert first["delivery_created"] is True
    assert first["recommendations_created"] == 2
    assert first["recommendation_deliveries_created"] == 2
    assert first["selection_links_created"] == 2
    assert retry["delivery_created"] is False
    assert retry["recommendations_created"] == 0
    assert retry["recommendation_deliveries_created"] == 0
    assert retry["selection_links_created"] == 0
    linked = reads.get(saved["id"])
    assert linked is not None
    assert all(item["published_recommendation_id"] for item in linked["selections"])
    with session_factory() as session:
        assert session.query(PublishedRecommendation).count() == 2
        assert session.query(RecommendationDelivery).count() == 2


def test_visible_no_bet_match_read_is_audited_without_creating_a_recommendation(
    settings, engine, session_factory,
):
    from data_platform.models import PublishedRecommendation
    from data_platform.repositories.match_reads import MatchReadRepository
    from data_platform.repositories.publications import PublicationRepository
    from data_platform.services.match_read_compiler import compile_match_read, persist_match_read
    from data_platform.services.match_read_delivery import MatchReadDeliveryService
    from data_platform.services.match_reads import MatchReadService
    from data_platform.services.publications import PublicationService

    reads = MatchReadService(repo=MatchReadRepository(session_factory=session_factory))
    publications = PublicationService(repo=PublicationRepository(session_factory=session_factory))
    delivered = MatchReadDeliveryService(match_reads=reads, publications=publications)
    saved = persist_match_read(
        compile_match_read([_result(group="goals", status="no_bet")]),
        service=reads,
    )

    outcome = delivered.record_visible(
        saved["id"],
        surface="website",
        external_reference="/matches/fixture-123",
        delivered_at="2026-09-05T10:00:00Z",
    )

    assert outcome["delivery_created"] is True
    assert outcome["recommendations_created"] == 0
    assert outcome["selection_links_created"] == 0
    with session_factory() as session:
        assert session.query(PublishedRecommendation).count() == 0


def test_shared_card_projection_keeps_individual_angles_separate_from_packages(
    settings, engine, session_factory,
):
    from data_platform.repositories.match_reads import MatchReadRepository
    from data_platform.services.match_read_cards import build_match_read_card
    from data_platform.services.match_read_compiler import compile_match_read, persist_match_read
    from data_platform.services.match_reads import MatchReadService

    service = MatchReadService(repo=MatchReadRepository(session_factory=session_factory))
    saved = persist_match_read(
        compile_match_read([
            _result(group="goals", confidence="high", edge=0.12),
            _result(group="btts", key="btts", side="yes", line=None, edge=0.10),
        ]),
        service=service,
    )

    card = build_match_read_card(saved)

    assert card["schema_version"] == "match-read-card.v1"
    assert card["fixture"]["event_id"] == "fixture-123"
    assert [item["role"] for item in card["selections"]] == ["core", "supporting"]
    assert card["packages"] == []
    assert "not a combined-bet recommendation" in card["game_script"]["selection_relationship"]
