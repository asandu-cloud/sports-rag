"""Regression coverage for the shared settlement and reporting contract."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pytest


@pytest.fixture()
def pred_repo(settings, engine, session_factory):
    from data_platform.repositories.predictions import PredictionRepository
    return PredictionRepository(session_factory=session_factory)


def test_shared_metrics_support_legacy_aliases_and_asian_outcomes():
    from data_platform.tracking_metrics import build_calibration, build_track_record

    rows = [
        {"id": 1, "prediction_date": "2026-09-01", "outcome": "win", "odds": 2.0, "model_prob": 0.55, "confidence": "high", "market": "goals", "league": "EPL"},
        {"id": 2, "prediction_date": "2026-09-01", "outcome": "half_hit", "odds": 2.0, "model_prob": 0.55, "confidence": "high", "market": "goals", "league": "EPL"},
        {"id": 3, "prediction_date": "2026-09-02", "outcome": "loss", "odds": 2.0, "model_prob": 0.55, "confidence": "medium", "market": "cards", "league": "LaLiga"},
        {"id": 4, "prediction_date": "2026-09-02", "outcome": "half_loss", "odds": 2.0, "model_prob": 0.55, "confidence": "medium", "market": "cards", "league": "LaLiga"},
        {"id": 5, "prediction_date": "2026-09-03", "outcome": "push", "odds": 2.0, "model_prob": 0.55, "confidence": "low", "market": "btts", "league": "EPL"},
        {"id": 6, "prediction_date": "2026-09-03", "outcome": "void", "odds": 2.0, "model_prob": 0.55, "confidence": "low", "market": "btts", "league": "EPL"},
    ]

    record = build_track_record(rows, today=date(2026, 9, 3))
    assert {
        "total_graded", "hits", "half_hits", "misses", "half_misses", "pushes",
        "hit_rate", "roi_flat_stake", "by_confidence", "by_market", "by_league",
        "recent_streak", "best_streak", "daily_performance",
    }.issubset(record)
    assert record["total_graded"] == 6
    assert (record["hits"], record["half_hits"], record["misses"], record["half_misses"], record["pushes"]) == (1, 1, 1, 1, 2)
    assert record["hit_rate"] == pytest.approx(0.5)
    assert record["roi_flat_stake"] == pytest.approx(0.0)
    assert record["recent_streak"] == -1
    assert record["best_streak"] == 1

    calibration = build_calibration(rows)
    assert calibration == [{
        "bucket": "55-60%",
        "model_avg": pytest.approx(0.55),
        "actual_rate": pytest.approx(0.5),
        "count": pytest.approx(3.0),
    }]


def test_platform_daily_breakdown_uses_the_discord_recap_contract(pred_repo):
    prediction_id = pred_repo.log(
        home_team="A", away_team="B", league="EPL", market="goals",
        pick="Over 2.5", odds=2.0, confidence="high", model_prob=0.6,
    )
    assert pred_repo.set_outcome(prediction_id, outcome="full_win", actual_result=3)

    breakdown = pred_repo.get_daily_breakdown(target_date=date.today())
    assert breakdown["hits"] == 1
    assert breakdown["misses"] == 0
    assert breakdown["by_market"]["goals"]["hit_rate"] == 1.0
    assert breakdown["notable_hits"][0]["pick"] == "Over 2.5"


def test_platform_resolver_requests_results_with_date_then_league(monkeypatch):
    rag_ingest = Path(__file__).resolve().parents[2] / "rag_ingest"
    if str(rag_ingest) not in sys.path:
        sys.path.insert(0, str(rag_ingest))
    import prediction_tracker as tracker
    import data_platform.compat as compat

    fetched = []
    marked = []
    monkeypatch.setattr(tracker, "_platform_on", lambda: True)
    monkeypatch.setattr(
        compat,
        "platform_get_unresolved_for_grading",
        lambda **_: [{
            "id": 99, "prediction_date": "2026-09-01", "league": "EPL",
            "home_team": "A", "away_team": "B", "market": "goals",
        }],
    )
    monkeypatch.setattr(
        tracker,
        "_fetch_results_from_api",
        lambda prediction_date, league=None: fetched.append((prediction_date, league)) or [{"fixture_id": "1"}],
    )
    monkeypatch.setattr(tracker, "_match_prediction_to_result", lambda *_: {"goals_home": 2, "goals_away": 1})
    monkeypatch.setattr(tracker, "_grade_prediction", lambda *_: ("hit", 3))
    monkeypatch.setattr(
        compat,
        "platform_mark_outcome",
        lambda prediction_id, **kwargs: marked.append((prediction_id, kwargs)) or True,
    )

    result = tracker._maybe_platform_resolve({
        "prediction_date": "2026-09-01", "league": None, "db_path": tracker.DB_PATH,
    })
    assert fetched == [("2026-09-01", "EPL")]
    assert marked == [(99, {"outcome": "hit", "actual_result": 3})]
    assert result["graded"] == 1
    assert result["hit"] == 1


def _published_market_result() -> dict:
    return {
        "schema_version": "market_result.v1",
        "fixture": {
            "event_id": "fixture-999", "league": "EPL",
            "home_team": "Arsenal", "away_team": "Chelsea",
            "kickoff": "2026-09-06T15:00:00Z",
        },
        "market": {"key": "totals", "group": "goals", "unit": "goals", "participant": None},
        "projection": {"value": 3.1, "unit": "goals", "components": {}},
        "decision": {
            "status": "recommended",
            "quote": {"side": "over", "line": 2.5, "odds": 2.0, "bookmaker": "Book", "market_key": "totals"},
            "model_probability": 0.6,
            "implied_probability": 0.5,
            "value_edge": 0.1,
            "expected_value": 0.2,
            "confidence": "high",
            "reason": None,
        },
        "provenance": {
            "pipeline_version": "canonical-market-service.v1",
            "input_snapshot_id": "snapshot:fixture-999:2026-09-05",
            "generated_at": "2026-09-05T09:00:00Z",
            "model_version": "2026.09.1",
            "as_of": "2026-09-05T09:00:00Z",
            "sources": ["API-Football"],
        },
        "evidence": [{"code": "projection", "values": {"total": 3.1}}],
        "context": {},
    }


def test_publication_service_creates_one_auditable_recommendation_and_delivery_cohort(
    settings, engine, session_factory,
):
    from data_platform.models import Prediction, PublishedRecommendation, RecommendationDelivery
    from data_platform.repositories.predictions import PredictionRepository
    from data_platform.repositories.publications import PublicationRepository
    from data_platform.services.publications import PublicationService

    publication_repo = PublicationRepository(session_factory=session_factory)
    service = PublicationService(repo=publication_repo)
    payload = _published_market_result()

    first = service.publish(
        payload, surface="discord", external_reference="message:123",
        published_at="2026-09-05T10:00:00Z",
    )
    retry = service.publish(
        payload, surface="discord", external_reference="message:123",
        published_at="2026-09-05T10:00:00Z",
    )
    website = service.publish(
        payload, surface="website", external_reference="/matches/fixture-999",
        published_at="2026-09-05T10:00:00Z",
    )

    assert first["created"] is True
    assert retry["created"] is False
    assert retry["delivery_created"] is False
    assert website["created"] is False
    assert website["delivery_created"] is True
    assert first["prediction_id"] == website["prediction_id"]

    with session_factory() as session:
        assert session.query(Prediction).count() == 1
        recommendation = session.query(PublishedRecommendation).one()
        assert recommendation.input_snapshot_id == "snapshot:fixture-999:2026-09-05"
        assert recommendation.decision_json == payload
        assert session.query(RecommendationDelivery).count() == 2

    predictions = PredictionRepository(session_factory=session_factory)
    assert predictions.set_outcome(first["prediction_id"], outcome="win", actual_result=3)
    official = predictions.get_track_record(published_only=True)
    assert official["total_graded"] == 1
    assert official["hits"] == 1
    assert official["roi_flat_stake"] == pytest.approx(1.0)


def test_publication_service_rejects_non_recommendations(settings, engine, session_factory):
    from data_platform.repositories.publications import PublicationRepository
    from data_platform.services.publications import PublicationService, PublicationValidationError

    payload = _published_market_result()
    payload["decision"]["status"] = "no_bet"
    payload["decision"]["reason"] = "No qualifying price."
    service = PublicationService(repo=PublicationRepository(session_factory=session_factory))
    with pytest.raises(PublicationValidationError, match="Only a priced recommended decision"):
        service.publish(payload, surface="discord")
