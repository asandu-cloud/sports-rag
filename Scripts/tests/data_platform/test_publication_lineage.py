from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import hashlib
from pathlib import Path

import pytest

from data_platform.models import Prediction, PublishedRecommendation, RecommendationDelivery, MatchReadDelivery
from data_platform.repositories.match_reads import MatchReadRepository
from data_platform.repositories.predictions import PredictionRepository
from data_platform.repositories.publications import PublicationRepository
from data_platform.services.match_reads import MatchReadService
from data_platform.services.match_read_delivery import MatchReadDeliveryService
from data_platform.services.publications import PublicationService, _hash_key
from data_platform.publication_identity import tracking_identity
from Scripts.tests.data_platform.test_tracking_contract import _published_market_result


@pytest.fixture
def services(settings, engine, session_factory):
    reads = MatchReadService(MatchReadRepository(session_factory))
    publications = PublicationService(PublicationRepository(session_factory))
    return reads, publications, MatchReadDeliveryService(match_reads=reads, publications=publications), PredictionRepository(session_factory)


def save(reads, results, *, stage="pre_match", status="recommended"):
    return reads.create(canonical_results=results, thesis="Evidence-based test read", status=status,
                        stage=stage, selections=[{"result_index": i, "role": "core" if i == 0 else "supporting"}
                                                 for i in range(len(results))] if status == "recommended" else [])


def visible(delivery, read, *, surface="website", time="2026-09-05T10:00:00Z"):
    return delivery.record_visible(read["id"], surface=surface, external_reference=f"{surface}:fixture-999", delivered_at=time)


def test_first_card_preserves_multiple_picks_and_does_not_promote_settled_amendment(services):
    reads, pubs, delivery, predictions = services
    a = _published_market_result()
    a["provenance"]["system_version"] = "test-system:a"
    a["decision"]["quote"]["period"] = "regulation_time"
    b = deepcopy(a)
    b["market"].update(key="totals_corners_over_under", group="corners", unit="corners")
    b["projection"]["unit"] = "corners"
    b["decision"]["quote"].update(market_key="totals_corners_over_under", line=9.5)
    first = save(reads, [a, b])
    assert visible(delivery, first)["recommendations_created"] == 2
    assert visible(delivery, first, surface="discord")["recommendations_created"] == 0
    amendment = deepcopy(a)
    amendment["decision"]["quote"]["odds"] = 2.3
    amendment["provenance"]["input_snapshot_id"] = "lineup-snapshot"
    later = save(reads, [amendment], stage="confirmed_lineups")
    visible(delivery, later, time="2026-09-06T14:00:00Z")
    rows = predictions.get_recent(days=365, published_only=True, publication_scope="all")
    assert len(rows) == 3
    initial = [r for r in rows if r["publication_role"] == "initial"]
    changed = next(r for r in rows if r["publication_role"] == "amendment")
    assert len(initial) == 2
    assert changed["match_read_versions"][0]["stage"] == "confirmed_lineups"
    assert changed["fixture_date"] == "2026-09-06"
    assert initial[0]["publication_date"] == "2026-09-05"
    assert initial[0]["system_version"] == "test-system:a"
    assert initial[0]["market_period"] == "regulation_time"
    # A settled later winner cannot fill the gap left by unresolved originals.
    predictions.set_outcome(changed["id"], outcome="win")
    assert predictions.get_track_record(published_only=True)["total_graded"] == 0
    assert predictions.get_track_record(published_only=True, publication_scope="amendments")["hits"] == 1
    for row in initial:
        predictions.set_outcome(row["id"], outcome="loss")
    record = predictions.get_track_record(published_only=True)
    assert record["total_graded"] == 2 and record["roi_flat_stake"] == -1
    daily = predictions.get_daily_breakdown(target_date=date(2026, 9, 5), published_only=True)
    assert daily["total"] == 2 and daily["misses"] == 2
    assert len(predictions.get_recent(days=365, published_only=True, limit=1)) == 1


def test_no_bet_initial_card_does_not_turn_later_addition_into_original(services):
    reads, pubs, delivery, predictions = services
    result = _published_market_result()
    no_bet = deepcopy(result)
    no_bet["decision"].update(status="no_bet", quote=None, reason="No qualifying price")
    visible(delivery, save(reads, [no_bet], status="no_bet"))
    visible(delivery, save(reads, [result]), time="2026-09-05T11:00:00Z")
    rows = predictions.get_recent(days=365, published_only=True, publication_scope="all")
    assert rows[0]["publication_role"] == "amendment"
    assert predictions.get_recent(days=365, published_only=True) == []


def test_delivery_failure_rolls_back_all_publication_links(services, session_factory, monkeypatch):
    reads, pubs, delivery, predictions = services
    first = _published_market_result()
    second = deepcopy(first)
    second["decision"]["quote"]["line"] = 3.5
    read = save(reads, [first, second])
    original = PublicationService.publish
    count = 0
    def fail_second(self, *args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("injected failure")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(PublicationService, "publish", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        visible(delivery, read)
    with session_factory() as session:
        for table in (Prediction, PublishedRecommendation, RecommendationDelivery, MatchReadDelivery):
            assert session.query(table).count() == 0
    assert all(s["published_recommendation_id"] is None for s in reads.get(read["id"])["selections"])
    monkeypatch.setattr(PublicationService, "publish", original)
    assert visible(delivery, read)["recommendations_created"] == 2


def test_competing_surfaces_and_restart_create_one_prediction(services, session_factory):
    reads, pubs, delivery, predictions = services
    read = save(reads, [_published_market_result()])
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda surface: visible(delivery, read, surface=surface), ["website", "discord"]))
    assert sum(r["recommendations_created"] for r in results) == 1
    restarted = MatchReadDeliveryService(publications=PublicationService(PublicationRepository(session_factory)))
    assert visible(restarted, read)["recommendations_created"] == 0
    with session_factory() as session:
        assert session.query(Prediction).count() == 1
        assert session.query(RecommendationDelivery).count() == 2


@pytest.mark.parametrize("change", ["system", "model", "probability", "market", "period"])
def test_semantic_changes_do_not_collide_when_snapshot_id_stays_the_same(services, change):
    reads, pubs, delivery, predictions = services
    first = _published_market_result()
    changed = deepcopy(first)
    if change == "system": changed["provenance"]["system_version"] = "new-system"
    if change == "model": changed["provenance"]["model_version"] = "new-model"
    if change == "probability": changed["decision"]["model_probability"] = 0.7
    if change == "market": changed["decision"]["quote"]["market_key"] = "totals_yellow_cards"
    if change == "period": changed["decision"]["quote"]["period"] = "first_half"
    a = pubs.publish(first, surface="website")
    b = pubs.publish(changed, surface="website")
    assert a["prediction_id"] != b["prediction_id"]
    changed["provenance"]["generated_at"] = "2026-09-05T12:00:00Z"
    assert pubs.publish(changed, surface="discord")["prediction_id"] == b["prediction_id"]


def test_old_key_retry_preserves_original_release_and_unknown_provenance(services, session_factory):
    reads, pubs, delivery, predictions = services
    payload = _published_market_result()
    first = pubs.publish(payload, surface="website", external_reference="page")
    # Reproduce the exact v1 keys used before this change without rewriting the
    # fixture/price/evidence. This is a test database only.
    legacy = _hash_key({"event_id": payload["fixture"]["event_id"], "market_key": "totals", "market_group": "goals",
                        "quote": payload["decision"]["quote"], "input_snapshot_id": payload["provenance"]["input_snapshot_id"],
                        "pipeline_version": payload["provenance"]["pipeline_version"]})
    with session_factory() as session:
        rec = session.get(PublishedRecommendation, first["recommendation_id"])
        rec.recommendation_key = legacy
        release = session.get(RecommendationDelivery, first["delivery_id"])
        release.delivery_key = _hash_key({"recommendation_key": legacy, "surface": "website", "external_reference": "page"})
    retried = pubs.publish(payload, surface="website", external_reference="page")
    assert not retried["created"] and not retried["delivery_created"]
    assert retried["prediction_id"] == first["prediction_id"]
    row = predictions.get_recent(days=365, published_only=True, publication_scope="all")[0]
    assert row["publication_role"] == "unclassified"
    assert row["system_version"] is None and row["market_period"] is None


def test_exact_card_key_and_unknown_quote_times_are_preserved():
    result = _published_market_result()
    result["market"]["group"] = "cards"
    result["decision"]["quote"]["market_key"] = "totals_yellow_cards"
    identity = tracking_identity(result)
    assert identity["selection"]["market_key"] == "totals_yellow_cards"
    assert identity["card_definition_status"] == "requires_bookmaker_rule"
    assert identity["quote_time"] is None and identity["quote_captured_at"] is None


def test_read_only_tracking_audit(settings, session_factory, services):
    from pathlib import Path
    from Scripts.ops.prediction_tracking_audit import audit
    reads, pubs, delivery, predictions = services
    visible(delivery, save(reads, [_published_market_result()]))
    path = Path(settings.database_url.removeprefix("sqlite:///"))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    report = audit(path)
    assert report["published"]["unresolved"] == 1
    assert report["published"]["publication_vs_kickoff_day_differences"] == 1
    assert report["visible_selections_missing_publication_link"] == 0
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_official_website_and_discord_reads_use_canonical_store_even_if_flag_off(services, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "rag_ingest"))
    import prediction_tracker as tracker
    import data_platform.compat as compat
    from fastapi.testclient import TestClient
    from Scripts.web_app.track_record_api import create_standalone_app
    reads, pubs, delivery, predictions = services
    payload = _published_market_result()
    payload["provenance"]["system_version"] = "test-system:api"
    visible(delivery, save(reads, [payload]))
    prediction = predictions.get_recent(days=365, published_only=True)[0]
    predictions.set_outcome(prediction["id"], outcome="win")
    monkeypatch.setattr(tracker, "_platform_on", lambda: False)
    # Bind the compatibility service to this test database without relying on
    # cached service instances from a previous test.
    from data_platform.services.predictions import PredictionService
    import data_platform.compat.predictions_shim as shim
    monkeypatch.setattr(shim, "get_prediction_service", lambda: PredictionService(predictions))
    with TestClient(create_standalone_app()) as client:
        response = client.get("/api/track-record")
        assert response.status_code == 200
        assert response.json()["total_graded"] == tracker.get_track_record(published_only=True)["total_graded"] == 1
        recent = client.get("/api/track-record/recent").json()
        assert recent[0]["system_version"] == "test-system:api"
        assert recent[0]["input_snapshot_id"] == payload["provenance"]["input_snapshot_id"]
        assert recent[0]["publication_role"] == "initial"
        assert client.get("/api/track-record?publication_scope=amendments").json()["total_graded"] == 0
        assert client.get("/api/track-record?publication_scope=best_price").status_code == 422
    def unavailable(**kwargs):
        raise RuntimeError("canonical store unavailable")
    monkeypatch.setattr(compat, "platform_get_track_record", unavailable)
    with pytest.raises(RuntimeError, match="canonical store unavailable"):
        tracker.get_track_record(published_only=True)


def test_shadow_and_unit_legs_do_not_enter_published_cohort(services):
    reads, pubs, delivery, predictions = services
    for source in ("canonical_shadow", "unit_bets", "standalone"):
        predictions.log(home_team="A", away_team="B", league="EPL", market="goals", pick="Over 2.5", source=source)
    assert predictions.get_recent(published_only=True, publication_scope="all") == []
    assert {r["tracking_cohort"] for r in predictions.get_recent()} == {"shadow", "unit_bet_legs", "legacy_standalone"}


def test_publication_timezones_choose_actual_earliest_delivery(services):
    reads, pubs, delivery, predictions = services
    payload = _published_market_result()
    first = save(reads, [payload])
    visible(delivery, first, time="2026-09-05T10:00:00+03:00")  # 07:00 UTC
    changed = deepcopy(payload)
    changed["decision"]["quote"]["odds"] = 2.2
    second = save(reads, [changed])
    visible(delivery, second, time="2026-09-05T08:00:00Z")
    initial = predictions.get_recent(days=365, published_only=True)
    assert len(initial) == 1 and initial[0]["odds"] == 2.0


def test_contract_round_trip_keeps_system_and_period_but_does_not_invent_old_ones(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "rag_ingest"))
    from core.market_result import MarketResult
    payload = _published_market_result()
    payload["provenance"]["system_version"] = "system-test.v1"
    payload["decision"]["quote"]["period"] = "regulation_time"
    roundtrip = MarketResult.from_dict(payload).to_dict()
    assert roundtrip["provenance"]["system_version"] == "system-test.v1"
    assert roundtrip["decision"]["quote"]["period"] == "regulation_time"
    legacy = MarketResult.from_dict(_published_market_result()).to_dict()
    assert "system_version" not in legacy["provenance"]
    assert "period" not in legacy["decision"]["quote"]
