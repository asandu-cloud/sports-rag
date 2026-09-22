"""Tests for the scheduler-safe Match Read worker.

All dispatcher calls are injected: these tests prove scheduling, persistence,
freshness, and release behaviour without touching API-Football.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone


def _settings():
    from data_platform.config import load_match_read_worker_settings

    return replace(
        load_match_read_worker_settings(),
        leagues=("EPL",),
        outlook_hours=96,
        final_window_minutes=120,
        refresh_minutes=10,
        early_refresh_minutes=360,
        lineup_window_minutes=100,
        lineup_refresh_minutes=10,
        max_age_minutes=120,
        lease_seconds=300,
    )


def test_default_worker_scope_includes_all_public_match_read_leagues(monkeypatch):
    """A scheduler with no league override must promote the approved leagues."""
    monkeypatch.delenv("MATCH_READ_WORKER_LEAGUES", raising=False)
    from data_platform.config import (
        DEFAULT_MATCH_READ_WORKER_LEAGUES,
        load_match_read_worker_settings,
    )

    expected = (
        "EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1", "UCL", "UEL", "UECL",
        "Championship", "SuperLig", "Eredivisie", "PrimeiraLiga", "BelgianProLeague",
    )
    assert DEFAULT_MATCH_READ_WORKER_LEAGUES == expected
    assert load_match_read_worker_settings().leagues == expected


def test_default_worker_horizon_covers_four_days_with_a_slow_early_refresh(monkeypatch):
    monkeypatch.delenv("MATCH_READ_OUTLOOK_HOURS", raising=False)
    monkeypatch.delenv("MATCH_READ_EARLY_REFRESH_MINUTES", raising=False)
    from data_platform.config import load_match_read_worker_settings

    settings = load_match_read_worker_settings()
    assert settings.outlook_hours == 96
    assert settings.early_refresh_minutes == 360


def _fixture(*, fixture_id: str = "9001", kickoff: datetime) -> object:
    from data_platform.services.match_read_cycle import ScheduledMatchReadFixture

    return ScheduledMatchReadFixture(
        fixture_id=fixture_id,
        league="EPL",
        kickoff_utc=kickoff,
        target_date=kickoff.date(),
    )


def _market_result(fixture_id: str, kickoff: datetime, *, generated_at: str) -> dict:
    return {
        "schema_version": "market_result.v1",
        "fixture": {
            "event_id": fixture_id,
            "league": "EPL",
            "home_team": "Home",
            "away_team": "Away",
            "kickoff": kickoff.isoformat().replace("+00:00", "Z"),
        },
        "market": {"key": "totals", "group": "goals", "unit": "goals", "participant": None},
        "projection": {"value": 2.8, "unit": "goals", "components": {}},
        "decision": {
            "status": "recommended",
            "quote": {
                "side": "over", "line": 2.5, "odds": 1.95,
                "bookmaker": "Book", "market_key": "totals",
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
            "input_snapshot_id": f"snapshot:{fixture_id}",
            "generated_at": generated_at,
            "model_version": "test.v1",
        },
        "evidence": [],
        "context": {},
    }


def _persisting_dispatcher(service, kickoff: datetime, *, generated_at: str, pending_lineups: bool = False):
    """Build a dispatcher substitute that writes legitimate immutable reads."""
    from data_platform.services.match_read_cycle import _event_identity
    from Scripts.rag_ingest.rendering.match_read_dispatch import (
        FixtureMatchReadGeneration,
        MatchReadGenerationRun,
    )

    calls = []

    def dispatch(league, target_date, *, stage, event_filter, force_refresh, **_kwargs):
        calls.append({"league": league, "date": target_date, "stage": stage, "force_refresh": force_refresh})
        event = {"id": "9001"}
        if not event_filter(event):
            return MatchReadGenerationRun(league, target_date, stage, (), ())
        if stage == "confirmed_lineups" and pending_lineups:
            item = FixtureMatchReadGeneration(
                event_id="9001",
                fixture_label="Home vs Away",
                canonical_results=(),
                draft=None,
                error="Home vs Away: confirmed lineups are not available; no confirmed-lineups Match Read was generated.",
            )
            return MatchReadGenerationRun(league, target_date, stage, (item,), ())
        saved = service.create(
            canonical_results=[_market_result("9001", kickoff, generated_at=generated_at)],
            thesis=f"{stage} test read.",
            status="recommended",
            selections=[{"result_index": 0, "role": "core"}],
            game_script={"tags": ["test"]},
            stage=stage,
            evaluated_at=generated_at,
        )
        item = FixtureMatchReadGeneration(
            event_id="9001",
            fixture_label="Home vs Away",
            canonical_results=(),
            draft=None,
            record=saved,
        )
        return MatchReadGenerationRun(league, target_date, stage, (item,), ())

    return dispatch, calls


def test_plan_uses_a_fresh_pre_match_fallback_and_lineup_refresh(settings, engine, session_factory):
    from data_platform.services.match_read_cycle import build_match_read_cycle_plan

    now = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)
    plan = build_match_read_cycle_plan(
        [_fixture(kickoff=now + timedelta(minutes=90))],
        latest_observations={},
        now=now,
        settings=_settings(),
        mode="website",
    )

    assert [(job.stage, job.force_refresh) for job in plan.jobs] == [
        ("confirmed_lineups", True),
        ("pre_match", True),
    ]
    assert all(job.release_fixture_ids == ("9001",) for job in plan.jobs)


def test_plan_combines_same_slate_work_into_one_fresh_provider_fetch(settings, engine, session_factory):
    from data_platform.services.match_read_cycle import ScheduledMatchReadFixture, build_match_read_cycle_plan

    now = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)
    near = _fixture(fixture_id="near", kickoff=now + timedelta(minutes=110))
    later_kickoff = now + timedelta(minutes=300)
    later = ScheduledMatchReadFixture(
        fixture_id="later",
        league="EPL",
        kickoff_utc=later_kickoff,
        target_date=later_kickoff.date(),
    )
    plan = build_match_read_cycle_plan(
        [near, later],
        latest_observations={},
        now=now,
        settings=_settings(),
        mode="shadow",
    )

    assert len(plan.jobs) == 1
    job = plan.jobs[0]
    assert job.stage == "pre_match"
    assert job.fixture_ids == ("later", "near")
    assert job.force_refresh is True


def test_website_plan_releases_a_preliminary_read_three_days_before_kickoff(settings, engine, session_factory):
    from data_platform.services.match_read_cycle import build_match_read_cycle_plan

    now = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)
    plan = build_match_read_cycle_plan(
        [_fixture(kickoff=now + timedelta(days=3))],
        latest_observations={},
        now=now,
        settings=_settings(),
        mode="website",
    )

    assert len(plan.jobs) == 1
    job = plan.jobs[0]
    assert job.stage == "pre_match"
    assert job.fixture_ids == ("9001",)
    assert job.release_fixture_ids == ("9001",)
    assert job.preliminary_fixture_ids == ("9001",)
    assert job.force_refresh is False
    assert job.reasons == ("website early pre-match refresh",)


def test_plan_retries_a_failed_lineup_window_pre_match_fallback_quickly(settings, engine, session_factory):
    from data_platform.services.match_read_cycle import build_match_read_cycle_plan

    now = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)
    plan = build_match_read_cycle_plan(
        [_fixture(kickoff=now + timedelta(minutes=90))],
        latest_observations={
            ("9001", "pre_match"): {
                "status": "failed",
                "checked_at": (now - timedelta(minutes=11)).isoformat(),
            },
        },
        now=now,
        settings=_settings(),
        mode="website",
    )

    assert {job.stage for job in plan.jobs} == {"pre_match", "confirmed_lineups"}


def test_shadow_cycle_records_immutable_reads_and_observations_without_public_delivery(
    settings, engine, session_factory,
):
    from data_platform.models import MatchReadDelivery, MatchReadObservation, SyncRun
    from data_platform.repositories.match_reads import MatchReadRepository
    from data_platform.services.match_read_cycle import MatchReadCycleService
    from data_platform.services.match_reads import MatchReadService

    now = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)
    kickoff = now + timedelta(minutes=90)
    reads = MatchReadService(repo=MatchReadRepository(session_factory=session_factory))
    dispatcher, calls = _persisting_dispatcher(
        reads, kickoff, generated_at="2026-09-06T11:00:00Z",
    )
    worker = MatchReadCycleService(
        settings=_settings(),
        session_factory=session_factory,
        fixture_loader=lambda **_kwargs: [_fixture(kickoff=kickoff)],
        dispatcher=dispatcher,
    )

    report = worker.run_once(now=now, mode="shadow")

    assert report.errors == []
    assert report.run_id is not None
    assert {call["stage"] for call in calls} == {"pre_match", "confirmed_lineups"}
    assert all(call["force_refresh"] is True for call in calls)
    with session_factory() as session:
        assert session.query(MatchReadObservation).count() == 2
        assert {row.status for row in session.query(MatchReadObservation).all()} == {"verified"}
        assert session.query(MatchReadDelivery).count() == 0
        assert session.query(SyncRun).filter_by(run_kind="match_read_cycle", status="completed").count() == 1


def test_website_cycle_releases_a_worker_rechecked_unchanged_card_with_tracking(
    settings, engine, session_factory,
):
    from data_platform.models import MatchReadDelivery, MatchReadObservation, PublishedRecommendation
    from data_platform.repositories.match_reads import MatchReadRepository
    from data_platform.services.match_read_cycle import MatchReadCycleService
    from data_platform.services.match_reads import MatchReadService

    now = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)
    kickoff = now + timedelta(minutes=110)
    reads = MatchReadService(repo=MatchReadRepository(session_factory=session_factory))
    # Deliberately stale immutable provenance: the fresh worker observation,
    # not a fake revision, is what makes the release eligible.
    dispatcher, _calls = _persisting_dispatcher(
        reads, kickoff, generated_at="2026-09-06T06:00:00Z",
    )
    worker = MatchReadCycleService(
        settings=_settings(),
        session_factory=session_factory,
        fixture_loader=lambda **_kwargs: [_fixture(kickoff=kickoff)],
        dispatcher=dispatcher,
    )

    report = worker.run_once(now=now, mode="website")

    assert report.errors == []
    assert len(report.releases) == 1
    assert len(report.releases[0]["released"]) == 1
    # The public read path uses the same verified observation rather than
    # hiding an unchanged card solely because its immutable provenance clock
    # predates the most recent worker check.
    from web_app.routers import match_reads as match_reads_router
    public_reads = reads.list_delivered_for_matchday(
        league="EPL", target_date=date(2026, 9, 6), surface="website",
    )
    assert [card["fixture"]["event_id"] for card in match_reads_router._effective_cards(
        public_reads, now=now + timedelta(minutes=30),
    )] == ["9001"]
    with session_factory() as session:
        observation = session.query(MatchReadObservation).one()
        assert observation.status == "verified"
        assert session.query(MatchReadDelivery).filter_by(surface="website").count() == 1
        assert session.query(PublishedRecommendation).count() == 1


def test_pending_lineups_are_recorded_for_retry_without_a_public_amendment(
    settings, engine, session_factory,
):
    from data_platform.models import MatchReadDelivery, MatchReadObservation
    from data_platform.repositories.match_reads import MatchReadRepository
    from data_platform.services.match_read_cycle import MatchReadCycleService
    from data_platform.services.match_reads import MatchReadService

    now = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)
    kickoff = now + timedelta(minutes=90)
    reads = MatchReadService(repo=MatchReadRepository(session_factory=session_factory))
    dispatcher, _calls = _persisting_dispatcher(
        reads, kickoff, generated_at="2026-09-06T11:55:00Z", pending_lineups=True,
    )
    worker = MatchReadCycleService(
        settings=_settings(),
        session_factory=session_factory,
        fixture_loader=lambda **_kwargs: [_fixture(kickoff=kickoff)],
        dispatcher=dispatcher,
    )

    report = worker.run_once(now=now, mode="website")

    assert report.errors == []
    with session_factory() as session:
        observations = {(row.stage, row.status) for row in session.query(MatchReadObservation).all()}
        assert observations == {("pre_match", "verified"), ("confirmed_lineups", "lineups_pending")}
        # The fresh pre-match fallback can be public; no unverified XI card
        # was made public as an amendment.
        assert session.query(MatchReadDelivery).filter_by(surface="website").count() == 1


def test_worker_lease_makes_an_overlapping_cycle_a_harmless_skip(settings, engine, session_factory):
    from data_platform.repositories.match_read_cycle import MatchReadCycleRepository
    from data_platform.services.match_read_cycle import MATCH_READ_CYCLE_LEASE_KEY, MatchReadCycleService

    now = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)
    repo = MatchReadCycleRepository(session_factory=session_factory)
    assert repo.acquire_lease(
        lease_key=MATCH_READ_CYCLE_LEASE_KEY,
        owner_id="other-worker",
        ttl_seconds=300,
        now=now,
    )
    worker = MatchReadCycleService(
        settings=_settings(),
        cycle_repository=repo,
        session_factory=session_factory,
        fixture_loader=lambda **_kwargs: [],
        dispatcher=lambda *_args, **_kwargs: None,
    )

    report = worker.run_once(now=now, mode="shadow")

    assert report.errors == []
    assert report.lease_acquired is False
    assert report.skipped_reason


def test_delayed_generation_uses_live_release_clock(settings, engine, session_factory):
    from data_platform.repositories.match_reads import MatchReadRepository
    from data_platform.services.match_reads import MatchReadService
    from data_platform.services.match_read_cycle import MatchReadCycleService

    now = datetime(2026, 9, 6, 12, tzinfo=timezone.utc)
    current = [now]
    kickoff = now + timedelta(minutes=110)
    reads = MatchReadService(repo=MatchReadRepository(session_factory=session_factory))
    dispatcher, _ = _persisting_dispatcher(reads, kickoff, generated_at=(now + timedelta(minutes=6)).isoformat())

    def delayed(*args, **kwargs):
        current[0] += timedelta(minutes=6)
        return dispatcher(*args, **kwargs)

    worker = MatchReadCycleService(
        settings=replace(_settings(), lease_seconds=1200), session_factory=session_factory,
        fixture_loader=lambda **_: [_fixture(kickoff=kickoff)], dispatcher=delayed, clock=lambda: current[0],
    )
    report = worker.run_once(now=now, mode="website")
    assert not report.errors
    assert len(report.releases[0]["released"]) == 1  # Previously rejected as "in the future".


def test_fixture_publish_is_immediate_and_kickoff_ordered(settings, engine, session_factory, monkeypatch):
    from types import SimpleNamespace
    from data_platform.services.match_read_cycle import MatchReadCycleService, _dispatch_summary

    now = datetime(2026, 9, 6, 12, tzinfo=timezone.utc)
    current = [now]
    events = []
    checked = []
    fixtures = [_fixture(fixture_id="9001", kickoff=now + timedelta(hours=6)),
                _fixture(fixture_id="9002", kickoff=now + timedelta(hours=3))]

    def dispatch(job, *, slate_cache, checked_at, **_):
        events.append(("dispatch", job.fixture_ids[0]))
        checked.append(checked_at)
        slate_cache[(job.league, job.target_date, job.force_refresh)] = object()
        current[0] += timedelta(seconds=20)
        return _dispatch_summary(job, run=None), set(job.fixture_ids)

    def release(**kwargs):
        events.append(("release", kwargs["fixture_ids"][0]))
        assert kwargs["now"] == current[0]
        return {"released": kwargs["fixture_ids"], "skipped": []}

    worker = MatchReadCycleService(
        settings=_settings(), session_factory=session_factory, fixture_loader=lambda **_: fixtures,
        clock=lambda: current[0], release_service=SimpleNamespace(release_matchday_to_website=release),
    )
    monkeypatch.setattr(worker, "_dispatch_job", dispatch)
    report = worker.run_once(now=now, mode="website")
    assert not report.errors
    assert events == [("dispatch", "9002"), ("release", "9002"), ("dispatch", "9001"), ("release", "9001")]
    assert checked == [now, now]  # Reused prices retain the original observation time.


def test_budget_yields_remaining_work_for_next_tick(settings, engine, session_factory, monkeypatch):
    from data_platform.models import SyncRun
    from data_platform.services.match_read_cycle import MatchReadCycleService, _dispatch_summary

    now = datetime(2026, 9, 6, 12, tzinfo=timezone.utc)
    current = [now]
    fixtures = [_fixture(fixture_id=str(i), kickoff=now + timedelta(hours=3)) for i in range(3)]
    worker = MatchReadCycleService(
        settings=replace(_settings(), max_cycle_seconds=60), session_factory=session_factory,
        fixture_loader=lambda **_: fixtures, clock=lambda: current[0],
    )

    def dispatch(job, **_):
        current[0] += timedelta(seconds=61)
        return _dispatch_summary(job, run=None), set()

    monkeypatch.setattr(worker, "_dispatch_job", dispatch)
    report = worker.run_once(now=now, mode="shadow")
    assert not report.errors
    assert len(report.dispatches) == 1
    assert report.deferred_fixture_stages == 2
    with session_factory() as session:
        assert session.get(SyncRun, report.run_id).status == "deferred"


def test_fixture_timeout_does_not_abort_other_fixtures(settings, engine, session_factory, monkeypatch):
    import signal
    from data_platform.services.match_read_cycle import MatchReadCycleService, _dispatch_summary
    from data_platform.models import MatchReadObservation, SyncRun

    now = datetime(2026, 9, 6, 12, tzinfo=timezone.utc)
    worker = MatchReadCycleService(
        settings=_settings(), session_factory=session_factory,
        fixture_loader=lambda **_: [_fixture(fixture_id=str(i), kickoff=now + timedelta(hours=3)) for i in range(2)],
    )

    def dispatch(job, **_):
        if job.fixture_ids == ("0",):
            from core.prediction_metrics import measure_generation, timed_call

            @measure_generation
            def timed_dispatch(*args):
                timed_call("profile_context", signal.raise_signal, signal.SIGALRM)

            timed_dispatch("EPL", now.date())
        return _dispatch_summary(job, run=None), set()

    monkeypatch.setattr(worker, "_dispatch_job", dispatch)
    report = worker.run_once(now=now, mode="shadow")
    assert len(report.dispatches) == 2
    assert len(report.errors) == 1
    assert "exceeded" in report.errors[0]
    with session_factory() as session:
        assert session.query(MatchReadObservation).one().status == "failed"
        performance = session.get(SyncRun, report.run_id).stats["fixture_performance"]
        assert performance[0]["performance"]["counts"]["profile_context.calls"] == 1
        assert performance[0]["performance"]["interrupted_by"] == "DispatchTimeout"


def test_expired_lease_cannot_publish_after_resume(settings, engine, session_factory, monkeypatch):
    from data_platform.services.match_read_cycle import MatchReadCycleService, _dispatch_summary
    from data_platform.models import MatchReadObservation

    now = datetime(2026, 9, 6, 12, tzinfo=timezone.utc)
    current = [now]
    worker = MatchReadCycleService(
        settings=_settings(), session_factory=session_factory, clock=lambda: current[0],
        fixture_loader=lambda **_: [_fixture(kickoff=now + timedelta(hours=3))],
    )

    def dispatch(job, **_):
        current[0] += timedelta(hours=1)
        return _dispatch_summary(job, run=None), set(job.fixture_ids)

    monkeypatch.setattr(worker, "_dispatch_job", dispatch)
    report = worker.run_once(now=now, mode="website")
    assert report.releases == []
    assert report.deferred_fixture_stages == 1
    with session_factory() as session:
        assert session.query(MatchReadObservation).one().status == "failed"


def test_release_skip_is_partial_and_retries_on_short_cadence(settings, engine, session_factory, monkeypatch):
    from types import SimpleNamespace
    from data_platform.models import SyncRun
    from data_platform.services.match_read_cycle import MatchReadCycleService, _dispatch_summary

    now = datetime(2026, 9, 6, 12, tzinfo=timezone.utc)
    worker = MatchReadCycleService(
        settings=_settings(), session_factory=session_factory,
        fixture_loader=lambda **_: [_fixture(kickoff=now + timedelta(hours=8))],
        release_service=SimpleNamespace(release_matchday_to_website=lambda **_: {
            "released": [], "skipped": [{"fixture_id": "9001", "reason": "Read is stale."}],
        }),
    )
    monkeypatch.setattr(worker, "_dispatch_job", lambda job, **_: (_dispatch_summary(job, run=None), set(job.fixture_ids)))
    report = worker.run_once(now=now, mode="website")
    assert len(report.errors) == 1
    with session_factory() as session:
        assert session.get(SyncRun, report.run_id).status == "partial"
    assert worker.plan(now=now + timedelta(minutes=9), mode="website").jobs == ()
    assert len(worker.plan(now=now + timedelta(minutes=10), mode="website").jobs) == 1
