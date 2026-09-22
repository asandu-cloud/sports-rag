"""One-shot scheduled Match Read refresh worker.

The worker deliberately owns *operations*, not model logic. It plans which
already-supported fixture/stage checks are due, invokes the existing
fixture-level dispatcher, records an immutable-or-unchanged outcome, and—only
in explicit ``website`` mode—promotes current cards through the established
release/tracking bridge.

An external scheduler (launchd, cron, or a cloud scheduler) invokes
``run_once`` at the desired cadence. There is no hidden background loop in the
website or Discord process.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
import logging
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import aliased

from ..config import MatchReadWorkerSettings, load_match_read_worker_settings
from ..db import session_scope, retry_database_busy
from ..models import Competition, Fixture, SyncRun, Team
from ..repositories.match_read_cycle import MatchReadCycleRepository, SUCCESSFUL_OBSERVATION_STATUSES
from .match_read_observations import MatchReadObservationService
from .match_read_release import MatchReadReleaseService
from .worker_runtime import DispatchTimeout, dispatch_deadline
from .refresh_coordination import data_access, refresh_generation, RefreshBusy


MATCH_READ_CYCLE_LEASE_KEY = "match-read-cycle"
logger = logging.getLogger(__name__)


def _as_utc(value: Optional[Any] = None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _parse_observed_at(value: Optional[Any]) -> Optional[datetime]:
    if value is None:
        return None
    try:
        return _as_utc(value)
    except (TypeError, ValueError):
        return None


def _event_identity(event: Mapping[str, Any]) -> str:
    for key in ("id", "event_id", "fixture_id", "_fixture_id"):
        value = str(event.get(key) or "").strip()
        if value:
            return value
    return ""


@dataclass(frozen=True)
class ScheduledMatchReadFixture:
    """A future fixture sourced from the canonical platform schedule."""

    fixture_id: str
    league: str
    kickoff_utc: datetime
    target_date: date
    home_team: str = ""
    away_team: str = ""


@dataclass(frozen=True)
class MatchReadCycleJob:
    """Planned league/date/stage work, executed one fixture at a time."""

    league: str
    target_date: date
    stage: str
    fixture_ids: Tuple[str, ...]
    force_refresh: bool
    release_fixture_ids: Tuple[str, ...] = ()
    # Only these pre-match fixtures receive the longer preliminary-card
    # freshness policy. A stale same-day card must never acquire that status
    # merely because it happens to be hours from kickoff.
    preliminary_fixture_ids: Tuple[str, ...] = ()
    reasons: Tuple[str, ...] = ()
    kickoff_by_fixture: Mapping[str, datetime] = field(default_factory=dict)
    priority_by_fixture: Mapping[str, Tuple[Any, ...]] = field(default_factory=dict)
    events_by_fixture: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchReadCyclePlan:
    now_utc: datetime
    mode: str
    jobs: Tuple[MatchReadCycleJob, ...]
    skipped: Tuple[Dict[str, Any], ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "now": self.now_utc.isoformat(),
            "mode": self.mode,
            "job_count": len(self.jobs),
            "jobs": [
                {
                    "league": job.league,
                    "date": job.target_date.isoformat(),
                    "stage": job.stage,
                    "fixture_ids": list(job.fixture_ids),
                    "force_refresh": job.force_refresh,
                    "release_fixture_ids": list(job.release_fixture_ids),
                    "preliminary_fixture_ids": list(job.preliminary_fixture_ids),
                    "reasons": list(job.reasons),
                }
                for job in self.jobs
            ],
            "skipped": list(self.skipped),
        }


@dataclass
class MatchReadCycleReport:
    """Serializable outcome of one worker invocation."""

    mode: str
    now_utc: datetime
    run_id: Optional[int] = None
    lease_acquired: bool = False
    dry_run: bool = False
    plan: Optional[MatchReadCyclePlan] = None
    dispatches: list[Dict[str, Any]] = field(default_factory=list)
    releases: list[Dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skipped_reason: Optional[str] = None
    deferred_fixture_stages: int = 0
    yield_reason: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.lease_acquired and not self.errors and self.skipped_reason is None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "now": self.now_utc.isoformat(),
            "run_id": self.run_id,
            "lease_acquired": self.lease_acquired,
            "dry_run": self.dry_run,
            "succeeded": self.succeeded,
            "skipped_reason": self.skipped_reason,
            "plan": self.plan.as_dict() if self.plan is not None else None,
            "dispatches": self.dispatches,
            "releases": self.releases,
            "errors": self.errors,
            "deferred_fixture_stages": self.deferred_fixture_stages,
            "yield_reason": self.yield_reason,
        }


def build_match_read_cycle_plan(
    fixtures: Iterable[ScheduledMatchReadFixture],
    *,
    latest_observations: Mapping[Tuple[str, str], Mapping[str, Any]],
    now: Optional[Any] = None,
    settings: Optional[MatchReadWorkerSettings] = None,
    mode: Optional[str] = None,
) -> MatchReadCyclePlan:
    """Purely decide what is due; it makes no database/provider calls.

    Policy:

    * Both modes create a pre-match baseline throughout the four-day outlook.
      Website mode releases that preliminary card, then keeps it current on a
      slower early cadence; shadow mode keeps it internal.
    * In the final window, both modes switch to the normal rapid refresh
      cadence so every successful website result meets the strict final-window
      freshness requirement.
    * Near kickoff, a verified-XI stage is retried independently. A fresh
      pre-match card is kept as fallback while lineups are unavailable.
    """
    config = settings or load_match_read_worker_settings()
    normalised_mode = _normalise_mode(mode or config.mode)
    reference_now = _as_utc(now)
    _validate_matchday_timezone(config)

    actions: list[Tuple[ScheduledMatchReadFixture, str, bool, bool, bool, str]] = []
    skipped: list[Dict[str, Any]] = []
    outlook_minutes = config.outlook_hours * 60

    for fixture in sorted(fixtures, key=lambda row: (row.kickoff_utc, row.league, row.fixture_id)):
        kickoff = _as_utc(fixture.kickoff_utc)
        remaining_minutes = (kickoff - reference_now).total_seconds() / 60.0
        if remaining_minutes <= 0:
            skipped.append({
                "fixture_id": fixture.fixture_id,
                "league": fixture.league,
                "reason": "Fixture has started or finished.",
            })
            continue
        if remaining_minutes > outlook_minutes:
            skipped.append({
                "fixture_id": fixture.fixture_id,
                "league": fixture.league,
                "reason": "Fixture is outside the Match Read outlook window.",
            })
            continue

        pre = latest_observations.get((fixture.fixture_id, "pre_match"))
        confirmed = latest_observations.get((fixture.fixture_id, "confirmed_lineups"))
        in_final_window = remaining_minutes <= config.final_window_minutes
        in_lineup_window = remaining_minutes <= config.lineup_window_minutes

        if in_lineup_window:
            # Keep one fresh price-only fallback while trying for verified XIs.
            if _observation_due(
                pre,
                reference_now,
                config.max_age_minutes,
                retry_minutes=config.refresh_minutes,
            ):
                actions.append((fixture, "pre_match", True, normalised_mode == "website", False, "fresh pre-match fallback"))
            if _observation_due(confirmed, reference_now, config.lineup_refresh_minutes):
                actions.append((fixture, "confirmed_lineups", True, normalised_mode == "website", False, "verified-lineup refresh"))
        elif in_final_window:
            if _observation_due(pre, reference_now, config.refresh_minutes):
                actions.append((fixture, "pre_match", True, normalised_mode == "website", False, "final-window price refresh"))
        else:
            # Earlier than the final window, a pre-match read is useful for
            # planning a matchday. It is deliberately refreshed on a slower
            # cadence, rather than treating every four-day-ahead fixture as a
            # live ten-minute odds check. Website mode releases the same
            # immutable early card; shadow mode retains it for review only.
            if _observation_due(pre, reference_now, config.early_refresh_minutes, retry_minutes=config.refresh_minutes):
                actions.append((
                    fixture,
                    "pre_match",
                    False,
                    normalised_mode == "website",
                    normalised_mode == "website",
                    "website early pre-match refresh" if normalised_mode == "website" else "shadow baseline",
                ))

    grouped: Dict[Tuple[str, date, str], Dict[str, Any]] = {}
    for fixture, stage, force_refresh, should_release, is_preliminary, reason in actions:
        key = (fixture.league, fixture.target_date, stage)
        bucket = grouped.setdefault(key, {
            "fixture_ids": set(),
            "release_fixture_ids": set(),
            "preliminary_fixture_ids": set(),
            "reasons": set(),
            "force_refresh": False,
            "kickoffs": {},
            "priorities": {},
            "events": {},
        })
        bucket["fixture_ids"].add(fixture.fixture_id)
        bucket["kickoffs"][fixture.fixture_id] = _as_utc(fixture.kickoff_utc)
        bucket["events"][fixture.fixture_id] = {
            "id": fixture.fixture_id, "home_team": fixture.home_team, "away_team": fixture.away_team,
            "commence_time": _as_utc(fixture.kickoff_utc).isoformat(),
        }
        observation = latest_observations.get((fixture.fixture_id, stage)) or {}
        checked = _parse_observed_at(observation.get("checked_at"))
        has_success = observation.get("status") in SUCCESSFUL_OBSERVATION_STATUSES
        near = (_as_utc(fixture.kickoff_utc) - reference_now).total_seconds() <= config.final_window_minutes * 60
        # Urgent missing fallbacks, urgent XI updates, other missing cards,
        # then routine rechecks. Oldest attempt first within each class:
        # a repeatedly failing fixture cannot monopolise the next cycle.
        priority = (0 if near and stage == "pre_match" and not has_success else
                    1 if near else 2 if not has_success else 3)
        bucket["priorities"][fixture.fixture_id] = (
            priority, checked or datetime.min.replace(tzinfo=timezone.utc),
            _as_utc(fixture.kickoff_utc), fixture.fixture_id, 0 if stage == "pre_match" else 1,
        )
        # A group fetches the provider slate once then filters locally. If any
        # fixture in that slate is in its final window, use one fresh provider
        # fetch for the entire group instead of a cached early pass plus a
        # duplicate final-window pass for the same league/date.
        bucket["force_refresh"] = bool(bucket["force_refresh"] or force_refresh)
        if should_release:
            bucket["release_fixture_ids"].add(fixture.fixture_id)
        if is_preliminary:
            bucket["preliminary_fixture_ids"].add(fixture.fixture_id)
        bucket["reasons"].add(reason)

    jobs = tuple(
        MatchReadCycleJob(
            league=league,
            target_date=target_date,
            stage=stage,
            fixture_ids=tuple(sorted(values["fixture_ids"])),
            force_refresh=bool(values["force_refresh"]),
            release_fixture_ids=tuple(sorted(values["release_fixture_ids"])),
            preliminary_fixture_ids=tuple(sorted(values["preliminary_fixture_ids"])),
            reasons=tuple(sorted(values["reasons"])),
            kickoff_by_fixture=values["kickoffs"],
            priority_by_fixture=values["priorities"],
            events_by_fixture=values["events"],
        )
        for (league, target_date, stage), values in sorted(
            grouped.items(), key=lambda item: (item[0][1], item[0][0], item[0][2])
        )
    )
    return MatchReadCyclePlan(
        now_utc=reference_now,
        mode=normalised_mode,
        jobs=jobs,
        skipped=tuple(skipped),
    )


class MatchReadCycleService:
    """Run the planned work using the existing dispatcher/release services."""

    def __init__(
        self,
        *,
        settings: Optional[MatchReadWorkerSettings] = None,
        cycle_repository: Optional[MatchReadCycleRepository] = None,
        observations: Optional[MatchReadObservationService] = None,
        release_service: Optional[MatchReadReleaseService] = None,
        session_factory=session_scope,
        fixture_loader: Optional[Callable[..., Sequence[ScheduledMatchReadFixture]]] = None,
        dispatcher: Optional[Callable[..., Any]] = None,
        lineup_provider: Optional[Callable[..., Any]] = None,
        lineup_preflight: Optional[Callable[..., Any]] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._settings = settings or load_match_read_worker_settings()
        self._repo = cycle_repository or MatchReadCycleRepository(session_factory=session_factory)
        self._observations = observations or MatchReadObservationService(repo=self._repo)
        self._release = release_service or MatchReadReleaseService()
        self._session_factory = session_factory
        self._fixture_loader = fixture_loader or _load_upcoming_platform_fixtures
        self._dispatcher = dispatcher or _default_dispatcher
        self._lineup_provider = lineup_provider
        self._lineup_preflight = lineup_preflight
        self._uses_default_dispatcher = dispatcher is None
        self._clock = clock

    def plan(self, *, now: Optional[Any] = None, mode: Optional[str] = None) -> MatchReadCyclePlan:
        config = self._settings
        normalised_mode = _normalise_mode(mode or config.mode)
        reference_now = _as_utc(now)
        fixtures = self._fixture_loader(
            now=reference_now,
            until=reference_now + timedelta(hours=config.outlook_hours),
            leagues=config.leagues,
        )
        latest = self._observations.latest_for_fixture_stages(
            (fixture.fixture_id for fixture in fixtures),
        )
        return build_match_read_cycle_plan(
            fixtures,
            latest_observations=latest,
            now=reference_now,
            settings=config,
            mode=normalised_mode,
        )

    def run_once(
        self,
        *,
        now: Optional[Any] = None,
        mode: Optional[str] = None,
        dry_run: bool = False,
    ) -> MatchReadCycleReport:
        """Run at most one safe Match Read cycle.

        ``dry_run`` is genuinely non-mutating: it only reads the fixture
        schedule and observation ledger, then prints the dispatch/release
        plan. It does not acquire a lease, call API-Football, or start a
        SyncRun.
        """
        reference_now = _as_utc(now)
        normalised_mode = _normalise_mode(mode or self._settings.mode)
        # Explicit --now remains a deterministic diagnostic clock. Normal
        # scheduler invocations must re-read wall time, including after sleep.
        clock = self._clock or (_as_utc if now is None else lambda: reference_now)
        started_monotonic = time.monotonic()
        budget = min(self._settings.max_cycle_seconds, max(1, self._settings.lease_seconds - 30))

        def remaining_seconds() -> float:
            elapsed = max(time.monotonic() - started_monotonic, (clock() - reference_now).total_seconds())
            return budget - elapsed

        report = MatchReadCycleReport(
            mode=normalised_mode,
            now_utc=reference_now,
            dry_run=bool(dry_run),
        )
        planned_generation = None
        try:
            if dry_run:
                plan = self.plan(now=reference_now, mode=normalised_mode)
            else:
                with data_access():
                    plan = self.plan(now=reference_now, mode=normalised_mode)
                    planned_generation = refresh_generation()
        except RefreshBusy as exc:
            report.skipped_reason = str(exc)
            return report
        except Exception as exc:
            report.errors.append(f"Could not plan Match Read cycle: {type(exc).__name__}: {exc}")
            return report
        report.plan = plan
        if dry_run:
            report.lease_acquired = True
            return report

        owner_id = uuid4().hex
        try:
            # No lease/run writes while a full refresh owns the data gate.
            with data_access():
                acquired = self._repo.acquire_lease(
                    lease_key=MATCH_READ_CYCLE_LEASE_KEY, owner_id=owner_id,
                    ttl_seconds=self._settings.lease_seconds, now=reference_now,
                    metadata={"mode": normalised_mode, "job_count": len(plan.jobs)},
                )
        except RefreshBusy as exc:
            report.skipped_reason = str(exc)
            return report
        except Exception as exc:
            report.errors.append(f"Could not acquire worker lease: {type(exc).__name__}: {exc}")
            return report
        report.lease_acquired = acquired
        if not acquired:
            report.skipped_reason = "Another Match Read cycle currently holds the worker lease."
            return report

        try:
            run_id = self._start_run(mode=normalised_mode, started_at=reference_now, plan=plan)
        except Exception as exc:
            report.errors.append(f"Could not start worker run: {type(exc).__name__}: {exc}")
            self._safe_release_lease(owner_id, report)
            return report
        report.run_id = run_id
        # Scope this cache to exactly one one-shot run. It shares a fresh
        # league/date provider slate between pre-match and lineup stages but
        # never survives to a later scheduled invocation.
        slate_cache: Dict[Any, Any] = {}
        slate_checked_at: Dict[Any, datetime] = {}
        units = [
            replace(
                job,
                fixture_ids=(fixture_id,),
                release_fixture_ids=tuple(fid for fid in job.release_fixture_ids if fid == fixture_id),
                preliminary_fixture_ids=tuple(fid for fid in job.preliminary_fixture_ids if fid == fixture_id),
            )
            for job in plan.jobs for fixture_id in job.fixture_ids
        ]
        units.sort(key=lambda job: job.priority_by_fixture[job.fixture_ids[0]])

        def owns_lease() -> bool:
            return self._repo.owns_lease(lease_key=MATCH_READ_CYCLE_LEASE_KEY, owner_id=owner_id, now=clock())

        data_gate = None
        input_generation = planned_generation
        try:
            for position, job in enumerate(units):
                if data_gate is not None:
                    data_gate.__exit__(None, None, None)
                    data_gate = None
                minimum_budget = min(self._settings.fixture_timeout_seconds, max(1, budget - 5))
                if remaining_seconds() < minimum_budget or not owns_lease():
                    report.deferred_fixture_stages = len(units) - position
                    report.yield_reason = "Insufficient budget for another full dispatch or lease expired; remaining work is replanned next tick."
                    break
                fixture_id = job.fixture_ids[0]
                kickoff = job.kickoff_by_fixture.get(fixture_id)
                if kickoff is not None and kickoff <= clock():
                    continue
                try:
                    gate = data_access()
                    gate.__enter__()
                    data_gate = gate
                    generation = refresh_generation()
                    if input_generation is not None and generation != input_generation:
                        report.deferred_fixture_stages = len(units) - position
                        report.yield_reason = "Data refreshed between fixtures; start a new process to reload all model inputs."
                        break
                    input_generation = generation
                except RefreshBusy as exc:
                    report.deferred_fixture_stages = len(units) - position
                    report.yield_reason = str(exc)
                    break
                cache_key = (job.league, job.target_date, job.force_refresh)
                if cache_key not in slate_cache:
                    slate_checked_at[cache_key] = clock()
                logger.info("Match Read %s %s %s: starting (%s/%s)", job.league, fixture_id, job.stage, position + 1, len(units))
                try:
                    with dispatch_deadline(min(self._settings.fixture_timeout_seconds, remaining_seconds())):
                        dispatch_report, successful_ids = self._dispatch_job(
                            job,
                            sync_run_id=run_id,
                            checked_at=slate_checked_at[cache_key],
                            slate_cache=slate_cache,
                        )
                except DispatchTimeout as exc:
                    message = f"{job.league} {fixture_id} {job.stage}: {exc}"
                    observation = self._record_failed_observation(
                        fixture_id=fixture_id, job=job, sync_run_id=run_id, checked_at=clock(), error=message,
                    )
                    dispatch_report = _dispatch_summary(job, run=None, observations=[observation], errors=[message])
                    dispatch_report["performance"] = getattr(exc, "prediction_performance", {})
                    successful_ids = set()
                report.dispatches.append(dispatch_report)
                if not owns_lease():
                    for successful_id in successful_ids:
                        self._record_failed_observation(
                            fixture_id=successful_id, job=job, sync_run_id=run_id, checked_at=clock(),
                            error="Lease expired during dispatch; publication was deferred.",
                        )
                    report.deferred_fixture_stages = len(units) - position
                    report.yield_reason = "Worker lease expired during dispatch; no further publication is allowed."
                    break
                # Publish every completed fixture immediately. The existing
                # release selector preserves confirmed-lineup precedence.
                if normalised_mode == "website" and successful_ids & set(job.release_fixture_ids):
                    release_started = time.monotonic()
                    try:
                        release = self._release.release_matchday_to_website(
                            league=job.league,
                            target_date=job.target_date,
                            now=clock(),
                            max_age_minutes=self._settings.max_age_minutes,
                            fixture_ids=[fixture_id],
                        )
                        report.releases.append(release)
                        logger.info("Match Read %s %s: released=%s skipped=%s", job.league, fixture_id,
                                    len(release.get("released", ())), len(release.get("skipped", ())))
                        for skipped in release.get("skipped", ()):
                            reason = str(skipped.get("reason") or "Website release skipped")
                            if reason != "Fixture has started or finished.":
                                message = f"{job.league} {fixture_id}: website release skipped: {reason}"
                                report.errors.append(message)
                                self._record_failed_observation(
                                    fixture_id=fixture_id, job=job, sync_run_id=run_id, checked_at=clock(), error=message,
                                )
                    except Exception as exc:
                        message = f"{job.league} {fixture_id}: website release failed ({type(exc).__name__}: {exc})."
                        report.errors.append(message)
                        self._record_failed_observation(
                            fixture_id=fixture_id, job=job, sync_run_id=run_id, checked_at=clock(), error=message,
                        )
                    finally:
                        dispatch_report["release_seconds"] = round(time.monotonic() - release_started, 6)

            if data_gate is not None:
                data_gate.__exit__(None, None, None)
                data_gate = None
            for dispatch in report.dispatches:
                if dispatch.get("errors"):
                    report.errors.extend(str(item) for item in dispatch["errors"])
            self._safe_finish_run(
                report,
                run_id,
                status="partial" if report.errors else ("deferred" if report.yield_reason else "completed"),
                finished_at=clock(),
                stats=_run_stats(report),
                error_text="\n".join(report.errors) if report.errors else None,
            )
        except Exception as exc:
            report.errors.append(f"Cycle execution failed ({type(exc).__name__}: {exc}).")
            self._safe_finish_run(
                report,
                run_id,
                status="failed",
                finished_at=_as_utc(),
                stats=_run_stats(report),
                error_text="\n".join(report.errors),
            )
        finally:
            if data_gate is not None:
                data_gate.__exit__(None, None, None)
            self._safe_release_lease(owner_id, report)
        return report

    def _dispatch_job(
        self,
        job: MatchReadCycleJob,
        *,
        sync_run_id: int,
        checked_at: datetime,
        slate_cache: Optional[Dict[Any, Any]] = None,
    ) -> Tuple[Dict[str, Any], set[str]]:
        """Invoke one grouped dispatcher call and append per-fixture facts."""
        allowed_ids = set(job.fixture_ids)

        def event_filter(event: Mapping[str, Any]) -> bool:
            return _event_identity(event) in allowed_ids

        lineup_provider = self._lineup_provider
        if job.stage == "confirmed_lineups" and lineup_provider is None:
            lineup_provider = _default_lineup_provider()

        try:
            preflight = self._lineup_preflight or (lineup_provider if self._uses_default_dispatcher else None)
            if job.stage == "confirmed_lineups" and preflight is not None:
                event = job.events_by_fixture[job.fixture_ids[0]]
                # Resolve verified XIs using the canonical fixture ID before
                # any odds slate/enrichment or team-profile evaluation.
                preflight_started = time.monotonic()
                context = preflight(event, job.league, job.target_date)
                preflight_seconds = round(time.monotonic() - preflight_started, 6)
                if not (getattr(context, "is_available", False) and getattr(context, "source", None) == "lineups"):
                    observation = self._observations.record(
                        fixture_api_id=job.fixture_ids[0], league=job.league, stage=job.stage,
                        status="lineups_pending", checked_at=checked_at, sync_run_id=sync_run_id,
                        error="Confirmed lineups are not available yet.",
                        detail={"preflight": True},
                    )
                    summary = _dispatch_summary(job, run=None, observations=[observation])
                    summary["performance"] = {
                        "timings_seconds": {"lineup_preflight": preflight_seconds},
                        "counts": {"lineup_preflight.calls": 1},
                    }
                    return summary, set()
                lineup_provider = lambda *_: context
            run = self._dispatcher(
                job.league,
                job.target_date,
                stage=job.stage,
                persist=True,
                lineup_provider=lineup_provider,
                event_filter=event_filter,
                preliminary_fixture_ids=set(job.preliminary_fixture_ids),
                force_refresh=job.force_refresh,
                slate_cache=slate_cache,
            )
        except Exception as exc:
            message = f"{job.league} {job.target_date} {job.stage}: dispatcher failed ({type(exc).__name__}: {exc})."
            for fixture_id in job.fixture_ids:
                self._record_failed_observation(
                    fixture_id=fixture_id,
                    job=job,
                    sync_run_id=sync_run_id,
                    checked_at=checked_at,
                    error=message,
                )
            summary = _dispatch_summary(job, run=None, errors=[message])
            summary["performance"] = getattr(exc, "prediction_performance", {})
            return summary, set()

        by_id = {
            str(getattr(item, "event_id", "") or "").strip(): item
            for item in getattr(run, "fixtures", ())
            if str(getattr(item, "event_id", "") or "").strip()
        }
        successful: set[str] = set()
        errors: list[str] = []
        observations: list[Dict[str, Any]] = []
        for fixture_id in job.fixture_ids:
            item = by_id.get(fixture_id)
            if item is None:
                message = (
                    f"{job.league} {job.target_date} {job.stage}: fixture {fixture_id} "
                    "was not returned by the dispatcher."
                )
                observations.append(self._record_failed_observation(
                    fixture_id=fixture_id,
                    job=job,
                    sync_run_id=sync_run_id,
                    checked_at=checked_at,
                    error=message,
                ))
                errors.append(message)
                continue

            record = getattr(item, "record", None) or {}
            item_error = str(getattr(item, "error", "") or "").strip()
            if record:
                created = bool(record.get("created"))
                observation = self._observations.record(
                    fixture_api_id=fixture_id,
                    league=job.league,
                    stage=job.stage,
                    status="verified" if created else "unchanged",
                    checked_at=checked_at,
                    sync_run_id=sync_run_id,
                    match_read_id=record.get("id"),
                    input_snapshot_id=(record.get("provenance") or {}).get("input_snapshot_id"),
                    read_key=record.get("read_key"),
                    provider_calls={"force_refresh_requested": job.force_refresh},
                    detail={"created": created, "fixture_label": getattr(item, "fixture_label", None)},
                )
                observations.append(observation)
                successful.add(fixture_id)
                continue

            if job.stage == "confirmed_lineups" and _is_lineups_pending(item_error):
                observation = self._observations.record(
                    fixture_api_id=fixture_id,
                    league=job.league,
                    stage=job.stage,
                    status="lineups_pending",
                    checked_at=checked_at,
                    sync_run_id=sync_run_id,
                    provider_calls={"force_refresh_requested": job.force_refresh},
                    detail={"fixture_label": getattr(item, "fixture_label", None)},
                    error=item_error or "Confirmed lineups are not available yet.",
                )
                observations.append(observation)
                continue

            message = item_error or f"{getattr(item, 'fixture_label', fixture_id)} did not persist a Match Read."
            observations.append(self._record_failed_observation(
                fixture_id=fixture_id,
                job=job,
                sync_run_id=sync_run_id,
                checked_at=checked_at,
                error=message,
            ))
            errors.append(message)

        return _dispatch_summary(job, run=run, observations=observations, errors=errors), successful

    def _record_failed_observation(
        self,
        *,
        fixture_id: str,
        job: MatchReadCycleJob,
        sync_run_id: int,
        checked_at: datetime,
        error: str,
    ) -> Dict[str, Any]:
        previous = self._observations.latest_for_fixture_stages([fixture_id]).get((fixture_id, job.stage), {})
        failures = (int((previous.get("detail") or {}).get("consecutive_failures", 1)) + 1
                    if previous.get("status") == "failed" else 1)
        retry_minutes = min(30, self._settings.refresh_minutes * (2 ** min(failures - 1, 4)))
        return self._observations.record(
            fixture_api_id=fixture_id,
            league=job.league,
            stage=job.stage,
            status="failed",
            checked_at=checked_at,
            sync_run_id=sync_run_id,
            provider_calls={"force_refresh_requested": job.force_refresh},
            error=error,
            detail={"consecutive_failures": failures,
                    "next_retry_at": (checked_at + timedelta(minutes=retry_minutes)).isoformat()},
        )

    def _safe_release_lease(self, owner_id, report):
        try:
            self._repo.release_lease(lease_key=MATCH_READ_CYCLE_LEASE_KEY, owner_id=owner_id, now=_as_utc())
        except Exception as exc:
            message = f"Worker lease cleanup failed (lease will expire): {type(exc).__name__}: {exc}"
            logger.error(message)
            report.errors.append(message)

    def _safe_finish_run(self, report, *args, **kwargs):
        try:
            self._finish_run(*args, **kwargs)
        except Exception as exc:
            message = f"Could not save worker run outcome: {type(exc).__name__}: {exc}"
            logger.error(message)
            report.errors.append(message)

    @retry_database_busy
    def _start_run(self, *, mode: str, started_at: datetime, plan: MatchReadCyclePlan) -> int:
        with self._session_factory() as session:
            row = SyncRun(
                run_kind="match_read_cycle",
                scope=f"{mode}:{','.join(self._settings.leagues)}",
                started_at=started_at,
                status="running",
                stats={"planned_jobs": len(plan.jobs), "mode": mode},
            )
            session.add(row)
            session.flush()
            return int(row.id)

    @retry_database_busy
    def _finish_run(
        self,
        run_id: int,
        *,
        status: str,
        finished_at: datetime,
        stats: Mapping[str, Any],
        error_text: Optional[str],
    ) -> None:
        with self._session_factory() as session:
            row = session.get(SyncRun, int(run_id))
            if row is None:
                return
            row.status = status
            row.finished_at = finished_at
            row.stats = dict(stats)
            row.error_text = error_text


def _load_upcoming_platform_fixtures(
    *,
    now: datetime,
    until: datetime,
    leagues: Sequence[str],
) -> Sequence[ScheduledMatchReadFixture]:
    """Read the canonical fixture schedule without hitting API-Football.

    The regular season-refresh job owns schedule ingestion. Keeping worker
    planning database-only is what makes a ten-minute cycle inexpensive and
    lets a failed refresh surface clearly as missing schedule data instead of
    multiplying provider calls from every delivery process.
    """
    allowed = tuple(str(value).strip() for value in leagues if str(value).strip())
    if not allowed:
        return ()
    reference = _as_utc(now)
    horizon = _as_utc(until)
    home, away = aliased(Team), aliased(Team)
    with session_scope() as session:
        rows = session.execute(
            select(Fixture.api_football_id, Competition.code, Fixture.kickoff_utc, home.name, away.name)
            .join(Competition, Fixture.competition_id == Competition.id)
            .join(home, Fixture.home_team_id == home.id)
            .join(away, Fixture.away_team_id == away.id)
            .where(
                Competition.code.in_(allowed),
                Fixture.kickoff_utc.is_not(None),
                Fixture.status.in_(("NS", "TBD")),
            )
            .order_by(Fixture.kickoff_utc.asc(), Competition.code.asc(), Fixture.api_football_id.asc())
        ).all()
    scheduled = []
    for api_id, league, kickoff, home_name, away_name in rows:
        if kickoff is None:
            continue
        # SQLite returns these UTC values without tzinfo. Compare after
        # normalisation in Python so local SQLite and hosted PostgreSQL obey
        # exactly the same future-window contract.
        kickoff_utc = _as_utc(kickoff)
        if not (reference < kickoff_utc <= horizon):
            continue
        scheduled.append(ScheduledMatchReadFixture(
            fixture_id=str(api_id),
            league=str(league),
            kickoff_utc=kickoff_utc,
            # Persisted Match Reads group by the source UTC ISO date today.
            # Keep the scheduler aligned with that established contract.
            target_date=kickoff_utc.date(),
            home_team=home_name,
            away_team=away_name,
        ))
    return tuple(scheduled)


def _default_dispatcher(*args: Any, **kwargs: Any) -> Any:
    rag_root = Path(__file__).resolve().parents[2] / "rag_ingest"
    if str(rag_root) not in sys.path:
        sys.path.insert(0, str(rag_root))
    try:
        from rendering.match_read_dispatch import generate_match_reads_sync
    except ImportError:
        from Scripts.rag_ingest.rendering.match_read_dispatch import generate_match_reads_sync  # type: ignore[import]
    return generate_match_reads_sync(*args, **kwargs)


def _default_lineup_provider() -> Callable[..., Any]:
    rag_root = Path(__file__).resolve().parents[2] / "rag_ingest"
    if str(rag_root) not in sys.path:
        sys.path.insert(0, str(rag_root))
    try:
        from lineup_context import get_confirmed_lineup_context_for_event
    except ImportError:
        from Scripts.rag_ingest.lineup_context import get_confirmed_lineup_context_for_event  # type: ignore[import]
    return get_confirmed_lineup_context_for_event


def _normalise_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode not in {"shadow", "website"}:
        raise ValueError("mode must be 'shadow' or 'website'.")
    return mode


def _validate_matchday_timezone(settings: MatchReadWorkerSettings) -> None:
    value = str(settings.matchday_timezone or "").strip().upper()
    if value not in {"UTC", "ETC/UTC", "GMT"}:
        raise ValueError(
            "MATCH_READ_MATCHDAY_TIMEZONE must remain UTC until Match Read storage and website date grouping "
            "are migrated together."
        )


def _observation_due(
    observation: Optional[Mapping[str, Any]],
    now: datetime,
    interval_minutes: int,
    *,
    retry_minutes: Optional[int] = None,
) -> bool:
    if not observation:
        return True
    retry_at = _parse_observed_at((observation.get("detail") or {}).get("next_retry_at"))
    if observation.get("status") == "failed" and retry_at is not None:
        return now >= retry_at
    checked_at = _parse_observed_at(observation.get("checked_at"))
    if checked_at is None:
        return True
    status = str(observation.get("status") or "").strip().lower()
    # A successful immutable read may remain the valid fallback until its
    # publication-age limit. A failed provider/model call is not a valid
    # fallback, so retry it at the normal short cadence instead of waiting two
    # hours merely because the *success* age limit is two hours.
    cadence = (
        retry_minutes
        if status not in SUCCESSFUL_OBSERVATION_STATUSES and retry_minutes is not None
        else interval_minutes
    )
    return now - checked_at >= timedelta(minutes=max(1, int(cadence)))


def _is_lineups_pending(error: str) -> bool:
    text = str(error or "").lower()
    return "confirmed lineups are not available" in text or "confirmed-lineups lookup" in text


def _dispatch_summary(
    job: MatchReadCycleJob,
    *,
    run: Optional[Any],
    observations: Sequence[Mapping[str, Any]] = (),
    errors: Sequence[str] = (),
) -> Dict[str, Any]:
    return {
        "league": job.league,
        "date": job.target_date.isoformat(),
        "stage": job.stage,
        "fixture_ids": list(job.fixture_ids),
        "force_refresh": job.force_refresh,
        "dispatcher_fixture_count": len(getattr(run, "fixtures", ()) or ()) if run is not None else 0,
        "dispatcher_notes": list(getattr(run, "notes", ()) or ()) if run is not None else [],
        "observations": [dict(value) for value in observations],
        "errors": list(errors),
        "performance": getattr(run, "performance", {}) if run is not None else {},
    }


def _run_stats(report: MatchReadCycleReport) -> Dict[str, Any]:
    observations = [
        observation
        for dispatch in report.dispatches
        for observation in dispatch.get("observations", ())
        if isinstance(observation, Mapping)
    ]
    statuses: Dict[str, int] = defaultdict(int)
    for observation in observations:
        statuses[str(observation.get("status") or "unknown")] += 1
    return {
        "mode": report.mode,
        "planned_jobs": len(report.plan.jobs) if report.plan is not None else 0,
        "dispatch_count": len(report.dispatches),
        "release_count": len(report.releases),
        "observation_counts": dict(statuses),
        "error_count": len(report.errors),
        "deferred_fixture_stages": report.deferred_fixture_stages,
        "yield_reason": report.yield_reason,
        "fixture_performance": [
            {"fixture_ids": item["fixture_ids"], "league": item["league"], "stage": item["stage"],
             "performance": item.get("performance", {}), "release_seconds": item.get("release_seconds")}
            for item in report.dispatches
        ],
    }
