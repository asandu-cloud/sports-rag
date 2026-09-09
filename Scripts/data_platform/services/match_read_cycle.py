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
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from sqlalchemy import select

from ..config import MatchReadWorkerSettings, load_match_read_worker_settings
from ..db import session_scope
from ..models import Competition, Fixture, SyncRun
from ..repositories.match_read_cycle import MatchReadCycleRepository, SUCCESSFUL_OBSERVATION_STATUSES
from .match_read_observations import MatchReadObservationService
from .match_read_release import MatchReadReleaseService


MATCH_READ_CYCLE_LEASE_KEY = "match-read-cycle"


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


@dataclass(frozen=True)
class MatchReadCycleJob:
    """One dispatcher invocation, grouped by league/date/stage."""

    league: str
    target_date: date
    stage: str
    fixture_ids: Tuple[str, ...]
    force_refresh: bool
    release_fixture_ids: Tuple[str, ...] = ()
    reasons: Tuple[str, ...] = ()


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

    * Shadow mode creates one early pre-match baseline inside the outlook
      window, then refreshes in the final two hours.
    * Website mode does not make early cards public. It starts work in the
      final window only, where every successful result can meet the release
      freshness requirement.
    * Near kickoff, a verified-XI stage is retried independently. A fresh
      pre-match card is kept as fallback while lineups are unavailable.
    """
    config = settings or load_match_read_worker_settings()
    normalised_mode = _normalise_mode(mode or config.mode)
    reference_now = _as_utc(now)
    _validate_matchday_timezone(config)

    actions: list[Tuple[ScheduledMatchReadFixture, str, bool, bool, str]] = []
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
                actions.append((fixture, "pre_match", True, normalised_mode == "website", "fresh pre-match fallback"))
            if _observation_due(confirmed, reference_now, config.lineup_refresh_minutes):
                actions.append((fixture, "confirmed_lineups", True, normalised_mode == "website", "verified-lineup refresh"))
        elif in_final_window:
            if _observation_due(pre, reference_now, config.refresh_minutes):
                actions.append((fixture, "pre_match", True, normalised_mode == "website", "final-window price refresh"))
        elif normalised_mode == "shadow":
            # Earlier than the final window, one baseline makes the shadow
            # comparison useful but does not spend calls every ten minutes.
            if _observation_due(pre, reference_now, max(config.refresh_minutes * 6, 60)):
                actions.append((fixture, "pre_match", False, False, "shadow baseline"))
        else:
            skipped.append({
                "fixture_id": fixture.fixture_id,
                "league": fixture.league,
                "reason": "Website mode waits for the final publication window.",
            })

    grouped: Dict[Tuple[str, date, str], Dict[str, Any]] = {}
    for fixture, stage, force_refresh, should_release, reason in actions:
        key = (fixture.league, fixture.target_date, stage)
        bucket = grouped.setdefault(key, {
            "fixture_ids": set(),
            "release_fixture_ids": set(),
            "reasons": set(),
            "force_refresh": False,
        })
        bucket["fixture_ids"].add(fixture.fixture_id)
        # A group fetches the provider slate once then filters locally. If any
        # fixture in that slate is in its final window, use one fresh provider
        # fetch for the entire group instead of a cached early pass plus a
        # duplicate final-window pass for the same league/date.
        bucket["force_refresh"] = bool(bucket["force_refresh"] or force_refresh)
        if should_release:
            bucket["release_fixture_ids"].add(fixture.fixture_id)
        bucket["reasons"].add(reason)

    jobs = tuple(
        MatchReadCycleJob(
            league=league,
            target_date=target_date,
            stage=stage,
            fixture_ids=tuple(sorted(values["fixture_ids"])),
            force_refresh=bool(values["force_refresh"]),
            release_fixture_ids=tuple(sorted(values["release_fixture_ids"])),
            reasons=tuple(sorted(values["reasons"])),
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
    ) -> None:
        self._settings = settings or load_match_read_worker_settings()
        self._repo = cycle_repository or MatchReadCycleRepository(session_factory=session_factory)
        self._observations = observations or MatchReadObservationService(repo=self._repo)
        self._release = release_service or MatchReadReleaseService()
        self._session_factory = session_factory
        self._fixture_loader = fixture_loader or _load_upcoming_platform_fixtures
        self._dispatcher = dispatcher or _default_dispatcher
        self._lineup_provider = lineup_provider

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
        report = MatchReadCycleReport(
            mode=normalised_mode,
            now_utc=reference_now,
            dry_run=bool(dry_run),
        )
        try:
            plan = self.plan(now=reference_now, mode=normalised_mode)
        except Exception as exc:
            report.errors.append(f"Could not plan Match Read cycle: {type(exc).__name__}: {exc}")
            return report
        report.plan = plan
        if dry_run:
            report.lease_acquired = True
            return report

        owner_id = uuid4().hex
        acquired = self._repo.acquire_lease(
            lease_key=MATCH_READ_CYCLE_LEASE_KEY,
            owner_id=owner_id,
            ttl_seconds=self._settings.lease_seconds,
            now=reference_now,
            metadata={"mode": normalised_mode, "job_count": len(plan.jobs)},
        )
        report.lease_acquired = acquired
        if not acquired:
            report.skipped_reason = "Another Match Read cycle currently holds the worker lease."
            return report

        run_id = self._start_run(mode=normalised_mode, started_at=reference_now, plan=plan)
        report.run_id = run_id
        release_candidates: Dict[Tuple[str, date], set[str]] = defaultdict(set)
        # Scope this cache to exactly one one-shot run. It shares a fresh
        # league/date provider slate between pre-match and lineup stages but
        # never survives to a later scheduled invocation.
        slate_cache: Dict[Any, Any] = {}
        try:
            for job in plan.jobs:
                dispatch_report, successful_ids = self._dispatch_job(
                    job,
                    sync_run_id=run_id,
                    checked_at=reference_now,
                    slate_cache=slate_cache,
                )
                report.dispatches.append(dispatch_report)
                if normalised_mode == "website":
                    permitted = set(job.release_fixture_ids)
                    release_candidates[(job.league, job.target_date)].update(successful_ids & permitted)

            if normalised_mode == "website":
                for (league, target_date), fixture_ids in sorted(release_candidates.items()):
                    if not fixture_ids:
                        continue
                    try:
                        release = self._release.release_matchday_to_website(
                            league=league,
                            target_date=target_date,
                            now=reference_now,
                            max_age_minutes=self._settings.max_age_minutes,
                            fixture_ids=sorted(fixture_ids),
                        )
                        report.releases.append(release)
                    except Exception as exc:
                        report.errors.append(
                            f"{league} {target_date}: website release failed ({type(exc).__name__}: {exc})."
                        )

            for dispatch in report.dispatches:
                if dispatch.get("errors"):
                    report.errors.extend(str(item) for item in dispatch["errors"])
            self._finish_run(
                run_id,
                status="completed" if not report.errors else "partial",
                finished_at=_as_utc(),
                stats=_run_stats(report),
                error_text="\n".join(report.errors) if report.errors else None,
            )
        except Exception as exc:
            report.errors.append(f"Cycle execution failed ({type(exc).__name__}: {exc}).")
            self._finish_run(
                run_id,
                status="failed",
                finished_at=_as_utc(),
                stats=_run_stats(report),
                error_text="\n".join(report.errors),
            )
        finally:
            self._repo.release_lease(
                lease_key=MATCH_READ_CYCLE_LEASE_KEY,
                owner_id=owner_id,
                now=_as_utc(),
            )
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
            run = self._dispatcher(
                job.league,
                job.target_date,
                stage=job.stage,
                persist=True,
                lineup_provider=lineup_provider,
                event_filter=event_filter,
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
            return _dispatch_summary(job, run=None, errors=[message]), set()

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
        return self._observations.record(
            fixture_api_id=fixture_id,
            league=job.league,
            stage=job.stage,
            status="failed",
            checked_at=checked_at,
            sync_run_id=sync_run_id,
            provider_calls={"force_refresh_requested": job.force_refresh},
            error=error,
        )

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
    with session_scope() as session:
        rows = session.execute(
            select(Fixture.api_football_id, Competition.code, Fixture.kickoff_utc)
            .join(Competition, Fixture.competition_id == Competition.id)
            .where(
                Competition.code.in_(allowed),
                Fixture.kickoff_utc.is_not(None),
            )
            .order_by(Fixture.kickoff_utc.asc(), Competition.code.asc(), Fixture.api_football_id.asc())
        ).all()
    scheduled = []
    for api_id, league, kickoff in rows:
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
    }
