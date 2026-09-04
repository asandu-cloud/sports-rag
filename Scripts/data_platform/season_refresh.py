"""One fail-fast refresh workflow for the live pre-match model.

The project currently has two data paths that need to stay aligned:

* ``data_platform`` keeps the structured, auditable source of truth current;
* the established ``Output/`` -> normalized documents -> Chroma/model path is
  still what the Discord bot and prediction code read in production.

This module deliberately orchestrates the existing commands instead of
duplicating their logic.  That makes the weekly operator workflow one command
while preserving the proven legacy pipeline until the SQL-native prediction
path replaces it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]

# The default is intentionally the domestic leagues the regular product
# currently covers.  European competitions can be included explicitly once
# their season is active, avoiding unnecessary provider calls in quiet weeks.
DOMESTIC_COMPETITIONS: Tuple[str, ...] = (
    "EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1",
)
EUROPEAN_COMPETITIONS: Tuple[str, ...] = ("UCL", "UEL", "UECL")
SUPPORTED_COMPETITIONS = frozenset(DOMESTIC_COMPETITIONS + EUROPEAN_COMPETITIONS)


@dataclass(frozen=True)
class RefreshStage:
    """A child command in the season refresh workflow."""

    name: str
    command: Tuple[str, ...]
    accepted_exit_codes: Tuple[int, ...] = (0,)
    description: str = ""


@dataclass
class StageResult:
    """Result of a single child command, safe to emit in an operator report."""

    name: str
    command: List[str]
    return_code: Optional[int]
    elapsed_seconds: float
    status: str  # completed | warning | failed | not_run
    accepted_exit_codes: List[int]
    description: str = ""
    error: Optional[str] = None


@dataclass
class SeasonRefreshReport:
    """Structured outcome for a run of the full weekly workflow."""

    season: int
    competitions: List[str]
    dry_run: bool
    started_at: str
    finished_at: Optional[str] = None
    succeeded: bool = False
    failed_stage: Optional[str] = None
    stages: List[StageResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    next_action: Optional[str] = None

    def as_dict(self) -> dict:
        payload = asdict(self)
        # A long-lived bot may have cached Chroma/model objects.  The workflow
        # never restarts it automatically because that would be unsafe in a
        # managed deployment; make the required operator action unambiguous.
        payload["restart_long_lived_workers"] = self.succeeded and not self.dry_run
        return payload


Runner = Callable[..., object]


def current_season_year(today: Optional[date] = None) -> int:
    """Return the API-Football season start year for the current campaign."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def resolve_competitions(
    requested: Optional[Iterable[str]], *, include_europe: bool = False,
) -> Tuple[str, ...]:
    """Return de-duplicated valid competition codes in a predictable order."""
    codes = list(requested or DOMESTIC_COMPETITIONS)
    if include_europe:
        codes.extend(EUROPEAN_COMPETITIONS)

    result: List[str] = []
    for code in codes:
        if code not in result:
            result.append(code)
    unknown = [code for code in result if code not in SUPPORTED_COMPETITIONS]
    if unknown:
        valid = ", ".join(DOMESTIC_COMPETITIONS + EUROPEAN_COMPETITIONS)
        raise ValueError(f"Unsupported competition code(s): {', '.join(unknown)}. Valid: {valid}")
    if not result:
        raise ValueError("At least one competition is required.")
    return tuple(result)


def build_refresh_plan(
    *,
    season: int,
    competitions: Iterable[str],
    lookback_days: int = 7,
    sync_platform: bool = True,
    build_referees: bool = True,
    embed: bool = True,
    retrain_ml: bool = True,
    build_platform_features: bool = True,
) -> List[RefreshStage]:
    """Build the ordered refresh plan without executing it.

    Every normal stage must finish before the next one begins.  In particular,
    feature snapshots are imported only after the legacy feature engineering
    output has been rebuilt, and health is checked only after both paths have
    had a chance to update.
    """
    if season < 2000 or season > 2100:
        raise ValueError(f"Invalid season year: {season}")
    if lookback_days < 1:
        raise ValueError("lookback_days must be at least 1")

    codes = resolve_competitions(competitions)
    py = sys.executable
    codes_args = tuple(codes)
    plan: List[RefreshStage] = []

    if sync_platform:
        plan.append(RefreshStage(
            name="platform_incremental_sync",
            description="Update canonical fixtures, results, and raw input archive.",
            command=(
                py, "-m", "Scripts.data_platform", "refresh",
                "--competition", *codes_args,
                "--season", str(season),
                "--lookback-days", str(lookback_days),
            ),
        ))

    legacy_command: List[str] = [
        py,
        str(ROOT / "Scripts" / "pull_season.py"),
        "--season", str(season),
        "--league", *codes_args,
        "--normalize",
    ]
    if build_referees:
        legacy_command.append("--build-referees")
    if embed:
        legacy_command.append("--embed")
    if retrain_ml:
        legacy_command.append("--retrain-ml")
    plan.append(RefreshStage(
        name="legacy_model_refresh",
        description=(
            "Refresh Output data, rebuild features and domestic player profiles, "
            "then normalize the current season" + (", update Chroma" if embed else "") +
            (", and retrain cumulative ML artefacts." if retrain_ml else ".")
        ),
        command=tuple(legacy_command),
    ))

    if build_platform_features:
        plan.append(RefreshStage(
            name="platform_feature_snapshots",
            description="Import the rebuilt feature snapshots into the canonical database.",
            command=(
                py, "-m", "Scripts.data_platform", "build-features",
                "--competition", *codes_args,
                "--season", str(season),
            ),
        ))

    # A health warning (exit 1) is significant and recorded, but an old
    # watermark outside the selected competition must not disguise a successful
    # refresh.  Exit 2 is a real health error and stops the run.
    plan.append(RefreshStage(
        name="platform_health",
        description="Run the fast post-refresh health check.",
        command=(py, "-m", "Scripts.data_platform", "health", "--quick"),
        accepted_exit_codes=(0, 1),
    ))
    return plan


def _run_command(command: Sequence[str], *, runner: Runner) -> object:
    """Run a child command from repository root (small seam for test fakes)."""
    return runner(list(command), cwd=str(ROOT), check=False)


def execute_refresh_plan(
    plan: Iterable[RefreshStage],
    *,
    season: int,
    competitions: Iterable[str],
    dry_run: bool = False,
    runner: Runner = subprocess.run,
    monotonic: Callable[[], float] = time.monotonic,
) -> SeasonRefreshReport:
    """Execute a plan in order, stopping immediately at the first failure."""
    report = SeasonRefreshReport(
        season=season,
        competitions=list(competitions),
        dry_run=dry_run,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    stages = list(plan)

    if dry_run:
        for stage in stages:
            report.stages.append(StageResult(
                name=stage.name,
                command=list(stage.command),
                return_code=None,
                elapsed_seconds=0.0,
                status="not_run",
                accepted_exit_codes=list(stage.accepted_exit_codes),
                description=stage.description,
            ))
        report.succeeded = True
        report.finished_at = datetime.now(timezone.utc).isoformat()
        report.next_action = "Review the plan, then rerun without --dry-run."
        return report

    for stage in stages:
        print(f"\n[refresh-season] RUN {stage.name}: {' '.join(stage.command)}", flush=True)
        started = monotonic()
        try:
            completed = _run_command(stage.command, runner=runner)
            return_code = getattr(completed, "returncode", None)
            if not isinstance(return_code, int):
                raise RuntimeError("child runner returned no integer return code")
        except Exception as exc:
            elapsed = round(monotonic() - started, 3)
            report.stages.append(StageResult(
                name=stage.name,
                command=list(stage.command),
                return_code=None,
                elapsed_seconds=elapsed,
                status="failed",
                accepted_exit_codes=list(stage.accepted_exit_codes),
                description=stage.description,
                error=str(exc),
            ))
            report.failed_stage = stage.name
            report.finished_at = datetime.now(timezone.utc).isoformat()
            report.next_action = "Fix the failed stage and rerun refresh-season; completed stages are idempotent."
            print(f"[refresh-season] FAIL {stage.name}: {exc}", flush=True)
            return report

        elapsed = round(monotonic() - started, 3)
        if return_code not in stage.accepted_exit_codes:
            report.stages.append(StageResult(
                name=stage.name,
                command=list(stage.command),
                return_code=return_code,
                elapsed_seconds=elapsed,
                status="failed",
                accepted_exit_codes=list(stage.accepted_exit_codes),
                description=stage.description,
                error=f"exit code {return_code}",
            ))
            report.failed_stage = stage.name
            report.finished_at = datetime.now(timezone.utc).isoformat()
            report.next_action = "Fix the failed stage and rerun refresh-season; completed stages are idempotent."
            print(f"[refresh-season] FAIL {stage.name}: exit code {return_code}", flush=True)
            return report

        status = "completed" if return_code == 0 else "warning"
        report.stages.append(StageResult(
            name=stage.name,
            command=list(stage.command),
            return_code=return_code,
            elapsed_seconds=elapsed,
            status=status,
            accepted_exit_codes=list(stage.accepted_exit_codes),
            description=stage.description,
        ))
        if status == "warning":
            report.warnings.append(
                f"{stage.name} completed with accepted warning exit code {return_code}."
            )
            print(f"[refresh-season] WARN {stage.name} ({elapsed:.1f}s)", flush=True)
        else:
            print(f"[refresh-season] OK {stage.name} ({elapsed:.1f}s)", flush=True)

    report.succeeded = True
    report.finished_at = datetime.now(timezone.utc).isoformat()
    report.next_action = (
        "Restart long-lived Discord/web workers so cached Chroma and model artefacts reload, "
        "then inspect the health report and a live /market response."
    )
    return report


def run_refresh_season(args, *, runner: Runner = subprocess.run) -> SeasonRefreshReport:
    """Translate CLI arguments into the concrete plan and run it."""
    season = args.season if args.season is not None else current_season_year()
    competitions = resolve_competitions(
        args.competition,
        include_europe=getattr(args, "include_europe", False),
    )
    plan = build_refresh_plan(
        season=season,
        competitions=competitions,
        lookback_days=args.lookback_days,
        sync_platform=not args.skip_platform_sync,
        build_referees=not args.skip_referees,
        embed=not args.skip_embedding,
        retrain_ml=not args.skip_training,
        build_platform_features=not args.skip_platform_features,
    )
    return execute_refresh_plan(
        plan,
        season=season,
        competitions=competitions,
        dry_run=args.dry_run,
        runner=runner,
    )


def report_json(report: SeasonRefreshReport) -> str:
    """Render a stable, human-readable report without serialising secrets."""
    return json.dumps(report.as_dict(), indent=2, sort_keys=True)
