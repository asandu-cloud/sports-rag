"""Canonical data platform CLI — ``spixctl``.

Entry points
------------
``init-db``
    Apply Alembic migrations to create/upgrade the schema.

``bootstrap --competition EPL --season 2025 [--no-players]``
    Full backfill of a competition + season. Idempotent.

``refresh [--competition EPL] [--season 2025] [--lookback-days 7]``
    Incremental refresh driven by ``sync_watermarks``.

``build-features [--competition EPL] [--season 2025]``
    Import feature snapshots from existing ``Output/*_feature_engineering``
    JSON files — useful for bringing historical data into the DB without
    re-running the pull/FE pipeline.

``migrate-users [--sqlite-path ...] [--dry-run]``
    Copy users / subscription_events / referrals from
    ``Index/predictions.db`` into the canonical tables.

``status``
    Print watermarks + recent sync runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional

from sqlalchemy import select

from .config import SETTINGS
from .db import session_scope
from .models import SyncRun, SyncWatermark

logger = logging.getLogger("spixctl")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def cmd_init_db(args) -> int:
    from alembic import command
    from alembic.config import Config

    here = Path(__file__).resolve().parent
    cfg = Config(str(here / "alembic.ini"))
    cfg.set_main_option("script_location", str(here / "alembic"))
    cfg.set_main_option("sqlalchemy.url", SETTINGS.database_url)
    command.upgrade(cfg, "head")
    print(f"[init-db] Schema upgraded on {SETTINGS.database_url}")
    return 0


def cmd_bootstrap(args) -> int:
    from .storage import PayloadArchiver
    from .sync import ApiFootballClient, bootstrap_season

    client = ApiFootballClient()
    summary: List[dict] = []
    with session_scope() as session:
        archiver = PayloadArchiver(session) if args.archive else None
        for code in args.competition:
            for year in args.season:
                res = bootstrap_season(
                    session,
                    client=client,
                    code=code,
                    year=year,
                    fetch_players=not args.no_players,
                    archiver=archiver,
                    finished_only=not args.include_unfinished,
                )
                summary.append({"competition": code, "season": year, **res})
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_refresh(args) -> int:
    from .storage import PayloadArchiver
    from .sync import ApiFootballClient, incremental_refresh

    client = ApiFootballClient()
    codes = args.competition or None
    years = args.season or None
    with session_scope() as session:
        archiver = PayloadArchiver(session) if args.archive else None
        res = incremental_refresh(
            session,
            client=client,
            codes=codes,
            years=years,
            lookback_days=args.lookback_days,
            fetch_players=not args.no_players,
            archiver=archiver,
        )
    print(json.dumps(res, indent=2, default=str))
    return 0


def cmd_build_features(args) -> int:
    from .sync import build_feature_snapshots_from_files

    codes = args.competition or None
    years = args.season or None
    as_of = date.fromisoformat(args.as_of) if args.as_of else None
    with session_scope() as session:
        res = build_feature_snapshots_from_files(
            session,
            codes=codes,
            seasons=years,
            as_of_date=as_of,
            auto_fetch_teams=not args.no_fetch_teams,
        )
    print(json.dumps(res, indent=2, default=str))
    return 0


def cmd_migrate_users(args) -> int:
    from .compat import migrate_sqlite_users

    sqlite_path = Path(args.sqlite_path) if args.sqlite_path else None
    counts = migrate_sqlite_users(sqlite_path=sqlite_path, dry_run=args.dry_run)
    print(json.dumps(counts, indent=2))
    return 0


def cmd_migrate_predictions(args) -> int:
    from .compat import migrate_sqlite_predictions

    sqlite_path = Path(args.sqlite_path) if args.sqlite_path else None
    counts = migrate_sqlite_predictions(sqlite_path=sqlite_path, dry_run=args.dry_run)
    print(json.dumps(counts, indent=2))
    return 0


def cmd_migrate_parlays(args) -> int:
    from .compat import migrate_sqlite_parlay_sessions

    sqlite_path = Path(args.sqlite_path) if args.sqlite_path else None
    counts = migrate_sqlite_parlay_sessions(sqlite_path=sqlite_path, dry_run=args.dry_run)
    print(json.dumps(counts, indent=2))
    return 0


def cmd_migrate_live(args) -> int:
    from .compat import migrate_sqlite_live

    sqlite_path = Path(args.sqlite_path) if args.sqlite_path else None
    counts = migrate_sqlite_live(
        sqlite_path=sqlite_path,
        dry_run=args.dry_run,
        keep_rows_per_fixture=args.keep_rows_per_fixture,
    )
    print(json.dumps(counts, indent=2))
    return 0


def cmd_kb_refresh(args) -> int:
    from .kb import refresh_kb

    upserter = None
    if args.dry_run:
        upserter = lambda _docs: 0  # noqa: E731
    stats = refresh_kb(batch_size=args.batch_size, chroma_upserter=upserter, dry_run=args.dry_run)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_kb_enqueue_all(args) -> int:
    from .kb import enqueue_all_entities

    codes = args.competition or None
    counts = enqueue_all_entities(competitions=codes)
    print(json.dumps(counts, indent=2))
    return 0


def cmd_kb_status(args) -> int:
    from .repositories.kb import KBDocumentRepository

    repo = KBDocumentRepository()
    print(json.dumps({
        "queue": repo.queue_stats(),
        "docs": repo.count(),
        "pending_embed": len(repo.list_pending_embed()),
    }, indent=2))
    return 0


def cmd_health(args) -> int:
    from .observability import build_health_report

    report = build_health_report(include_data_quality=not args.quick)
    print(json.dumps(report.as_dict(), indent=2, default=str))
    if report.status == "error":
        return 2
    if report.status == "warn":
        return 1
    return 0


def cmd_registry(args) -> int:
    from .registry import load_registry

    registry = load_registry()
    rows = []
    for entry in registry.filter(label=args.label, competition_type=args.type, enabled_only=not args.all):
        rows.append({
            "code": entry.code,
            "name": entry.name,
            "country": entry.country,
            "type": entry.competition_type,
            "tier": entry.tier,
            "providers": list(entry.providers.keys()),
            "labels": list(entry.labels),
            "enabled": entry.enabled,
        })
    print(json.dumps({"version": registry.version, "count": len(rows), "entries": rows}, indent=2))
    return 0


def cmd_replay(args) -> int:
    from datetime import date as _date
    from .replay import ReplayScope, build_plan, ReplayExecutor

    scope = ReplayScope(
        fixture_api_ids=args.fixture or (),
        date_from=_date.fromisoformat(args.date_from) if args.date_from else None,
        date_to=_date.fromisoformat(args.date_to) if args.date_to else None,
        competition_codes=tuple(args.competition or ()),
        season_years=tuple(args.season or ()),
        kb_entities=tuple(
            tuple(item.split(":", 1)) if ":" in item else (item, "")
            for item in (args.kb_entity or ())
        ),
    )
    plan = build_plan(scope)
    print(json.dumps({"plan": plan.summary()}, indent=2))
    if args.plan_only:
        return 0

    if args.no_executor:
        return 0

    # No-op executor for operators running offline — prints the plan and
    # stops. A production build would pass a real API-Football client here.
    def _noop(api_id):
        return None

    executor = ReplayExecutor(
        fetch_fixture_row=_noop,
        kb_enqueuer=lambda etype, ekey, reason: 0,
    )
    result = executor.execute(plan)
    print(json.dumps(result.as_dict(), indent=2, default=str))
    return 0


def cmd_status(args) -> int:
    with session_scope() as session:
        watermarks = session.scalars(select(SyncWatermark)).all()
        recent = session.scalars(
            select(SyncRun).order_by(SyncRun.started_at.desc()).limit(20)
        ).all()
        out = {
            "database_url": SETTINGS.database_url,
            "platform_enabled": SETTINGS.platform_enabled,
            "raw_archive_backend": SETTINGS.raw_archive_backend,
            "raw_archive_root": str(SETTINGS.raw_archive_root),
            "watermarks": [{
                "scope": w.scope,
                "last_cursor": w.last_cursor,
                "last_successful_at": w.last_successful_at.isoformat() if w.last_successful_at else None,
            } for w in watermarks],
            "recent_runs": [{
                "id": r.id,
                "kind": r.run_kind,
                "scope": r.scope,
                "status": r.status,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                "stats": r.stats,
            } for r in recent],
        }
    print(json.dumps(out, indent=2, default=str))
    return 0


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spixctl", description="Betting RAG data platform CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-db", help="Apply Alembic migrations to the configured DATABASE_URL")
    _add_common_args(p)
    p.set_defaults(func=cmd_init_db)

    p = sub.add_parser("bootstrap", help="Full backfill for one or more competition/season pairs")
    _add_common_args(p)
    p.add_argument("--competition", nargs="+", required=True, help="Competition codes, e.g. EPL LaLiga UCL")
    p.add_argument("--season", nargs="+", type=int, required=True, help="API-Football season years, e.g. 2024 2025")
    p.add_argument("--no-players", action="store_true", help="Skip /fixtures/players (faster bootstrap)")
    p.add_argument("--include-unfinished", action="store_true", help="Include non-FT fixtures in bootstrap")
    p.add_argument("--archive", action="store_true", default=True, help="Archive raw payloads (default: on)")
    p.add_argument("--no-archive", dest="archive", action="store_false")
    p.set_defaults(func=cmd_bootstrap)

    p = sub.add_parser("refresh", help="Incremental refresh driven by watermarks")
    _add_common_args(p)
    p.add_argument("--competition", nargs="*", help="Restrict to these codes (default: all)")
    p.add_argument("--season", nargs="*", type=int, help="Restrict to these season years (default: current)")
    p.add_argument("--lookback-days", type=int, default=7)
    p.add_argument("--no-players", action="store_true")
    p.add_argument("--archive", action="store_true", default=True)
    p.add_argument("--no-archive", dest="archive", action="store_false")
    p.set_defaults(func=cmd_refresh)

    p = sub.add_parser("build-features", help="Import feature snapshots from Output/*_feature_engineering JSON files")
    _add_common_args(p)
    p.add_argument("--competition", nargs="*", help="Competition codes (default: all)")
    p.add_argument("--season", nargs="*", type=int, help="Season years (default: all found)")
    p.add_argument("--as-of", type=str, help="Override as_of_date (ISO, defaults to today)")
    p.add_argument(
        "--no-fetch-teams", action="store_true",
        help="Skip the one-time /teams?league=X&season=Y prefetch (offline mode); "
             "team snapshots will be skipped for unknown teams.",
    )
    p.set_defaults(func=cmd_build_features)

    p = sub.add_parser("migrate-users", help="Copy users/events/referrals from Index/predictions.db into Postgres")
    _add_common_args(p)
    p.add_argument("--sqlite-path", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_migrate_users)

    p = sub.add_parser("migrate-predictions", help="Copy predictions table from Index/predictions.db into Postgres")
    _add_common_args(p)
    p.add_argument("--sqlite-path", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_migrate_predictions)

    p = sub.add_parser("migrate-parlays", help="Copy parlay_sessions + trial_users + trial_parlays from Index/predictions.db")
    _add_common_args(p)
    p.add_argument("--sqlite-path", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_migrate_parlays)

    p = sub.add_parser("migrate-live", help="Copy Index/live.db contents into Postgres")
    _add_common_args(p)
    p.add_argument("--sqlite-path", type=str, default=None)
    p.add_argument("--keep-rows-per-fixture", type=int, default=None,
                   help="Cap high-volume snapshot tables at N rows per fixture")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_migrate_live)

    # KB sub-commands (also exposed as a single namespace for readability)
    p = sub.add_parser("kb-refresh", help="Drain kb_refresh_queue and delta-embed changed docs to Chroma")
    _add_common_args(p)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--dry-run", action="store_true", help="Skip the Chroma embed call; still updates kb_documents")
    p.set_defaults(func=cmd_kb_refresh)

    p = sub.add_parser("kb-enqueue-all", help="Enqueue every known team/player/fixture for KB refresh")
    _add_common_args(p)
    p.add_argument("--competition", nargs="*", help="Restrict to these codes (default: all)")
    p.set_defaults(func=cmd_kb_enqueue_all)

    p = sub.add_parser("kb-status", help="Print KB queue + doc stats")
    _add_common_args(p)
    p.set_defaults(func=cmd_kb_status)

    p = sub.add_parser("status", help="Print watermarks and recent sync runs")
    _add_common_args(p)
    p.set_defaults(func=cmd_status)

    # V3 commands
    p = sub.add_parser("health", help="Structured platform health + data-quality report")
    _add_common_args(p)
    p.add_argument("--quick", action="store_true", help="Skip the data-quality scan (faster liveness probe)")
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("registry", help="Dump the competition registry")
    _add_common_args(p)
    p.add_argument("--label", help="Filter to entries carrying this label (e.g. top5)")
    p.add_argument("--type", help="Filter to competition_type (domestic_league | continental_cup)")
    p.add_argument("--all", action="store_true", help="Include disabled entries")
    p.set_defaults(func=cmd_registry)

    p = sub.add_parser("replay", help="Plan / execute a replay over fixtures, date range, or KB entities")
    _add_common_args(p)
    p.add_argument("--fixture", nargs="*", type=int, help="Specific fixture api_football_ids")
    p.add_argument("--date-from", help="Inclusive ISO date (YYYY-MM-DD)")
    p.add_argument("--date-to", help="Inclusive ISO date (YYYY-MM-DD)")
    p.add_argument("--competition", nargs="*", help="Competition codes")
    p.add_argument("--season", nargs="*", type=int, help="Season years")
    p.add_argument("--kb-entity", nargs="*", help="e.g. team:EPL:42 or fixture:12345")
    p.add_argument("--plan-only", action="store_true", help="Just print the plan")
    p.add_argument("--no-executor", action="store_true", help="Skip calling the executor")
    p.set_defaults(func=cmd_replay)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
