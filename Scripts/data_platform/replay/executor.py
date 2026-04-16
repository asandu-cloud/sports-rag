"""Execute a replay plan.

For each target in the plan, re-fetch the upstream payload, re-run the
same upsert path used by the sync pipeline, and re-enqueue KB docs.
Everything is idempotent — running the same plan twice is safe.

The executor separates the "fetch + write" concerns behind injectable
callables so tests can drive it without hitting API-Football.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import session_scope
from ..models import Competition, Fixture, Season, SyncRun
from ..sync.pipeline import SyncStats, sync_fixture_details
from ..sync.apifootball import ApiFootballClient
from ..sync.upserts import upsert_competition, upsert_fixture_from_api_row, upsert_season
from .planner import ReplayPlan

logger = logging.getLogger(__name__)


FixturePayloadFetcher = Callable[[int], Optional[Dict[str, Any]]]
FixtureStatsFetcher = Callable[[int], List[Dict[str, Any]]]
KBEnqueuer = Callable[[str, str, Optional[str]], int]


@dataclass
class ReplayResult:
    fixtures_replayed: int = 0
    fixture_failures: List[Dict[str, Any]] = field(default_factory=list)
    kb_enqueued: int = 0
    sync_run_id: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fixtures_replayed": self.fixtures_replayed,
            "fixture_failures": len(self.fixture_failures),
            "kb_enqueued": self.kb_enqueued,
            "sync_run_id": self.sync_run_id,
            "failures": self.fixture_failures[:10],
        }


class ReplayExecutor:
    """Run a :class:`ReplayPlan`.

    ``fetch_fixture_row`` is expected to return an API-Football-style
    ``fixture`` response dict (the element under ``response[i]``). In
    production callers pass ``ApiFootballClient``-backed adapters; tests
    pass stubs.
    """

    def __init__(
        self,
        *,
        fetch_fixture_row: FixturePayloadFetcher,
        fetch_team_stats: Optional[FixtureStatsFetcher] = None,
        fetch_player_stats: Optional[FixtureStatsFetcher] = None,
        kb_enqueuer: Optional[KBEnqueuer] = None,
        session_factory=session_scope,
        archiver=None,
    ):
        self._fetch_fixture_row = fetch_fixture_row
        self._fetch_team_stats = fetch_team_stats
        self._fetch_player_stats = fetch_player_stats
        self._kb_enqueuer = kb_enqueuer
        self._factory = session_factory
        self._archiver = archiver

    @classmethod
    def from_client(cls, client: ApiFootballClient, **kwargs) -> "ReplayExecutor":
        """Wire a real :class:`ApiFootballClient` into the executor."""
        def fetch_fixture(api_id: int):
            rows = client.fixtures(league=0, season=0)  # fallback; replaced below
            # API-Football requires league+season; we query by id via a separate path
            # The client doesn't expose /fixtures?id=X yet — so we use the stats endpoint
            # as an indirect probe; callers on production should provide their own.
            return None

        def fetch_team_stats(api_id: int):
            return client.fixture_statistics(api_id)

        def fetch_player_stats(api_id: int):
            return client.fixture_players(api_id)

        return cls(
            fetch_fixture_row=kwargs.pop("fetch_fixture_row", fetch_fixture),
            fetch_team_stats=kwargs.pop("fetch_team_stats", fetch_team_stats),
            fetch_player_stats=kwargs.pop("fetch_player_stats", fetch_player_stats),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    def execute(self, plan: ReplayPlan) -> ReplayResult:
        result = ReplayResult()
        run_scope = "replay:" + "|".join([
            f"fixtures={len(plan.fixtures)}",
            f"comp_seasons={len(plan.competition_seasons)}",
            f"kb={len(plan.kb_entities)}",
        ])
        with self._factory() as session:
            run = SyncRun(
                run_kind="replay", scope=run_scope,
                started_at=datetime.now(timezone.utc),
                status="running", stats={},
            )
            session.add(run)
            session.flush()
            result.sync_run_id = int(run.id)

        try:
            for api_id in plan.fixtures:
                self._replay_fixture(api_id, result)

            if self._kb_enqueuer is not None:
                for etype, ekey in plan.kb_entities:
                    try:
                        self._kb_enqueuer(etype, ekey, "replay")
                        result.kb_enqueued += 1
                    except Exception as exc:
                        logger.warning("kb enqueue failed for %s:%s — %s", etype, ekey, exc)

            with self._factory() as session:
                row = session.get(SyncRun, result.sync_run_id)
                if row is not None:
                    row.status = "completed"
                    row.finished_at = datetime.now(timezone.utc)
                    row.stats = result.as_dict()
        except Exception as exc:
            logger.exception("Replay executor failed")
            with self._factory() as session:
                row = session.get(SyncRun, result.sync_run_id)
                if row is not None:
                    row.status = "failed"
                    row.finished_at = datetime.now(timezone.utc)
                    row.error_text = str(exc)
                    row.stats = result.as_dict()
            raise
        return result

    # ------------------------------------------------------------------
    def _replay_fixture(self, api_id: int, result: ReplayResult) -> None:
        row = None
        try:
            row = self._fetch_fixture_row(api_id)
        except Exception as exc:
            result.fixture_failures.append({"fixture_api_id": api_id, "stage": "fetch_row", "error": str(exc)})
            return
        if row is None:
            result.fixture_failures.append({"fixture_api_id": api_id, "stage": "fetch_row", "error": "missing"})
            return
        league_block = row.get("league") or {}
        league_id = league_block.get("id")
        season_year = league_block.get("season")
        if not league_id or not season_year:
            result.fixture_failures.append({"fixture_api_id": api_id, "stage": "dispatch", "error": "missing league/season"})
            return

        with self._factory() as session:
            comp_spec = self._resolve_competition(session, int(league_id))
            if comp_spec is None:
                result.fixture_failures.append({
                    "fixture_api_id": api_id, "stage": "dispatch",
                    "error": f"unknown provider_id={league_id}",
                })
                return
            competition = upsert_competition(session, comp_spec)
            season = upsert_season(session, competition=competition, year=int(season_year))
            session.flush()
            try:
                fixture, _ = upsert_fixture_from_api_row(
                    session, api_row=row, competition=competition, season=season,
                )
            except Exception as exc:
                result.fixture_failures.append({"fixture_api_id": api_id, "stage": "upsert", "error": str(exc)})
                return

            stats = SyncStats()
            if self._fetch_team_stats is not None:
                try:
                    team_payload = self._fetch_team_stats(api_id)
                    from ..sync.upserts import upsert_fixture_team_stats
                    upsert_fixture_team_stats(session, fixture=fixture, stats_response=team_payload)
                except Exception as exc:
                    result.fixture_failures.append({"fixture_api_id": api_id, "stage": "team_stats", "error": str(exc)})
            if self._fetch_player_stats is not None:
                try:
                    player_payload = self._fetch_player_stats(api_id)
                    from ..sync.upserts import upsert_fixture_player_stats
                    upsert_fixture_player_stats(session, fixture=fixture, players_response=player_payload)
                except Exception as exc:
                    result.fixture_failures.append({"fixture_api_id": api_id, "stage": "player_stats", "error": str(exc)})

            if self._kb_enqueuer is not None:
                try:
                    self._kb_enqueuer("fixture", str(api_id), "replay")
                    result.kb_enqueued += 1
                except Exception:
                    logger.warning("kb enqueue for fixture %s failed", api_id, exc_info=True)
        result.fixtures_replayed += 1

    def _resolve_competition(self, session: Session, provider_id: int):
        # Lazy import avoids a cycle; we need the registry-driven spec.
        from ..sync.apifootball import COMPETITIONS
        for spec in COMPETITIONS.values():
            if spec.api_football_id == provider_id:
                return spec
        return None
