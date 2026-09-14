"""Drain the refresh queue, build docs, delta-upsert to Chroma."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Callable, Dict, Iterable, List, Optional

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from ..db import session_scope
from ..models import (
    Competition,
    Fixture,
    Player,
    PlayerFeatureSnapshot,
    Season,
    Team,
    TeamFeatureSnapshot,
)
from ..repositories.kb import KBDocumentRepository
from ..services.kb import KBService
from ..vectors import VectorBackend, build_backend
from .chroma_adapter import embed_and_upsert_chroma
from .doc_builders import build_docs_for_entity, make_doc_id

logger = logging.getLogger(__name__)


def _default_chroma_upserter(docs: List[Dict[str, Any]]) -> int:
    """Legacy path kept for back-compat — prefer :func:`_make_default_upserter`."""
    legacy_docs = [
        {"id": d["id"], "text": d["text"], "metadata": d.get("metadata", {})}
        for d in docs
    ]
    return embed_and_upsert_chroma(legacy_docs)


def _make_default_upserter(backend: Optional[VectorBackend] = None) -> Callable[[List[Dict[str, Any]]], int]:
    """Return a refresher upserter bound to the configured vector backend.

    The backend is resolved lazily at call time so tests can swap it via
    ``VECTOR_BACKEND=inmemory`` without re-importing.
    """
    def _upsert(docs: List[Dict[str, Any]]) -> int:
        target = backend or build_backend()
        payload = [
            {"id": d["id"], "text": d["text"], "metadata": d.get("metadata") or {}}
            for d in docs
        ]
        return target.upsert(payload).upserted

    return _upsert


def refresh_kb(
    *,
    batch_size: int = 50,
    chroma_upserter: Optional[Callable[[List[Dict[str, Any]]], int]] = None,
    vector_backend: Optional[VectorBackend] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Drain the refresh queue and push changed docs to the vector backend.

    The default backend is resolved via :func:`vectors.build_backend` so
    operators can set ``VECTOR_BACKEND=inmemory`` (tests) or
    ``VECTOR_BACKEND=chroma`` (production) without code changes.
    """
    if chroma_upserter is None and not dry_run:
        chroma_upserter = _make_default_upserter(vector_backend)
    service = KBService(
        repo=KBDocumentRepository(),
        doc_builder=build_docs_for_entity,
        chroma_upserter=chroma_upserter,
    )
    totals = {
        "dequeued": 0,
        "docs_scanned": 0,
        "docs_changed": 0,
        "docs_embedded": 0,
        "error_count": 0,
    }
    while True:
        stats = service.refresh(batch_size=batch_size, dry_run=dry_run)
        if stats["dequeued"] == 0:
            break
        for k in totals:
            totals[k] += int(stats.get(k, 0))
    # A previous process may have saved canonical docs but crashed before its
    # Chroma write.  Reconcile from the canonical sync markers once the queue
    # itself is drained; this also repairs interrupted historical imports.
    totals["docs_embedded"] += service.sync_pending(dry_run=dry_run)
    totals["pending_embed"] = service.pending_embed_count()
    return totals


def migrate_fixture_document_ids(
    *,
    competitions: Optional[Iterable[str]] = None,
    vector_backend: Optional[VectorBackend] = None,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Move legacy fixture docs to ids based on provider fixture identity.

    Earlier fixture-document ids used team names and matchup text.  That is
    not unique when a pair meets again in the same season (including
    play-offs).  Vector rows under the old ids are removed and canonical rows
    are marked unsynced so a normal ``kb-refresh`` rebuilds them safely.
    """
    repo = KBDocumentRepository()
    docs = repo.list_documents(doc_type="team_fixture", leagues=competitions)
    mapping: Dict[str, str] = {}
    skipped = 0
    for doc in docs:
        meta = doc.get("metadata") or {}
        fixture_id = meta.get("fixture_api_id", meta.get("fixture_id"))
        team_id = meta.get("team_id")
        league = doc.get("league") or meta.get("league")
        season = doc.get("season") or meta.get("season")
        if fixture_id is None or team_id is None or not league or not season:
            skipped += 1
            continue
        mapping[doc["doc_id"]] = make_doc_id([
            league,
            season,
            "team_fixture",
            fixture_id,
            team_id,
        ])

    mapping = {old: new for old, new in mapping.items() if old != new}
    if dry_run:
        return {"scanned": len(docs), "rekeyed": len(mapping), "skipped": skipped, "deleted_vectors": 0}

    # Remove legacy vector ids before their canonical counterparts are
    # rekeyed.  A subsequent refresh writes only the corrected identities.
    deleted_vectors = 0
    if mapping:
        target = vector_backend or build_backend()
        deleted_vectors = int(target.delete(mapping.keys()) or 0)
    rekeyed = repo.rekey_documents(mapping, invalidate_sync=True)
    return {
        "scanned": len(docs),
        "rekeyed": rekeyed,
        "skipped": skipped,
        "deleted_vectors": deleted_vectors,
    }


def enqueue_all_entities(*, competitions: Optional[Iterable[str]] = None) -> Dict[str, int]:
    """Bulk-enqueue every team + player + fixture known to the DB.

    Used once after bootstrapping the canonical DB to populate
    ``kb_documents`` from structured data instead of the old
    ``Index/normalized_*.json`` files.
    """
    counts = {"team": 0, "player": 0, "fixture": 0}
    svc = KBService(
        repo=KBDocumentRepository(),
        doc_builder=build_docs_for_entity,
    )
    with session_scope() as session:
        if competitions is None:
            comp_codes = [c.code for c in session.scalars(select(Competition)).all()]
        else:
            comp_codes = list(competitions)

        for code in comp_codes:
            comp = session.scalar(select(Competition).where(Competition.code == code))
            if comp is None:
                continue

            # Teams discovered via either fixtures OR feature snapshots — so this
            # works after `bootstrap` and after a fixture-less `build-features` run.
            from_fixtures = session.scalars(
                select(Team.api_football_id).distinct()
                .join(Fixture, (Fixture.home_team_id == Team.id) | (Fixture.away_team_id == Team.id))
                .where(Fixture.competition_id == comp.id)
            ).all()
            from_snapshots = session.scalars(
                select(Team.api_football_id).distinct()
                .join(TeamFeatureSnapshot, TeamFeatureSnapshot.team_id == Team.id)
                .join(Season, Season.id == TeamFeatureSnapshot.season_id)
                .where(Season.competition_id == comp.id)
            ).all()
            team_ids = {x for x in (list(from_fixtures) + list(from_snapshots)) if x is not None}
            for api_id in sorted(team_ids):
                svc.enqueue(entity_type="team", entity_key=f"{code}:{api_id}", reason="bulk_enqueue")
                counts["team"] += 1

            # Players discovered via stats OR snapshots
            from ..models import FixturePlayerStats  # local import to avoid cycle
            player_from_fixtures = session.scalars(
                select(Player.api_football_id).distinct()
                .join(FixturePlayerStats, FixturePlayerStats.player_id == Player.id)
                .join(Fixture, Fixture.id == FixturePlayerStats.fixture_id)
                .where(Fixture.competition_id == comp.id)
            ).all()
            player_from_snapshots = session.scalars(
                select(Player.api_football_id).distinct()
                .join(PlayerFeatureSnapshot, PlayerFeatureSnapshot.player_id == Player.id)
                .join(Season, Season.id == PlayerFeatureSnapshot.season_id)
                .where(Season.competition_id == comp.id)
            ).all()
            player_api_ids = {x for x in (list(player_from_fixtures) + list(player_from_snapshots)) if x is not None}
            for api_id in sorted(player_api_ids):
                svc.enqueue(entity_type="player", entity_key=f"{code}:{api_id}", reason="bulk_enqueue")
                counts["player"] += 1

            fixture_ids = session.scalars(
                select(Fixture.api_football_id).where(Fixture.competition_id == comp.id)
            ).all()
            for api_id in fixture_ids:
                if api_id is None:
                    continue
                svc.enqueue(entity_type="fixture", entity_key=str(api_id), reason="bulk_enqueue")
                counts["fixture"] += 1
    return counts


def enqueue_after_fixture_change(fixture_api_id: int, *, reason: str = "fixture_change") -> int:
    """Helper for the sync pipeline: enqueue fixture + both its teams."""
    svc = KBService(repo=KBDocumentRepository(), doc_builder=build_docs_for_entity)
    enqueued = 0
    with session_scope() as session:
        fixture = session.scalar(select(Fixture).where(Fixture.api_football_id == fixture_api_id))
        if fixture is None:
            return 0
        competition = session.get(Competition, fixture.competition_id)
        code = competition.code if competition else None
        svc.enqueue(entity_type="fixture", entity_key=str(fixture_api_id), reason=reason)
        enqueued += 1
        if code:
            home = session.get(Team, fixture.home_team_id)
            away = session.get(Team, fixture.away_team_id)
            if home:
                svc.enqueue(entity_type="team", entity_key=f"{code}:{home.api_football_id}", reason=reason)
                enqueued += 1
            if away:
                svc.enqueue(entity_type="team", entity_key=f"{code}:{away.api_football_id}", reason=reason)
                enqueued += 1
    return enqueued
