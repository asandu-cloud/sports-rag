"""Vector backend abstraction + KB refresher using the InMemory backend."""

from __future__ import annotations

import pytest


def test_inmemory_backend_round_trip():
    from data_platform.vectors import InMemoryVectorBackend
    backend = InMemoryVectorBackend()
    result = backend.upsert([
        {"id": "t1", "text": "Arsenal snapshot", "metadata": {"team": "Arsenal", "league": "EPL"}},
        {"id": "t2", "text": "City snapshot", "metadata": {"team": "City", "league": "EPL"}},
    ])
    assert result.upserted == 2
    assert "t1" in backend and "t2" in backend

    # Filter query
    hits = backend.query("arsenal", limit=5, where={"team": "Arsenal"})
    assert [h["id"] for h in hits] == ["t1"]

    removed = backend.delete(["t1"])
    assert removed == 1
    assert "t1" not in backend


def test_factory_returns_inmemory_when_env_set(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "inmemory")
    from data_platform.vectors import build_backend, InMemoryVectorBackend
    assert isinstance(build_backend(), InMemoryVectorBackend)


def test_refresher_uses_backend(session_factory, monkeypatch):
    """End-to-end: enqueue an entity, inject a doc builder, assert the
    configured vector backend received the upsert."""
    import data_platform.db as db_module
    monkeypatch.setattr(db_module, "session_scope", session_factory)
    from data_platform.repositories import kb as kb_mod
    monkeypatch.setattr(kb_mod, "session_scope", session_factory)

    from data_platform.repositories.kb import KBDocumentRepository
    from data_platform.services.kb import KBService
    from data_platform.vectors import InMemoryVectorBackend

    backend = InMemoryVectorBackend()

    def doc_builder(entity_type, entity_key):
        return [{
            "doc_id": "team:1:profile",
            "entity_type": "team", "entity_id": "42",
            "league": "EPL", "season": "2025/26", "doc_type": "team_profile",
            "text": "Arsenal snapshot v1",
            "metadata": {"goals_for_pm": 2.1},
        }]

    def upserter(docs):
        return backend.upsert([
            {"id": d["id"], "text": d["text"], "metadata": d.get("metadata") or {}}
            for d in docs
        ]).upserted

    repo = KBDocumentRepository()
    service = KBService(repo=repo, doc_builder=doc_builder, chroma_upserter=upserter)
    service.enqueue(entity_type="team", entity_key="EPL:42")
    stats = service.refresh(batch_size=10)
    assert stats["docs_embedded"] == 1
    assert "team:1:profile" in backend


def test_fixture_document_id_migration_rekeys_and_invalidates_vector_sync(session_factory, monkeypatch):
    import data_platform.db as db_module
    monkeypatch.setattr(db_module, "session_scope", session_factory)
    from data_platform.repositories import kb as kb_mod
    monkeypatch.setattr(kb_mod, "session_scope", session_factory)

    from data_platform.kb.doc_builders import make_doc_id
    from data_platform.kb.refresher import migrate_fixture_document_ids
    from data_platform.repositories.kb import KBDocumentRepository
    from data_platform.vectors import InMemoryVectorBackend

    repo = KBDocumentRepository()
    old_id = "legacy-repeat-matchup"
    repo.upsert(
        doc_id=old_id,
        entity_type="team",
        entity_id="41",
        league="Championship",
        season="2025/26",
        doc_type="team_fixture",
        text="Middlesbrough vs Southampton",
        metadata={"fixture_api_id": 1386855, "team_id": 41},
    )
    repo.mark_synced(doc_ids=[old_id])
    backend = InMemoryVectorBackend()
    backend.upsert([{"id": old_id, "text": "legacy", "metadata": {}}])

    result = migrate_fixture_document_ids(vector_backend=backend)
    new_id = make_doc_id(["Championship", "2025/26", "team_fixture", 1386855, 41])

    assert result == {"scanned": 1, "rekeyed": 1, "skipped": 0, "deleted_vectors": 1}
    assert old_id not in backend
    assert repo.get_by_doc_id(old_id) is None
    assert repo.get_by_doc_id(new_id)["chroma_synced_hash"] is None
