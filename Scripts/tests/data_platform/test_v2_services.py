"""V2 service + KB delta-refresh tests."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest


@pytest.fixture()
def monkeypatched_factories(session_factory, monkeypatch):
    """Swap every default ``session_scope`` the service layer grabs with
    the test's isolated factory.
    """
    import data_platform.db as db_module
    monkeypatch.setattr(db_module, "session_scope", session_factory)
    from data_platform.repositories import (
        kb as kb_mod,
        live as live_mod,
        parlays as parlays_mod,
        predictions as pred_mod,
    )
    monkeypatch.setattr(pred_mod, "session_scope", session_factory)
    monkeypatch.setattr(parlays_mod, "session_scope", session_factory)
    monkeypatch.setattr(live_mod, "session_scope", session_factory)
    monkeypatch.setattr(kb_mod, "session_scope", session_factory)
    return session_factory


def test_prediction_service_matches_repo(monkeypatched_factories):
    from data_platform.services import PredictionService
    svc = PredictionService()
    pid = svc.log(
        home_team="A", away_team="B", league="EPL",
        market="goals", pick="Over 2.5", side="over", odds=2.0,
        model_prob=0.55, confidence="medium",
    )
    assert pid > 0
    svc.mark_outcome(pid, outcome="win", actual_result=3)
    tr = svc.track_record()
    assert tr["total"] == 1
    assert tr["wins"] == 1


def test_live_service_list_active(monkeypatched_factories):
    from data_platform.services import LiveService
    svc = LiveService()
    svc.ensure_fixture(fixture_id=1, league="EPL", home_team="A", away_team="B", kickoff_utc=None)
    active = svc.list_active()
    assert len(active) == 1
    assert active[0]["fixture_id"] == 1


def test_kb_service_refresh_delta(monkeypatched_factories):
    """End-to-end: enqueue one entity → builder generates two docs → upserter
    sees exactly the changed docs → queue item completed.
    """
    from data_platform.services import KBService
    from data_platform.repositories import KBDocumentRepository

    generations = [
        [
            {
                "doc_id": "team:1:profile",
                "entity_type": "team", "entity_id": "42",
                "league": "EPL", "season": "2025/26", "doc_type": "team_profile",
                "text": "Arsenal snapshot v1",
                "metadata": {"goals_for_pm": 2.1},
            },
            {
                "doc_id": "team:1:recent",
                "entity_type": "team", "entity_id": "42",
                "league": "EPL", "season": "2025/26", "doc_type": "team_fixture",
                "text": "Arsenal vs Spurs 2-1",
                "metadata": {"goals": 2},
            },
        ],
        [
            {  # content unchanged
                "doc_id": "team:1:profile",
                "entity_type": "team", "entity_id": "42",
                "league": "EPL", "season": "2025/26", "doc_type": "team_profile",
                "text": "Arsenal snapshot v1",
                "metadata": {"goals_for_pm": 2.1},
            },
            {  # content changed
                "doc_id": "team:1:recent",
                "entity_type": "team", "entity_id": "42",
                "league": "EPL", "season": "2025/26", "doc_type": "team_fixture",
                "text": "Arsenal vs Spurs 3-1",
                "metadata": {"goals": 3},
            },
        ],
    ]

    calls: list = []

    def doc_builder(entity_type, entity_key):
        return generations.pop(0)

    def upserter(docs):
        calls.append([d["id"] for d in docs])
        return len(docs)

    repo = KBDocumentRepository()
    service = KBService(repo=repo, doc_builder=doc_builder, chroma_upserter=upserter)

    # First refresh: both docs go through because neither was synced before
    service.enqueue(entity_type="team", entity_key="EPL:42")
    result_1 = service.refresh(batch_size=10)
    assert result_1["dequeued"] == 1
    assert result_1["docs_changed"] == 2
    assert result_1["docs_embedded"] == 2
    assert len(calls) == 1
    assert set(calls[0]) == {"team:1:profile", "team:1:recent"}

    # Second refresh after re-enqueue: only the changed doc should embed
    service.enqueue(entity_type="team", entity_key="EPL:42")
    result_2 = service.refresh(batch_size=10)
    assert result_2["dequeued"] == 1
    assert result_2["docs_changed"] == 1
    assert result_2["docs_embedded"] == 1
    assert len(calls) == 2
    assert calls[1] == ["team:1:recent"]


def test_shim_routes_to_platform_when_enabled(monkeypatch, settings, engine, session_factory):
    """ParlaySessionStore should route through PlatformParlayStore when
    PLATFORM_ENABLED=1 and the default db_path is used.
    """
    import os
    import sys
    os.environ["PLATFORM_ENABLED"] = "1"

    # Ensure the shim sees the test session factory
    import data_platform.db as db_module
    monkeypatch.setattr(db_module, "session_scope", session_factory)
    from data_platform.repositories import parlays as parlays_mod
    monkeypatch.setattr(parlays_mod, "session_scope", session_factory)

    # Clear the cached parlay service + settings
    from data_platform.compat import parlay_shim as shim
    shim.get_parlay_service.cache_clear()
    from data_platform import config as config_mod
    from data_platform.config import load_settings
    config_mod.SETTINGS = load_settings()

    # Import the legacy ParlaySessionStore fresh so the shim bootstrap runs.
    sys.path.insert(0, str(settings.project_root / "Scripts"))
    if "rag_ingest.core.parlay_service" in sys.modules:
        del sys.modules["rag_ingest.core.parlay_service"]
    from rag_ingest.core.parlay_service import ParlaySessionStore

    store = ParlaySessionStore()
    assert store._platform is not None, "shim should pick up PLATFORM_ENABLED=1"

    class _Stub:
        def to_dict(self):
            return {
                "parent_session_id": None,
                "discord_user_id": 1, "guild_id": 2, "channel_id": 3, "thread_id": 4,
                "source_command": "parlay", "action_type": "build",
                "target_date": "2026-04-14",
            }

    session_id = store.create_session(_Stub(), _Stub())
    assert session_id > 0
