"""API tests for the consolidated live router.

Uses FastAPI's TestClient against a small app that mounts only the live
router — this avoids pulling in the whole web_app/api.py dependency tree
(engine modules, Chroma, OpenAI) just to verify the router wiring.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest


def test_live_router_mounted(settings, engine, session_factory, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    # Route platform services to the test DB
    import data_platform.db as db_module
    monkeypatch.setattr(db_module, "session_scope", session_factory)
    from data_platform.repositories import live as live_mod
    monkeypatch.setattr(live_mod, "session_scope", session_factory)

    from web_app.routers.live import router, _NoopPoller
    # Force the router to use the noop poller (no external live stack)
    from web_app.routers import live as live_router_module
    monkeypatch.setattr(live_router_module, "_poller", _NoopPoller())

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    resp = client.get("/api/live/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["running"] is False

    resp = client.get("/api/live/fixtures")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0

    # Seed a fixture + snapshot via the platform and confirm the /alerts and
    # fallback /fixtures/{id} paths read it back.
    from data_platform.services import LiveService
    svc = LiveService()
    svc.ensure_fixture(fixture_id=777, league="EPL", home_team="A", away_team="B", kickoff_utc=None)
    svc.record_model_snapshot(
        fixture_id=777,
        snapshot_payload={"model": {"goals": 2.3}},
        snapshot_ts="2026-04-14T15:00:00+00:00",
        elapsed_min=30.0, score="0-0",
    )
    svc.record_alerts(
        fixture_id=777, snapshot_ts="2026-04-14T15:00:00+00:00",
        alerts=[{"type": "goals_over", "severity": "high", "message": "value"}],
    )

    # GET /api/live/fixtures falls back to DB rows when poller has none
    resp = client.get("/api/live/fixtures")
    assert resp.status_code == 200
    assert resp.json()["count"] == 1

    resp = client.get("/api/live/fixtures/777")
    assert resp.status_code == 200
    assert resp.json()["source"] == "db_snapshot"

    resp = client.get("/api/live/fixtures/777/alerts")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["alerts"][0]["type"] == "goals_over"
