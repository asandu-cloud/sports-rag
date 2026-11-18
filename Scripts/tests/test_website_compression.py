"""The website gzips what it sends when the browser asks (D127): the API's JSON and the static files, but never a
small answer or the live event stream."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from starlette.middleware.gzip import GZipMiddleware

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Scripts" / "rag_ingest"))
sys.path.insert(0, str(ROOT / "Scripts"))

from web_app import api  # noqa: E402
from web_app.routers import home  # noqa: E402

GZIP, PLAIN = {"Accept-Encoding": "gzip, deflate"}, {"Accept-Encoding": "identity"}


def _reel(wins=30):
    win = lambda i: {"event_id": str(1000 + i), "home": f"Home {i}", "away": f"Away {i}", "market": "goals",
                     "side": "over", "line": 2.5, "label": "Over 2.5 goals", "odds": 1.85,
                     "kickoff": "2026-11-21T15:00:00Z"}
    return {"version": "home_wins.v1", "week": {"from": "2026-11-16", "to": "2026-11-22"}, "record": [],
            "wins": [win(i) for i in range(wins)]}


def test_json_is_gzipped_when_the_browser_asks_and_plain_when_not(tmp_path, monkeypatch):
    reel = _reel()
    (tmp_path / "home_wins.json").write_text(json.dumps(reel))
    monkeypatch.setattr(home, "WINS_PATH", tmp_path / "home_wins.json")
    client = TestClient(api.app)

    packed, plain = client.get("/api/home/wins", headers=GZIP), client.get("/api/home/wins", headers=PLAIN)
    assert packed.headers["content-encoding"] == "gzip" and "Accept-Encoding" in packed.headers["vary"]
    assert int(packed.headers["content-length"]) < int(plain.headers["content-length"]) / 3
    assert "content-encoding" not in plain.headers
    assert packed.json() == plain.json() == reel

    monkeypatch.setattr(home, "WINS_PATH", tmp_path / "missing.json")  # the empty reel: under 1 KB, sent as it is
    assert "content-encoding" not in client.get("/api/home/wins", headers=GZIP).headers


def test_static_files_are_gzipped_when_asked_and_keep_their_etag_and_cache_rules():
    client = TestClient(api.app)
    source = (ROOT / "Scripts" / "web_app" / "static" / "app.js").read_bytes()

    with client.stream("GET", "/static/app.js", headers=GZIP) as packed:
        wire = b"".join(packed.iter_raw())
    plain = client.get("/static/app.js", headers=PLAIN)
    assert packed.headers["content-encoding"] == "gzip" and gzip.decompress(wire) == source
    assert len(wire) < len(source) / 2
    assert "content-encoding" not in plain.headers and plain.content == source
    for header in ("etag", "last-modified", "cache-control"):
        assert packed.headers.get(header) == plain.headers.get(header)
    unchanged = client.get("/static/app.js", headers={**GZIP, "If-None-Match": plain.headers["etag"]})
    assert unchanged.status_code == 304 and not unchanged.content

    shell = client.get("/app", headers=GZIP)  # the shell is still revalidated on every visit
    assert shell.headers["content-encoding"] == "gzip" and shell.headers["cache-control"] == "no-cache"


def test_the_live_event_stream_is_never_packed():
    """Packing an event stream would hold its events back until a block fills (``/api/live/stream``)."""
    (packer,) = [entry for entry in api.app.user_middleware if entry.cls is GZipMiddleware]
    events = FastAPI()

    @events.get("/stream")
    def stream():
        return StreamingResponse(iter(["event: heartbeat\ndata: {}\n\n"] * 100), media_type="text/event-stream")

    events.add_middleware(packer.cls, *packer.args, **packer.kwargs)
    response = TestClient(events).get("/stream", headers=GZIP)
    assert "content-encoding" not in response.headers and response.text.count("event: heartbeat") == 100
