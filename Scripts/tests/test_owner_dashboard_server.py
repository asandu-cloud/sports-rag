"""The owner dashboard's server (tracker D67): routes, "Update now", and a localhost-only bind."""
import http.client
import threading
from datetime import datetime, timezone

import pytest

from Scripts.data_platform.services.refresh_coordination import RefreshBusy
from Scripts.ops.owner_dashboard import __main__ as server, store, sync


@pytest.fixture
def conn(tmp_path):
    connection = store.connect(tmp_path / "owner_dashboard.db")
    yield connection
    connection.close()


def meta(conn, key):
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def test_routes_serve_both_pages_the_stylesheet_and_404(conn):
    status, kind, body = server.route(conn, "/", {"kind": "value"})
    assert (status, kind) == (200, "text/html; charset=utf-8") and b"No settled Value picks yet" in body
    status, kind, body = server.route(conn, "/history", {"day": "2026-11-03"})
    assert status == 200 and b"No played matches with picks yet" in body
    status, kind, body = server.route(conn, "/dashboard.css", {})
    assert (status, kind) == (200, "text/css; charset=utf-8") and b"--chart-1: #0A6D37" in body
    for path in ("/nope", "/fonts/../../.env", "/fonts/other.woff2", "/static/style.css"):
        assert server.route(conn, path, {})[0] == 404


def test_the_report_route_opens_a_league(conn):
    conn.execute("INSERT INTO matches (fixture_api_id, league, kickoff, status) VALUES ('1', 'EPL', "
                 "'2026-11-07T15:00:00Z', 'FT')")
    conn.execute("INSERT INTO most_likely (fixture_api_id, position, market, side, line, label, chance, version, "
                 "outcome) VALUES ('1', 0, 'goals', 'over', 1.5, 'Over 1.5 goals', 0.8, 'recommendations.v1', 'hit')")
    body = server.route(conn, "/", {"league": "EPL"})[2].decode()
    assert "· All markets · Premier League</span>" in body


def test_update_now_runs_one_update_and_a_busy_refresh_is_reported_not_raised(conn, monkeypatch):
    calls, busy = [], {"value": False}
    monkeypatch.setattr(sync, "log_most_likely", lambda c, p, *, now, gate: calls.append("most_likely") or
                        {"logged": 2, "picks": 5, "skipped_busy": False})

    def log_value(c, *, load_rows, today):
        calls.append("value")
        if busy["value"]:
            raise RefreshBusy("refresh running")
        return {"picks": 1}

    monkeypatch.setattr(sync, "log_value", log_value)
    monkeypatch.setattr(sync, "grade", lambda c, p, *, now, gate: calls.append("grade") or
                        {"graded": 5, "pending": 0, "skipped_busy": False})
    first = datetime(2026, 11, 4, 14, 0, tzinfo=timezone.utc)
    result = server.update(conn, "platform.db", now=first)
    assert calls == ["most_likely", "value", "grade"] and result["busy"] is False
    assert meta(conn, "updated_at") == "2026-11-04T14:00:00Z"
    busy["value"] = True
    result = server.update(conn, "platform.db", now=datetime(2026, 11, 4, 15, 0, tzinfo=timezone.utc))
    assert result["busy"] is True and result["value"]["skipped_busy"] is True
    assert calls[3:] == ["most_likely", "value", "grade"]  # the rest of the update still runs, gated
    assert meta(conn, "updated_at") == "2026-11-04T14:00:00Z" and meta(conn, "skipped_at") == "2026-11-04T15:00:00Z"
    page = server.route(conn, "/", {})[2].decode()
    assert "skipped" in page and "daily refresh" in page
    busy["value"] = False
    server.update(conn, "platform.db", now=datetime(2026, 11, 4, 16, 0, tzinfo=timezone.utc))
    assert "skipped" not in server.route(conn, "/", {})[2].decode()


def request(port, method, path, headers=None, body=None):
    client = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    client.request(method, path, body=body, headers=headers or {})
    response = client.getresponse()
    out = response.status, response.getheader("Location"), response.read()
    client.close()
    return out


def test_the_server_binds_localhost_only_and_refuses_other_sites(conn):
    updates, seen = [], {}
    httpd = server.make_server(conn, 0, run_update=lambda: updates.append(1))
    host, port = httpd.server_address[:2]
    form = {"Content-Type": "application/x-www-form-urlencoded"}

    def visit():  # the requests run beside the server; the server keeps the test's thread, as it does in main()
        try:
            seen["page"] = request(port, "GET", "/")[0]
            seen["rebinding"] = request(port, "GET", "/", {"Host": "evil.example"})[0]
            seen["other site"] = request(port, "POST", "/update", dict(form, Origin="https://evil.example"),
                                         "back=/")[0]
            seen["updates before"] = list(updates)
            seen["update"] = request(port, "POST", "/update", dict(form, Origin=f"http://127.0.0.1:{port}"),
                                     "back=%2Fhistory%3Fday%3D2026-11-03")[:2]
            seen["off-site back"] = request(port, "POST", "/update", form, "back=https://evil.example")[1]
        finally:
            httpd.shutdown()

    worker = threading.Thread(target=visit)
    worker.start()
    httpd.serve_forever()
    httpd.server_close()
    worker.join(timeout=5)
    assert host == "127.0.0.1" and not worker.is_alive()
    assert seen == {"page": 200, "rebinding": 403, "other site": 403, "updates before": [],
                    "update": (303, "/history?day=2026-11-03"), "off-site back": "/"}
    assert updates == [1, 1]
