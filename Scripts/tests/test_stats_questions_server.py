"""The stats questions prototype's local server (tracker D92): its routes, the log behind them, the site's daily
ceiling, and a localhost-only bind. The model is a stand-in: no test makes a paid call."""
import http.client
import json
import threading

import pytest

from Scripts.ops.stats_questions import __main__ as server, log, model
from Scripts.tests.test_stats_questions import conn  # noqa: F401  (the shared stats file fixture)


@pytest.fixture()
def app(conn, tmp_path):
    asked = []

    def fake_ask(stats, question, *, previous=None):
        asked.append((question, previous))
        return {"question": question, "calls": [{"type": "recent_form", "params": {"team": "Brentford"}}],
                "answers": [{"type": "recent_form", "headline": "Brentford: 1 win"}], "model": "gpt-6-luna",
                "usage": {"input_tokens": 2500, "cached_tokens": 0, "output_tokens": 80}, "cost_usd": 0.00029,
                "ms": 640, "flagged_bet": False}

    def fake_sql(stats, question):
        return {"question": question, "sql": "SELECT name FROM teams ORDER BY name LIMIT 1", "title": "First team",
                "model": "gpt-6-luna", "usage": {"input_tokens": 700, "cached_tokens": 0, "output_tokens": 40},
                "cost_usd": 0.00009, "ms": 500}

    application = server.App(stats_path=tmp_path / "stats.db", log=log.connect(tmp_path / "log.db"), ask=fake_ask,
                             write_sql=fake_sql)
    application.asked = asked
    yield application
    application.log.close()


def call(app, method, path, body=None, query=None):
    status, kind, payload = server.handle(app, method, path, query or {}, body)
    return status, (json.loads(payload) if kind.startswith("application/json") else payload)


def test_the_page_types_and_fonts_are_served_and_nothing_else(app):
    status, page = call(app, "GET", "/")
    assert status == 200 and b"<title>Stats questions" in page
    status, types = call(app, "GET", "/api/types")
    assert set(types["types"]) == set(model.questions.TYPES) and types["competitions"][0]["code"] == "EPL"
    assert call(app, "GET", "/fonts/archivo-latin.woff2")[0] == 200
    for path in ("/nope", "/fonts/../../.env", "/fonts/other.woff2", "/api/ask"):
        assert call(app, "GET", path)[0] == 404


def test_choose_it_yourself_answers_without_the_model_and_is_logged(app):
    status, out = call(app, "POST", "/api/choose", {"type": "line_record", "params": {
        "team": "Brentford", "stat": "corners", "side": "match", "line": 9.5, "venue": "away"}})
    assert status == 200 and "4 of 6" in out["answers"][0]["headline"] and app.asked == []
    entry = log.recent(app.log)[0]
    assert entry["id"] == out["log_id"] and entry["mode"] == "choose" and entry["calls"][0]["type"] == "line_record"


def test_a_question_goes_to_the_model_and_its_cost_is_logged(app):
    previous = {"question": "Brentford form", "calls": []}
    status, out = call(app, "POST", "/api/ask", {"question": "and at home?", "previous": previous})
    assert status == 200 and out["answers"][0]["headline"] == "Brentford: 1 win"
    assert app.asked == [("and at home?", previous)]
    entry = log.recent(app.log)[0]
    assert (entry["mode"], entry["question"], entry["cost_usd"], entry["ms"]) == ("ask", "and at home?", 0.00029, 640)
    assert entry["usage"]["input_tokens"] == 2500 and entry["flagged_bet"] == 0


def test_model_errors_are_logged_and_returned_as_a_plain_kind(app):
    def broke(stats, question, *, previous=None):
        raise model.ModelError("no_credit", 402, "The OpenAI account has no credit left.")

    app.ask = broke
    status, out = call(app, "POST", "/api/ask", {"question": "Brentford corners"})
    assert status == 402 and out["error"] == {"kind": "no_credit", "message": "The OpenAI account has no credit left."}
    assert log.recent(app.log)[0]["error"] == "no_credit"


def test_the_site_stops_at_its_daily_ceiling_before_calling_the_model(app):
    app.daily_limit = 2
    for _ in range(2):
        assert call(app, "POST", "/api/ask", {"question": "Brentford corners"})[0] == 200
    status, out = call(app, "POST", "/api/ask", {"question": "Brentford corners"})
    assert status == 429 and out["error"]["kind"] == "site_limit" and len(app.asked) == 2
    assert call(app, "POST", "/api/sql", {"question": "Name one team"})[0] == 429
    assert call(app, "POST", "/api/choose", {"type": "recent_form", "params": {"team": "Brentford"}})[0] == 200


def test_free_sql_runs_typed_sql_or_the_models_sql_and_shows_it(app):
    status, out = call(app, "POST", "/api/sql", {"sql": "SELECT count(*) AS n FROM teams"})
    assert status == 200 and out["columns"] == ["n"] and out["rows"] == [[6]] and out["sql"].startswith("SELECT")
    status, out = call(app, "POST", "/api/sql", {"question": "Name one team"})
    assert out["rows"] == [["Arsenal"]] and out["title"] == "First team" and out["cost_usd"] == 0.00009
    modes = [e["mode"] for e in log.recent(app.log)]
    assert modes == ["sql_ask", "sql"]
    assert call(app, "POST", "/api/sql", {"sql": "DELETE FROM teams"})[1]["error"]["kind"] == "refused"


def test_the_owner_marks_answers_right_or_wrong_and_the_log_shows_it(app):
    log_id = call(app, "POST", "/api/ask", {"question": "Brentford corners"})[1]["log_id"]
    call(app, "POST", "/api/choose", {"type": "recent_form", "params": {"team": "Brentford"}})  # a hand check
    assert call(app, "POST", "/api/verdict", {"id": log_id, "verdict": "right"})[0] == 200
    assert call(app, "POST", "/api/verdict", {"id": log_id, "verdict": "maybe"})[0] == 400
    assert call(app, "POST", "/api/verdict", {"id": 999, "verdict": "wrong"})[0] == 404
    status, out = call(app, "GET", "/api/log")
    assert out["entries"][1]["verdict"] == "right" and out["summary"] == {  # only what the model read
        "questions": 1, "right": 1, "wrong": 0, "unjudged": 0, "cost_usd": 0.00029}


def test_a_missing_stats_file_says_how_to_build_it(app, tmp_path):
    app.stats_path = tmp_path / "missing.db"
    status, out = call(app, "POST", "/api/choose", {"type": "recent_form", "params": {"team": "Brentford"}})
    assert status == 503 and out["error"]["kind"] == "no_stats_file" and "build" in out["error"]["message"]


def request(port, method, path, headers=None, body=None):
    client = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    client.request(method, path, body=body, headers=headers or {})
    response = client.getresponse()
    out = response.status, response.read()
    client.close()
    return out


def test_the_server_listens_on_localhost_and_refuses_other_hosts_and_origins(app):
    httpd = server.make_server(app, port=0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        assert httpd.server_address[0] == "127.0.0.1"
        assert request(port, "GET", "/api/types")[0] == 200
        assert request(port, "GET", "/", {"Host": "evil.example"})[0] == 403
        body = json.dumps({"type": "recent_form", "params": {"team": "Brentford"}})
        assert request(port, "POST", "/api/choose", {"Origin": "http://evil.example",
                                                     "Content-Type": "application/json"}, body)[0] == 403
        status, payload = request(port, "POST", "/api/choose", {"Content-Type": "application/json"}, body)
        assert status == 200 and json.loads(payload)["answers"][0]["type"] == "recent_form"
    finally:
        httpd.shutdown()
        httpd.server_close()
