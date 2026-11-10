"""The stats questions prototype, separate from the website (tracker D92). Stop the server with Ctrl-C.

    python -m Scripts.ops.stats_questions build      rebuild Index/stats_questions/stats.db from platform.db (gated)
    python -m Scripts.ops.stats_questions serve      http://127.0.0.1:8400/ [--port 8400]
    python -m Scripts.ops.stats_questions ask "How often do Brentford's away games go over 9.5 corners?"
    python -m Scripts.ops.stats_questions choose line_record team=Brentford stat=corners side=match line=9.5

Only "ask" and the page's Ask and Free SQL (in words) call OpenAI (gpt-6-luna, about $0.0004 a question). "choose" and
typed SQL need no credit. Every question goes in Index/stats_questions/log.db; nothing here writes to platform.db.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlsplit

from Scripts.data_platform.services.refresh_coordination import RefreshBusy
from Scripts.ops.odds_archive import ROOT
from Scripts.ops.stats_questions import free_sql, log, model, questions, store

HOST = "127.0.0.1"
PORT = 8400
SITE_DAILY_LIMIT = 2000  # model questions a day across the whole site (D91)
PAGE = Path(__file__).with_name("page.html")
FONTS = ROOT / "Scripts" / "web_app" / "static" / "fonts"
FONT_FILES = ("archivo-latin.woff2", "archivo-latin-ext.woff2")
JSON = "application/json; charset=utf-8"
MAX_BODY = 16_384


@dataclass
class App:
    stats_path: Path = store.DEFAULT_PATH
    log: sqlite3.Connection | None = None
    ask: Callable = field(default=model.ask)
    write_sql: Callable = field(default=model.write_sql)
    daily_limit: int = SITE_DAILY_LIMIT


class Refusal(Exception):
    def __init__(self, status, kind, message):
        super().__init__(message)
        self.status, self.kind, self.message = status, kind, message


def _json(status, payload):
    return status, JSON, json.dumps(payload, ensure_ascii=False).encode()


def _stats(app):
    if not Path(app.stats_path).is_file():
        raise Refusal(503, "no_stats_file", "There is no stats file yet. Build it with: python -m "
                                            "Scripts.ops.stats_questions build")
    return store.connect(app.stats_path)


def _under_limit(app):
    if log.asked_today(app.log) >= app.daily_limit:
        raise Refusal(429, "site_limit", f"The site has reached its {app.daily_limit:,} questions for today.")


def _summary(conn):
    """The trial's measure: only questions the model read can be right or wrong (hand checks are left out)."""
    row = conn.execute(f"SELECT count(*), sum(verdict = 'right'), sum(verdict = 'wrong'), sum(verdict IS NULL), "
                       f"sum(cost_usd) FROM questions WHERE mode IN ({', '.join('?' * len(log.MODEL_MODES))})",
                       log.MODEL_MODES).fetchone()
    return {"questions": row[0], "right": row[1] or 0, "wrong": row[2] or 0, "unjudged": row[3] or 0,
            "cost_usd": round(row[4] or 0, 6)}


def _status(app):
    out = {"model": os.environ.get("STATS_QUESTIONS_MODEL", "").strip() or model.MODEL,
           "openai_key": "set" if os.environ.get("OPENAI_API_KEY", "").strip() else "not set",
           "asked_today": log.asked_today(app.log), "daily_limit": app.daily_limit, "stats": None}
    if Path(app.stats_path).is_file():
        conn = store.connect(app.stats_path)
        try:
            out["stats"] = dict(conn.execute("SELECT key, value FROM build").fetchall())
            out["stats"]["data_up_to_label"] = questions._data_up_to(conn)
        finally:
            conn.close()
    return out


def _types(app):
    conn = _stats(app)
    try:
        comps = [dict(r) for r in conn.execute("SELECT code, name FROM competitions")]
    finally:
        conn.close()
    return {"types": questions.TYPES, "defaults": questions.DEFAULTS, "competitions": comps}


def _get(app, path, query):
    if path == "/":
        return 200, "text/html; charset=utf-8", PAGE.read_bytes()
    if path.startswith("/fonts/") and path[7:] in FONT_FILES and (FONTS / path[7:]).is_file():
        return 200, "font/woff2", (FONTS / path[7:]).read_bytes()
    if path == "/api/types":
        return _json(200, _types(app))
    if path == "/api/status":
        return _json(200, _status(app))
    if path == "/api/log":
        limit = query.get("limit", "100")
        limit = int(limit) if limit.isdigit() else 100
        return _json(200, {"entries": log.recent(app.log, min(limit, 500)), "summary": _summary(app.log)})
    return 404, "text/plain; charset=utf-8", b"Not found"


def _ask(app, body):
    _under_limit(app)
    question = body.get("question") if isinstance(body.get("question"), str) else ""
    conn = _stats(app)
    try:
        out = app.ask(conn, question, previous=body.get("previous"))
    except model.ModelError as exc:
        log.record(app.log, mode="ask", question=question[:model.MAX_SQL_CHARS], error=exc.kind)
        raise Refusal(exc.status, exc.kind, exc.message) from None
    finally:
        conn.close()
    out["log_id"] = log.record(app.log, mode="ask", question=out["question"], calls=out["calls"],
                               answers=out["answers"], model=out["model"], ms=out["ms"], usage=out["usage"],
                               cost_usd=out["cost_usd"], flagged_bet=out["flagged_bet"])
    return out


def _choose(app, body):
    kind, params = body.get("type"), body.get("params") if isinstance(body.get("params"), dict) else {}
    conn = _stats(app)
    try:
        card = questions.answer(conn, kind if isinstance(kind, str) else "", params)
    finally:
        conn.close()
    calls = [{"type": kind, "params": params}]
    return {"calls": calls, "answers": [card],
            "log_id": log.record(app.log, mode="choose", calls=calls, answers=[card])}


def _sql(app, body):
    question = body.get("question") if isinstance(body.get("question"), str) else None
    written = None
    if question is not None:
        _under_limit(app)
        conn = _stats(app)
        try:
            written = app.write_sql(conn, question)
        except model.ModelError as exc:
            log.record(app.log, mode="sql_ask", question=question[:model.MAX_SQL_CHARS], error=exc.kind)
            raise Refusal(exc.status, exc.kind, exc.message) from None
        finally:
            conn.close()
        sql = written["sql"]
    else:
        sql = body.get("sql") if isinstance(body.get("sql"), str) else ""
        _stats(app).close()
    result = free_sql.run(app.stats_path, sql)
    calls = [{"type": "sql", "params": {"sql": result["sql"]}}]
    extra = {k: written[k] for k in ("question", "title", "model", "usage", "cost_usd")} if written else {}
    result.update(extra)
    result["log_id"] = log.record(
        app.log, mode="sql_ask" if written else "sql", question=question, calls=calls, answers=[result],
        model=extra.get("model"), ms=(written or {}).get("ms"), usage=extra.get("usage"),
        cost_usd=extra.get("cost_usd"), error=(result.get("error") or {}).get("kind"))
    return result


def _verdict(app, body):
    entry_id = body.get("id")
    if not isinstance(entry_id, int) or body.get("verdict") not in log.VERDICTS:
        raise Refusal(400, "bad_verdict", "Send an id and a verdict of right or wrong.")
    note = body.get("note") if isinstance(body.get("note"), str) else None
    if not log.set_verdict(app.log, entry_id, body["verdict"], note):
        raise Refusal(404, "not_found", "No question with that id.")
    return {"ok": True}


POSTS = {"/api/ask": _ask, "/api/choose": _choose, "/api/sql": _sql, "/api/verdict": _verdict}


def handle(app: App, method: str, path: str, query: dict, body) -> tuple[int, str, bytes]:
    """(status, content type, body) for one request. POST bodies are JSON objects."""
    try:
        if method == "GET":
            return _get(app, path, query)
        if method == "POST" and path in POSTS:
            return _json(200, POSTS[path](app, body if isinstance(body, dict) else {}))
        return 404, "text/plain; charset=utf-8", b"Not found"
    except Refusal as exc:
        return _json(exc.status, {"error": {"kind": exc.kind, "message": exc.message}})


def make_server(app: App, port: int = PORT) -> HTTPServer:
    """A one-thread server on 127.0.0.1 that answers only requests addressed to this machine."""
    class Handler(BaseHTTPRequestHandler):
        def _local(self):
            port_now = self.server.server_address[1]
            allowed = {f"127.0.0.1:{port_now}", f"localhost:{port_now}"}
            origin = self.headers.get("Origin")
            return self.headers.get("Host") in allowed and (origin is None or urlsplit(origin).netloc in allowed
                                                            and urlsplit(origin).scheme == "http")

        def _send(self, status, kind, body):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _route(self, method, body=None):
            if not self._local():
                return self._send(403, "text/plain; charset=utf-8", b"Forbidden")
            parts = urlsplit(self.path)
            query = {key: values[0] for key, values in parse_qs(parts.query).items()}
            self._send(*handle(app, method, parts.path, query, body))

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._send(413, "text/plain; charset=utf-8", b"Too large")
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                body = None
            self._route("POST", body)

        def log_message(self, format, *args):
            pass

    return HTTPServer((HOST, port), Handler)


def _load_env():
    """The OpenAI key, from the project's .env (loaded into this process only; never printed)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def _value(text):
    lowered = text.lower()
    if lowered in ("null", "none"):
        return None
    if lowered in ("true", "false"):
        return lowered == "true"
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def _show(out):
    for call, card in zip(out["calls"], out["answers"]):
        print(f"\n{call['type']} {json.dumps(call['params'], ensure_ascii=False)}")
        print(json.dumps(card, ensure_ascii=False, indent=2))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", type=Path, default=store.DEFAULT_PATH)
    parser.add_argument("--log", type=Path, default=log.DEFAULT_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build")
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=PORT)
    ask = commands.add_parser("ask")
    ask.add_argument("question")
    choose = commands.add_parser("choose")
    choose.add_argument("type", choices=sorted(questions.TYPES))
    choose.add_argument("params", nargs="*", help="name=value pairs")
    args = parser.parse_args(argv)

    if args.command == "build":
        try:
            report = store.build(args.stats)
        except RefreshBusy:
            print("The daily refresh is using the database. Try again once it finishes.")
            return 1
        print(json.dumps(report, indent=2))
        return 0
    app = App(stats_path=args.stats, log=log.connect(args.log))
    if args.command == "serve":
        _load_env()
        httpd = make_server(app, args.port)
        print(f"Stats questions: http://{HOST}:{args.port}/ (Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()
        return 0
    if args.command == "ask":
        _load_env()
        status, kind, payload = handle(app, "POST", "/api/ask", {}, {"question": args.question})
    else:
        params = dict(pair.split("=", 1) for pair in args.params if "=" in pair)
        status, kind, payload = handle(app, "POST", "/api/choose", {}, {
            "type": args.type, "params": {k: _value(v) for k, v in params.items()}})
    out = json.loads(payload)
    if status != 200:
        print(out["error"]["message"])
        return 1
    _show(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
