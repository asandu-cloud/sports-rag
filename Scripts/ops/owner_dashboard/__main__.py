"""Start the owner dashboard on http://127.0.0.1:8300/ (tracker D67). Stop it with Ctrl-C.

    python -m Scripts.ops.owner_dashboard [--port 8300] [--db Index/owner_dashboard.db] [--no-update]

It updates its own file first (skipped while the daily refresh holds the data gate), then serves the Report and
the History from that file. One thread: requests, and "Update now", run one at a time.
"""
from __future__ import annotations

import argparse
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from Scripts.data_platform.services.refresh_coordination import RefreshBusy, data_access
from Scripts.ops.odds_archive import ROOT
from Scripts.ops.owner_dashboard import pages, store, sync

HOST = "127.0.0.1"
PORT = 8300
CSS = Path(__file__).with_name("dashboard.css")
FONTS = ROOT / "Scripts" / "web_app" / "static" / "fonts"
FONT_FILES = ("archivo-latin.woff2", "archivo-latin-ext.woff2")
_update_lock = threading.Lock()


def platform_path() -> Path:
    """The database the data gate guards (the same ``DATABASE_URL`` the jobs use)."""
    from sqlalchemy.engine import make_url
    from Scripts.data_platform import config
    return Path(make_url(config.SETTINGS.database_url).database).resolve()


def _meta(conn, key):
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _set_meta(conn, key, value):
    with conn:
        conn.execute("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = "
                     "excluded.value", (key, value))


def update(conn, platform, *, now=None, gate=data_access, load_rows=sync._recent_published) -> dict:
    """Log new Most likely picks, refresh the Value picks, then grade (so a Value match gets its facts in the same
    update). A busy gate skips what is left of that step; ``updated_at`` only moves when nothing was skipped."""
    with _update_lock:
        now = now or datetime.now(timezone.utc)
        out = {"most_likely": sync.log_most_likely(conn, platform, now=now, gate=gate)}
        try:
            out["value"] = dict(sync.log_value(conn, load_rows=load_rows, today=now.date()), skipped_busy=False)
        except RefreshBusy:
            out["value"] = {"picks": 0, "skipped_busy": True}
        out["grade"] = sync.grade(conn, platform, now=now, gate=gate)
        out["busy"] = any(part["skipped_busy"] for part in out.values())
        _set_meta(conn, "skipped_at" if out["busy"] else "updated_at", sync._iso(now.isoformat()))
        return out


def _notice(conn):
    skipped, updated = _meta(conn, "skipped_at"), _meta(conn, "updated_at")
    if skipped and (not updated or skipped > updated):
        moment = pages.local(skipped)
        return (f"The update at {moment:%H:%M} on {pages._day(moment)} was skipped: the daily refresh was using the "
                f"database. These pages show what was saved before. Try again once it finishes.")
    return None


def route(conn, path: str, query: dict):
    """(status, content type, body) for a GET."""
    page = dict(updated=_meta(conn, "updated_at"), notice=_notice(conn))
    if path == "/":
        body = pages.report_page(conn, kind=query.get("kind") or "most_likely", market=query.get("market"), **page)
    elif path == "/history":
        body = pages.history_page(conn, day=query.get("day"), **page)
    elif path == "/dashboard.css":
        return 200, "text/css; charset=utf-8", CSS.read_bytes()
    elif path.startswith("/fonts/") and path[len("/fonts/"):] in FONT_FILES and (FONTS / path[7:]).is_file():
        return 200, "font/woff2", (FONTS / path[7:]).read_bytes()
    else:
        return 404, "text/plain; charset=utf-8", b"Not found"
    return 200, "text/html; charset=utf-8", body.encode()


def make_server(conn, port: int = PORT, *, run_update) -> HTTPServer:
    """A one-thread server on 127.0.0.1 that answers only requests addressed to this machine."""
    class Handler(BaseHTTPRequestHandler):
        def _local(self):
            port_now = self.server.server_address[1]
            allowed = {f"127.0.0.1:{port_now}", f"localhost:{port_now}"}
            origin = self.headers.get("Origin")
            return self.headers.get("Host") in allowed and (origin is None or urlsplit(origin).netloc in allowed
                                                            and urlsplit(origin).scheme == "http")

        def _send(self, status, kind, body, location=None):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if location:
                self.send_header("Location", location)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._local():
                return self._send(403, "text/plain; charset=utf-8", b"Forbidden")
            parts = urlsplit(self.path)
            query = {key: values[0] for key, values in parse_qs(parts.query).items()}
            self._send(*route(conn, parts.path, query))

        def do_POST(self):
            if not self._local() or urlsplit(self.path).path != "/update":
                return self._send(403, "text/plain; charset=utf-8", b"Forbidden")
            length = min(int(self.headers.get("Content-Length") or 0), 4096)
            back = (parse_qs(self.rfile.read(length).decode("utf-8", "replace")).get("back") or ["/"])[0]
            if not back.startswith("/") or back.startswith("//"):
                back = "/"
            run_update()
            self._send(303, "text/plain; charset=utf-8", b"", location=back)

        def log_message(self, format, *args):
            pass

    return HTTPServer((HOST, port), Handler)


def _say(result) -> str:
    likely, value, grade = result["most_likely"], result["value"], result["grade"]
    return (f"{likely['logged']} matches logged ({likely['picks']} Most likely picks), {value['picks']} Value picks "
            f"refreshed, {grade['graded']} graded, {grade['pending']} pending"
            + ("; the daily refresh held the data gate, so part of it was skipped" if result["busy"] else ""))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--db", type=Path, default=store.DEFAULT_PATH)
    parser.add_argument("--no-update", action="store_true", help="serve what is saved, without reading platform.db")
    args = parser.parse_args(argv)
    conn = store.connect(args.db)
    platform = platform_path()

    def run_update():
        print(f"Updated: {_say(update(conn, platform))}", flush=True)

    if not args.no_update:
        print("Updating from platform.db (read only)…", flush=True)
        run_update()
    httpd = make_server(conn, args.port, run_update=run_update)
    print(f"Owner dashboard: http://{HOST}:{args.port}/  (Ctrl-C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
