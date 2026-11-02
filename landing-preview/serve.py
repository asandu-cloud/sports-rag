"""Local preview of the new landing page, on its own port. Not part of the production website.

    .venv/bin/python -B landing-preview/serve.py    # then open http://127.0.0.1:8010

The page keeps the production paths, so it can move into the site unchanged: the schedule API is read through to the
local website on port 8000, and links to /app, /static/ and /auth/ redirect there.
"""
from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.error import URLError
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
PORT = 8010
SITE = "http://127.0.0.1:8000"
SCHEDULE = re.compile(r"^/api/match-reads/schedule/\d{4}-\d{2}-\d{2}$")
ON_SITE = ("/app", "/static/", "/auth/")


class Preview(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if SCHEDULE.match(path):
            return self._read_through(path)
        if path.endswith(".py"):
            return self.send_error(404)
        if path.startswith(ON_SITE):
            self.send_response(302)
            self.send_header("Location", SITE + self.path)
            return self.end_headers()
        return super().do_GET()

    def _read_through(self, path):
        try:
            with urlopen(SITE + path, timeout=8) as response:
                body = response.read()
        except (URLError, TimeoutError):
            return self.send_error(502, "The local website is not answering")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), partial(Preview, directory=str(HERE)))
    print(f"Landing preview on http://127.0.0.1:{PORT}", flush=True)
    server.serve_forever()
