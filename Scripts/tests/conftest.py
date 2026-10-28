"""Shared test setup: the engine's prediction switches start at their defaults.

The engine loads ``.env`` when it is imported, and ``.env`` turns rollout stages on in production
(A3.7, e.g. ``SPIX_COUNT_DISPERSION=fixed`` since 20 Oct 2026). ``load_dotenv`` never overrides a
variable that is already set, so pinning the defaults here, before any engine import, keeps the
tests (the golden bundle above all) independent of what is live. A test that needs a stage sets it
with ``monkeypatch.setenv``. ``DOCUMENT_STORE`` is left alone: it changes where documents are read,
not what is predicted (parity proven, tracker D11).
"""
import os
import tempfile

from Scripts.rag_ingest.core.engine_switches import SWITCHES

for _name, (_default, _accepted) in SWITCHES.items():
    if _name != "DOCUMENT_STORE":
        os.environ[_name] = _default

# ``.env`` also holds the owner's private alert chat (Telegram since 28 Oct). Blank, these stay unset when the
# engine loads ``.env``, so no test reaches the chat; a test that sends sets its own. Alert state goes to a
# temporary folder, never ``Index/ops-alerts``.
for _name in ("REFRESH_ALERT_TELEGRAM_TOKEN", "REFRESH_ALERT_TELEGRAM_CHAT_ID", "REFRESH_ALERT_WEBHOOK_URL"):
    os.environ[_name] = ""
os.environ["SPIX_OPS_ALERT_DIR"] = tempfile.mkdtemp(prefix="spix-ops-alerts-")
