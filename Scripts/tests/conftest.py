"""Shared test setup: the engine's prediction switches start at their defaults.

The engine loads ``.env`` when it is imported, and ``.env`` turns rollout stages on in production
(A3.7, e.g. ``SPIX_COUNT_DISPERSION=fixed`` since 20 Oct 2026). ``load_dotenv`` never overrides a
variable that is already set, so pinning the defaults here, before any engine import, keeps the
tests (the golden bundle above all) independent of what is live. A test that needs a stage sets it
with ``monkeypatch.setenv``. ``DOCUMENT_STORE`` is left alone: it changes where documents are read,
not what is predicted (parity proven, tracker D11).
"""
import os

from Scripts.rag_ingest.core.engine_switches import SWITCHES

for _name, (_default, _accepted) in SWITCHES.items():
    if _name != "DOCUMENT_STORE":
        os.environ[_name] = _default
