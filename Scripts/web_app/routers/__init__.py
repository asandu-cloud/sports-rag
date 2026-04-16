"""Modular FastAPI routers mounted by ``web_app.api``.

V2 consolidation: live-match endpoints moved here from the standalone
``Scripts/live/live_api.py`` process. Each router is self-contained; the
live process can still import one of them and run on port 8001.
"""
