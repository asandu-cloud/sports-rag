"""Shared test fixtures for data_platform tests.

Uses an in-memory SQLite database per test so runs are fast, isolated,
and don't require a running PostgreSQL instance. JSONB columns degrade
to plain JSON on SQLite — see ``models/json_type.py``.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from data_platform.config import Settings, load_settings  # noqa: E402
from data_platform.db import get_sessionmaker  # noqa: E402
from data_platform.models import Base  # noqa: E402


@pytest.fixture()
def settings(tmp_path) -> Settings:
    """Isolate each test: private sqlite file + private archive root.

    We both set env vars and rebind ``data_platform.config.SETTINGS`` so
    modules that already imported the singleton pick up the test DB.
    """
    db_path = tmp_path / "test_platform.db"
    archive_root = tmp_path / "archive"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ["RAW_ARCHIVE_ROOT"] = str(archive_root)
    os.environ["RAW_ARCHIVE_BACKEND"] = "local"
    os.environ["PLATFORM_ENABLED"] = "1"
    os.environ["API_FOOTBALL_KEY"] = os.environ.get("API_FOOTBALL_KEY", "test-key")

    fresh = load_settings()
    from data_platform import config as config_mod
    config_mod.SETTINGS = fresh
    return fresh


@pytest.fixture()
def engine(settings):
    from data_platform.db import get_engine
    engine = get_engine(settings)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture()
def session_factory(settings, engine):
    """Yield a context-managed session factory bound to the test engine."""
    factory = get_sessionmaker(settings)

    @contextmanager
    def _scope():
        session = factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return _scope
