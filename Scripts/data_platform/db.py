"""SQLAlchemy engine + session factory.

Provides a single ``engine`` / ``SessionLocal`` pair per database URL and
a context-managed ``session_scope`` for transactional use.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from functools import wraps
import time
from typing import Iterator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import OperationalError

from .config import SETTINGS, Settings


@lru_cache(maxsize=8)
def _engine_for(url: str, echo: bool = False) -> Engine:
    connect_args = {}
    if url.startswith("sqlite"):
        # Allow the engine to be used from background threads (bot + scheduler)
        connect_args = {"check_same_thread": False, "timeout": 2.0}
    return create_engine(url, future=True, echo=echo, connect_args=connect_args)


def get_engine(settings: Optional[Settings] = None, echo: bool = False) -> Engine:
    s = settings or SETTINGS
    return _engine_for(s.database_url, echo=echo)


@lru_cache(maxsize=8)
def _sessionmaker_for(url: str) -> sessionmaker:
    engine = _engine_for(url)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False)


def get_sessionmaker(settings: Optional[Settings] = None) -> sessionmaker:
    """Resolve the sessionmaker for ``settings`` (or the current module SETTINGS).

    Re-reads ``data_platform.config.SETTINGS`` on every call so tests that
    rebind ``SETTINGS`` after swapping env vars pick up the new URL.
    """
    if settings is not None:
        return _sessionmaker_for(settings.database_url)
    from . import config as _config
    return _sessionmaker_for(_config.SETTINGS.database_url)


# Kept for back-compat with early callers. Use get_sessionmaker() in new code.
SessionLocal = get_sessionmaker(SETTINGS)


@contextmanager
def session_scope(settings: Optional[Settings] = None) -> Iterator[Session]:
    """Context-managed session with commit on success, rollback on error."""
    factory = get_sessionmaker(settings)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def retry_database_busy(function):
    """Retry a complete short DB operation, never an API/model call.

    Each attempt must open its own session so a failed transaction is rolled
    back before retry. Non-lock database failures are never hidden.
    """
    @wraps(function)
    def wrapped(*args, **kwargs):
        for attempt in range(3):
            try:
                return function(*args, **kwargs)
            except OperationalError as exc:
                message = str(exc.orig).lower()
                if attempt == 2 or not any(value in message for value in (
                    "database is locked", "database table is locked", "database is busy",
                )):
                    raise
                time.sleep(0.1 * (attempt + 1))
    return wrapped
