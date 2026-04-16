"""Platform configuration.

Centralises environment-driven settings for the V1 data platform. Reads
from environment variables (which typically come from a project-level
``.env`` file). Safe to import at any point — side-effect-free.

Environment variables
---------------------
DATABASE_URL
    SQLAlchemy URL for the canonical store. Defaults to a file-backed
    SQLite database at ``Index/platform.db`` for local development. In
    production this should be a PostgreSQL URL, e.g.
    ``postgresql+psycopg://user:pass@host:5432/spix``.

PLATFORM_ENABLED
    Feature flag. When ``1`` / ``true`` / ``yes``, the shared app code
    (``web_app/users.py``, etc.) routes reads/writes through the new
    PostgreSQL-backed repositories. Otherwise the existing SQLite
    implementation is used.

RAW_ARCHIVE_ROOT
    Directory (for ``local`` backend) under which raw provider payloads
    are stored. Defaults to ``Index/raw_archive``.

RAW_ARCHIVE_BACKEND
    ``local`` (default) or ``s3``. The ``s3`` backend is a thin stub
    intended to be wired up when object storage is introduced.

S3_BUCKET, S3_REGION, S3_PREFIX, S3_ENDPOINT_URL
    Optional settings for the S3 backend. Not required for V1.

API_FOOTBALL_KEY
    Kept in sync with the existing ``API-FOOTBALL-KEY`` env var (the
    hyphenated variant used throughout the old scripts).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a soft dep for config
    pass


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _default_database_url() -> str:
    """SQLite fallback keeps local development bootstrapable without Postgres."""
    path = _PROJECT_ROOT / "Index" / "platform.db"
    return f"sqlite:///{path}"


def _resolve_relative_sqlite(url: str) -> str:
    """Anchor relative SQLite paths to the project root.

    SQLite resolves relative paths against the *current working directory*,
    which differs across processes (bot runs from Scripts/discord_bot/, web
    app from Scripts/web_app/, etc.). That breaks every non-root caller.
    Detect ``sqlite:///<rel>`` (one extra slash short of an absolute path)
    and rewrite to ``sqlite:////<project_root>/<rel>``. ``sqlite:////...``
    (already absolute) and non-sqlite URLs pass through untouched.
    """
    if not url.startswith("sqlite:///"):
        return url
    body = url[len("sqlite:///"):]
    if not body or body.startswith("/"):
        return url   # in-memory or already absolute
    resolved = (_PROJECT_ROOT / body).resolve()
    return f"sqlite:///{resolved}"


@dataclass(frozen=True)
class Settings:
    project_root: Path
    database_url: str
    platform_enabled: bool
    raw_archive_root: Path
    raw_archive_backend: str
    s3_bucket: Optional[str]
    s3_region: Optional[str]
    s3_prefix: Optional[str]
    s3_endpoint_url: Optional[str]
    api_football_key: Optional[str]

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgres")


def load_settings() -> Settings:
    """Read settings from the environment. Cheap — call ad hoc."""
    raw_root = os.environ.get("RAW_ARCHIVE_ROOT")
    root = Path(raw_root) if raw_root else _PROJECT_ROOT / "Index" / "raw_archive"
    backend = os.environ.get("RAW_ARCHIVE_BACKEND", "local").strip().lower()
    if backend not in {"local", "s3"}:
        backend = "local"

    api_key = (
        os.environ.get("API_FOOTBALL_KEY")
        or os.environ.get("API-FOOTBALL-KEY")
    )

    return Settings(
        project_root=_PROJECT_ROOT,
        database_url=_resolve_relative_sqlite(os.environ.get("DATABASE_URL") or _default_database_url()),
        platform_enabled=_bool_env("PLATFORM_ENABLED", default=False),
        raw_archive_root=root,
        raw_archive_backend=backend,
        s3_bucket=os.environ.get("S3_BUCKET"),
        s3_region=os.environ.get("S3_REGION"),
        s3_prefix=os.environ.get("S3_PREFIX"),
        s3_endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
        api_football_key=api_key,
    )


# Default settings singleton — callers may override by constructing their own
# Settings if they need to point at a test DB, alternate archive, etc.
SETTINGS = load_settings()
