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
from typing import Optional, Tuple

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


def _positive_int_env(name: str, default: int, *, minimum: int = 1) -> int:
    """Read a positive integer setting without making startup brittle."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


def _csv_env(name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    return values or default


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


DEFAULT_MATCH_READ_WORKER_LEAGUES = (
    "EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1", "UCL",
)


@dataclass(frozen=True)
class MatchReadWorkerSettings:
    """Settings for one scheduled Match Read cycle.

    The worker is deliberately one-shot: a platform scheduler invokes it at
    the configured cadence.  Keeping that scheduler outside the Python
    process makes local, launchd/cron, and cloud deployments use the same
    idempotent command.
    """

    mode: str
    leagues: Tuple[str, ...]
    # Match Read persistence currently selects matchdays from their UTC
    # kickoff strings. Keep the worker's grouping contract UTC until that
    # stored contract is intentionally migrated end-to-end.
    matchday_timezone: str
    outlook_hours: int
    final_window_minutes: int
    refresh_minutes: int
    lineup_window_minutes: int
    lineup_refresh_minutes: int
    max_age_minutes: int
    lease_seconds: int


def load_match_read_worker_settings() -> MatchReadWorkerSettings:
    """Load safe Match Read worker defaults from the environment.

    ``shadow`` is the default mode. ``website`` must be selected explicitly
    by the scheduled command or environment before any tracked public cards
    can be created.
    """
    raw_mode = os.environ.get("MATCH_READ_WORKER_MODE", "shadow").strip().lower()
    mode = raw_mode if raw_mode in {"shadow", "website"} else "shadow"
    return MatchReadWorkerSettings(
        mode=mode,
        leagues=_csv_env("MATCH_READ_WORKER_LEAGUES", DEFAULT_MATCH_READ_WORKER_LEAGUES),
        matchday_timezone=os.environ.get("MATCH_READ_MATCHDAY_TIMEZONE", "UTC").strip() or "UTC",
        outlook_hours=_positive_int_env("MATCH_READ_OUTLOOK_HOURS", 48),
        final_window_minutes=_positive_int_env("MATCH_READ_FINAL_WINDOW_MINUTES", 120),
        refresh_minutes=_positive_int_env("MATCH_READ_REFRESH_MINUTES", 10),
        lineup_window_minutes=_positive_int_env("MATCH_READ_LINEUP_WINDOW_MINUTES", 100),
        lineup_refresh_minutes=_positive_int_env("MATCH_READ_LINEUP_REFRESH_MINUTES", 10),
        max_age_minutes=_positive_int_env("MATCH_READ_MAX_AGE_MINUTES", 120),
        lease_seconds=_positive_int_env("MATCH_READ_WORKER_LEASE_SECONDS", 540),
    )


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
