"""Alembic environment.

Wires Alembic to the same ``DATABASE_URL`` the runtime uses so migrations
apply against whatever backend the caller configures. SQLAlchemy 2.x
async is not required in V1.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make the data_platform package importable when Alembic runs from alembic/
_HERE = Path(__file__).resolve()
_PKG = _HERE.parents[1]                # Scripts/data_platform
_SCRIPTS = _PKG.parent                 # Scripts
_PROJECT = _SCRIPTS.parent             # repo root
for p in (str(_PROJECT), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Import models so their metadata is attached to Base before autogen
from data_platform import models  # noqa: F401  (side-effect: register models)
from data_platform.config import SETTINGS
from data_platform.models import Base

config = context.config

# Logging config (alembic.ini) if present
if config.config_file_name is not None:
    try:
        fileConfig(config.config_file_name)
    except Exception:
        pass

target_metadata = Base.metadata

# Resolve URL at runtime — overrides the empty sqlalchemy.url in alembic.ini
config.set_main_option("sqlalchemy.url", SETTINGS.database_url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — emit SQL only."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        is_sqlite = connection.dialect.name == "sqlite"
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Alembic needs batch mode on SQLite to handle many ALTER TABLE ops
            render_as_batch=is_sqlite,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
