"""Cross-backend JSON column type.

Uses JSONB on PostgreSQL (indexable, binary-packed). Falls back to the
generic JSON variant for SQLite and other backends so local development
and tests stay portable.
"""

from __future__ import annotations

from sqlalchemy import JSON
from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB

# with_variant lets one column type decl work against every supported
# dialect; tests on sqlite get plain JSON, prod on postgres gets JSONB.
JSONB = JSON().with_variant(PG_JSONB(), "postgresql")
