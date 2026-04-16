"""Compatibility shims bridging legacy SQLite callers to the new platform.

The goal is *additive* migration: leave legacy modules and every existing
import site untouched, but route the data layer through the canonical
PostgreSQL repositories when ``PLATFORM_ENABLED=1``.
"""

from .live_shim import (  # noqa: F401
    PlatformLiveStore,
    get_live_service,
    is_platform_enabled as live_is_platform_enabled,
    migrate_sqlite_live,
)
from .parlay_shim import (  # noqa: F401
    PlatformParlayStore,
    get_parlay_service,
    is_platform_enabled as parlay_is_platform_enabled,
    migrate_sqlite_parlay_sessions,
)
from .predictions_shim import (  # noqa: F401
    get_prediction_service,
    is_platform_enabled as predictions_is_platform_enabled,
    migrate_sqlite_predictions,
    platform_capture_closing_odds,
    platform_get_calibration_data,
    platform_get_daily_breakdown,
    platform_get_recent_predictions,
    platform_get_track_record,
    platform_get_unresolved_for_grading,
    platform_log_prediction,
    platform_mark_outcome,
)
from .users_shim import (  # noqa: F401
    get_user_repository,
    is_platform_enabled,
    migrate_sqlite_users,
)
