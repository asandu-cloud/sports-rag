"""SQLAlchemy ORM models for the canonical data platform.

Import order matters for Alembic autogenerate — ``Base.metadata`` is
populated as a side-effect of importing these modules.
"""

from .base import Base, TimestampMixin  # noqa: F401
from .core import Competition, Season, Team, Player  # noqa: F401
from .fixtures import Fixture, FixtureTeamStats, FixturePlayerStats  # noqa: F401
from .sync import SyncRun, SyncWatermark, RawPayloadArchive  # noqa: F401
from .features import TeamFeatureSnapshot, PlayerFeatureSnapshot  # noqa: F401
from .users import AppUser, AppSubscriptionEvent, AppReferral  # noqa: F401
from .predictions import Prediction, TrialUser, TrialParlay  # noqa: F401
from .publications import PublishedRecommendation, RecommendationDelivery  # noqa: F401
from .parlays import ParlaySession  # noqa: F401
from .live import (  # noqa: F401
    LiveFixture,
    LiveStateSnapshot,
    LiveModelSnapshot,
    LiveOddsSnapshot,
    LiveAlert,
)
from .kb import KBDocument, KBRefreshQueueItem  # noqa: F401
from .odds import OddsSnapshot  # noqa: F401
from .feature_builds import FeatureBuild  # noqa: F401

__all__ = [
    "Base",
    "TimestampMixin",
    "Competition",
    "Season",
    "Team",
    "Player",
    "Fixture",
    "FixtureTeamStats",
    "FixturePlayerStats",
    "SyncRun",
    "SyncWatermark",
    "RawPayloadArchive",
    "TeamFeatureSnapshot",
    "PlayerFeatureSnapshot",
    "AppUser",
    "AppSubscriptionEvent",
    "AppReferral",
    "Prediction",
    "TrialUser",
    "TrialParlay",
    "PublishedRecommendation",
    "RecommendationDelivery",
    "ParlaySession",
    "LiveFixture",
    "LiveStateSnapshot",
    "LiveModelSnapshot",
    "LiveOddsSnapshot",
    "LiveAlert",
    "KBDocument",
    "KBRefreshQueueItem",
    "OddsSnapshot",
    "FeatureBuild",
]
