"""V2 service layer — orchestrates repositories + existing engine modules.

Services are the intended way for bots, the API, and future consumers to
reach into platform data. They accept plain values, return plain dicts,
and handle DI (session factory, optional cache backend) internally.
"""

from .fixtures import FixtureService  # noqa: F401
from .kb import KBService  # noqa: F401
from .live import LiveService  # noqa: F401
from .odds import OddsService  # noqa: F401
from .parlays import ParlayService  # noqa: F401
from .predictions import PredictionService  # noqa: F401
from .publications import PublicationService  # noqa: F401
