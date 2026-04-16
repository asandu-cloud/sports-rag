"""Competition registry — V3.

The single source of truth for which competitions exist, which provider
endpoints each one supports, their season rules, and any normalization
hints downstream sync/feature code needs. Replaces the hardcoded
``COMPETITIONS`` dict in ``sync/apifootball.py``.

Adding a new league is now: edit ``competitions.yaml``, add a mapping
entry if the provider uses non-standard team names, and re-run the
tests. No new bespoke ``Scripts/{League}/`` folder needed.
"""

from .loader import load_registry, DEFAULT_REGISTRY_PATH  # noqa: F401
from .models import (  # noqa: F401
    CompetitionEntry,
    ProviderMapping,
    SeasonRules,
    CompetitionRegistry,
)
