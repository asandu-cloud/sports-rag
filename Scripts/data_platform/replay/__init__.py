"""Replay + repair tooling (V3).

Rebuild any slice of canonical data deterministically:
  * a single fixture
  * a date range
  * a whole competition/season
  * KB docs for a specific entity

A ``ReplayPlan`` describes *what* to rebuild; the ``ReplayExecutor`` does it
against the configured data sources. ``compare_values`` diffs DB state
against freshly-regenerated values — useful when investigating drift.
"""

from .planner import ReplayPlan, ReplayScope, build_plan  # noqa: F401
from .executor import ReplayExecutor, ReplayResult  # noqa: F401
from .compare import compare_values, ValueDiff  # noqa: F401
