"""Feature versioning + lineage helpers (V3).

Every feature-snapshot upsert ought to carry a pointer back to the
``FeatureBuild`` that generated it. Version strings live in
``versions.CURRENT_FEATURE_VERSION`` and can be bumped when the pipeline
changes in a way that invalidates past snapshots.
"""

from .lineage import (  # noqa: F401
    FeatureBuildContext,
    feature_build,
    finish_feature_build,
    start_feature_build,
)
from .versions import CURRENT_FEATURE_VERSION, feature_version_for  # noqa: F401
