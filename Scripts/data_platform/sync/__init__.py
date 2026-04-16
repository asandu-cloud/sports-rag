"""Incremental + bootstrap sync from API-Football into the canonical DB."""

from .apifootball import ApiFootballClient, COMPETITIONS, CompetitionSpec  # noqa: F401
from .pipeline import bootstrap_season, incremental_refresh, sync_fixture_details  # noqa: F401
from .features_from_snapshot import build_feature_snapshots_from_files  # noqa: F401
