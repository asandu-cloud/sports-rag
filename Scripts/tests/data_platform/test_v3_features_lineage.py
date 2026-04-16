"""Feature build lineage + version tagging on snapshot upserts."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest


@pytest.fixture()
def lineage_env(monkeypatch, settings, engine, session_factory):
    """Route every session factory the lineage helpers touch to our test DB."""
    import data_platform.db as db_module
    monkeypatch.setattr(db_module, "session_scope", session_factory)
    from data_platform.features import lineage as lineage_mod
    monkeypatch.setattr(lineage_mod, "session_scope", session_factory)
    return session_factory


def test_start_feature_build_and_stamp_snapshot(lineage_env, session_factory):
    from data_platform.features import feature_build
    from data_platform.features.versions import CURRENT_FEATURE_VERSION
    from data_platform.models import FeatureBuild, Season, Team, Competition
    from data_platform.sync.apifootball import COMPETITIONS
    from data_platform.sync.upserts import (
        upsert_competition,
        upsert_season,
        upsert_team,
        upsert_team_feature_snapshot,
    )

    with session_factory() as session:
        comp = upsert_competition(session, COMPETITIONS["EPL"])
        season = upsert_season(session, competition=comp, year=2025)
        team = upsert_team(session, api_id=42, name="Arsenal")
        session.flush()

    with feature_build(scope="EPL:2025", as_of_date=date(2025, 12, 1),
                       pipeline="default", notes="unit test") as ctx:
        assert ctx.feature_version == CURRENT_FEATURE_VERSION
        with session_factory() as session:
            season = session.query(Season).first()
            team = session.query(Team).first()
            snap = upsert_team_feature_snapshot(
                session,
                team=team, season=season, as_of_date=date(2025, 12, 1),
                fields={"goals_for_pm": 2.2, "corners_pm": 5.5, "control_index": 0.6,
                        "matches_played": 10},
                feature_build_id=ctx.build_id,
                feature_version=ctx.feature_version,
            )
            assert snap.feature_build_id == ctx.build_id
            assert snap.feature_version == CURRENT_FEATURE_VERSION
            assert snap.built_at is not None

    with session_factory() as session:
        build_row = session.query(FeatureBuild).first()
        assert build_row is not None
        assert build_row.scope == "EPL:2025"
        assert build_row.feature_version == CURRENT_FEATURE_VERSION
        assert build_row.status == "completed"
        assert build_row.finished_at is not None


def test_reuses_existing_build_for_same_scope(lineage_env):
    from data_platform.features import start_feature_build
    from data_platform.models import FeatureBuild

    ctx1 = start_feature_build(scope="UCL:2025", as_of_date=date(2025, 11, 1))
    ctx2 = start_feature_build(scope="UCL:2025", as_of_date=date(2025, 11, 1))
    assert ctx1.build_id == ctx2.build_id
    with lineage_env() as session:
        rows = session.query(FeatureBuild).filter_by(scope="UCL:2025").all()
        assert len(rows) == 1
