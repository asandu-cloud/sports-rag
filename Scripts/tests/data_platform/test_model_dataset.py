from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from Scripts.data_platform.features.model_dataset import load_canonical_inputs, supported_competitions
from Scripts.ops import prediction_features as exporter


def _seed(session_factory, count=2):
    from data_platform.models import Competition, Season, Team, Fixture, FixtureTeamStats
    with session_factory() as session:
        competition = Competition(code="EPL", name="EPL", api_football_id=39)
        session.add(competition)
        session.flush()
        season = Season(competition_id=competition.id, year=2026, label="2026/27")
        home, away = Team(api_football_id=1, name="Same display name"), Team(api_football_id=2, name="Same display name")
        session.add_all([season, home, away])
        session.flush()
        for fid, day in [(100, 1), (101, 8)] + [(100 + i, 6 + i * 2) for i in range(2, count)]:
            fixture = Fixture(api_football_id=fid, competition_id=competition.id, season_id=season.id,
                              home_team_id=home.id, away_team_id=away.id, kickoff_utc=datetime(2026, 8, day, 12, tzinfo=timezone.utc),
                              status="FT", home_goals=0, away_goals=0, referee="Official",
                              last_fetched_at=datetime(2026, 8, day, 17, tzinfo=timezone.utc))
            session.add(fixture)
            session.flush()
            for team, opponent, is_home in [(home, away, True), (away, home, False)]:
                session.add(FixtureTeamStats(fixture_id=fixture.id, team_id=team.id, opponent_team_id=opponent.id,
                                            is_home=is_home, corners=0, shots_on=0, yellow_cards=0, red_cards=0,
                                            updated_at=datetime(2026, 8, day, 17, tzinfo=timezone.utc)))


def _path(settings):
    return Path(settings.database_url.removeprefix("sqlite:///"))


def test_registry_covers_all_public_competitions():
    assert set(supported_competitions()) == {"EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1", "Championship",
                                            "SuperLig", "Eredivisie", "PrimeiraLiga", "BelgianProLeague", "UCL", "UEL", "UECL"}


def test_canonical_reads_preserve_ids_zeros_and_database(settings, engine, session_factory):
    _seed(session_factory)
    path = _path(settings)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    fixtures, history, coverage = load_canonical_inputs(path)
    assert len(fixtures) == len(history) == 2
    assert history[0]["home_team_id"] != history[0]["away_team_id"]
    assert history[0]["home"]["goals"] == history[0]["home"]["cards"] == 0
    assert history[0]["home"]["xg"] is None
    assert history[0]["observed_at"] == "2026-08-01T17:00:00+00:00"
    assert coverage["quarantined"] == []
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_undated_and_badly_paired_rows_are_quarantined(settings, engine, session_factory):
    _seed(session_factory)
    with sqlite3.connect(_path(settings)) as db:
        db.execute("UPDATE fixtures SET kickoff_utc=NULL WHERE api_football_id=100")
        db.execute("UPDATE fixture_team_stats SET opponent_team_id=team_id WHERE fixture_id=(SELECT id FROM fixtures WHERE api_football_id=101)")
    fixtures, history, coverage = load_canonical_inputs(_path(settings))
    assert fixtures == history == []
    assert len(coverage["quarantined"]) == 2


def test_extra_time_is_not_treated_as_ninety_minutes(settings, engine, session_factory):
    _seed(session_factory)
    with sqlite3.connect(_path(settings)) as db:
        db.execute("UPDATE fixtures SET status='AET' WHERE api_football_id=100")
    _, history, _ = load_canonical_inputs(_path(settings))
    assert [row["fixture_id"] for row in history] == [101]


def test_export_isolated_replay_and_zero_target_eligibility(settings, engine, session_factory, tmp_path):
    _seed(session_factory)
    root = tmp_path / "project"
    active = root / "Index/ml_models/existing-model"
    active.parent.mkdir(parents=True)
    active.write_bytes(b"active model unchanged")
    target = exporter.export_dataset(root=root, database=_path(settings), name="phase1-test", seasons=[2026], availability="assumed_final")
    assert exporter.replay_dataset(target)["verified_rows"] == 2
    rows = [json.loads(line) for line in (target / "features.jsonl").read_text().splitlines()]
    assert rows[-1]["labels"] == {m: 0 for m in ("goals", "corners", "cards", "sot")}
    assert all(rows[-1]["feature_eligible"].values())
    assert active.read_bytes() == b"active model unchanged"
    coverage = json.loads((target / "coverage.json").read_text())
    assert coverage["cohorts"]["UECL:2026"]["exported"] == 0
    with pytest.raises(FileExistsError):
        exporter.export_dataset(root=root, database=_path(settings), name="phase1-test", seasons=[2026])
    with (target / "features.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        exporter.replay_dataset(target)


@pytest.mark.parametrize("name", ["../ml_models", "/tmp/model", "existing/model", ".", ""])
def test_experiment_paths_cannot_escape_root(tmp_path, name):
    with pytest.raises(ValueError):
        exporter.experiment_directory(tmp_path, name)


def test_symlink_experiment_root_is_rejected(tmp_path):
    (tmp_path / "Index").mkdir()
    active = tmp_path / "Index/ml_models"
    active.mkdir()
    (tmp_path / "Index/prediction_experiments").symlink_to(active, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        exporter.experiment_directory(tmp_path, "unsafe")
    assert not list(active.iterdir())


def test_prospective_snapshot_uses_only_observed_prior_inputs(settings, engine, session_factory, tmp_path):
    _seed(session_factory)
    as_of = datetime(2026, 8, 8, 10, tzinfo=timezone.utc)
    captured = exporter.export_fixture(root=tmp_path / "project", database=_path(settings), name="prospective", fixture_id=101, as_of=as_of)
    assert exporter.replay_dataset(captured)["verified_rows"] == 1
    snapshot = json.loads((captured / "input-snapshot.json").read_text())
    assert [row["fixture_id"] for row in snapshot["history"]] == [100]
    assert snapshot["fixture"]["referee"] is None  # Latest assignment observed after cutoff.
    assert "home" not in snapshot["fixture"]  # No target-match statistics.


def test_replay_uses_frozen_builder_not_current_functions(settings, engine, session_factory, tmp_path, monkeypatch):
    _seed(session_factory)
    captured = exporter.export_dataset(root=tmp_path / "project", database=_path(settings), name="frozen", seasons=[2026], availability="assumed_final")
    def wrong(*args, **kwargs):
        raise AssertionError("Must replay with the frozen implementation")
    monkeypatch.setattr(exporter, "capture_snapshot", wrong)
    monkeypatch.setattr(exporter, "inference_features", wrong)
    before = set(captured.rglob("*"))
    assert exporter.replay_dataset(captured)["verified_rows"] == 2
    assert set(captured.rglob("*")) == before
