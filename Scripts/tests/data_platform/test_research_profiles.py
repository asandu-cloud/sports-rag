"""Research contract checks on private databases, without prediction imports."""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from data_platform.models import (Competition, Fixture, FixturePlayerStats, FixtureTeamStats,
                                  Player, ResearchPeerStat, ResearchProfileBuild, Season, Team)
from data_platform.repositories.research import ResearchRepository
from data_platform.research_stats import COUNT_NULL_POLICY, normalize_research_row
from data_platform.services.research_profiles import (
    ResearchProfileService, eligibility, frequency, headline, percentiles,
    player_summary, referee_aliases, referee_summary, starter, team_summary,
    player_role, squad_leaders, referee_country,
)


def test_percentile_midpoints_ties_and_rounding():
    assert percentiles([10, 20, 20, 30]) == [12, 50, 50, 88]
    assert percentiles([7, 7, 7]) == [50, 50, 50]
    assert percentiles([4]) == [50]
    assert percentiles([]) == []


def test_starter_flag_wins_over_minutes_and_null_falls_back():
    assert starter({"minutes": 1, "substitute": False}) == (True, "substitute_flag")
    assert starter({"minutes": 90, "substitute": True}) == (False, "substitute_flag")
    assert starter({"minutes": 59, "substitute": None}) == (False, "minutes_fallback")
    assert starter({"minutes": 60, "substitute": None}) == (True, "minutes_fallback")


def test_player_rates_are_sums_and_counts_ignore_dnp():
    rows = [dict(minutes=60, position="F", substitute=False, goals=2, passes_accurate=1,
                 passes_total=2, duels_won=1, duels_total=2, rating=6),
            dict(minutes=30, position="M", substitute=True, goals=0, passes_accurate=9,
                 passes_total=10, duels_won=9, duels_total=10, rating=8),
            dict(minutes=0, position="G", substitute=False, goals=99)]
    summary = player_summary(rows)
    assert summary["totals"]["appearances"] == 2
    assert summary["totals"]["starts"] == 1
    assert summary["metrics"]["goals_per_90"]["value"] == 2
    assert summary["metrics"]["pass_accuracy_pct"]["value"] == pytest.approx(1000 / 12)
    assert summary["metrics"]["duel_win_pct"]["value"] == pytest.approx(1000 / 12)
    assert summary["metrics"]["average_rating"]["value"] == 7
    assert summary["peer_group"] == "F"  # tied appearance count, alphabetic tie break
    assert summary["frequencies"]["scored"] == {"count": 1, "denominator": 2, "unknown": 0}


def test_populated_row_count_nulls_are_zero_in_totals_rates_and_frequencies():
    summary = player_summary([
        dict(minutes=90, goals=1, shots_on=1, passes_accurate=8, passes_total=10),
        dict(minutes=90, goals=None, shots_on=None, passes_accurate=None, passes_total=100),
    ])
    assert summary["metrics"]["goals_per_90"]["value"] == .5
    assert summary["metrics"]["goals_per_90"]["coverage"]["missing_matches"] == 0
    assert summary["frequencies"]["scored"] == {"count": 1, "denominator": 2, "unknown": 0}
    assert summary["metrics"]["pass_accuracy_pct"]["value"] == pytest.approx(800 / 110)
    assert summary["metrics"]["cards_per_90"]["value"] == 0
    assert frequency([None, 0, 4, 5], lambda v: v >= 4) == {"count": 2, "denominator": 3, "unknown": 1}


def test_played_player_is_observed_even_when_all_count_fields_are_null():
    empty = dict(player_id=1, team_id=2, minutes=90, position="F", substitute=False,
                 captain=True, home_goals=3, away_goals=0, goals=None, shots_on=None)
    normal = normalize_research_row("player", empty)
    assert normal["_stats_available"] == {"subject": True}
    assert normal["goals"] == 0
    assert empty["goals"] is None  # no source mutation
    result = player_summary([empty, dict(minutes=90, goals=1, shots_on=None)])
    assert result["frequencies"]["scored"] == {"count": 1, "denominator": 2, "unknown": 0}
    assert result["frequencies"]["shots_on_target_1_plus"] == {"count": 0, "denominator": 2, "unknown": 0}
    assert result["metrics"]["average_rating"]["value"] is None


def test_zero_is_populated_but_non_count_measurements_remain_nullable():
    row = dict(minutes=90, tackles=0, rating=None, goals=None)
    normalized = normalize_research_row("player", row)
    assert normalized["goals"] == 0
    assert normalized["rating"] is None
    assert row["goals"] is None
    assert normalize_research_row("player", normalized) == normalized
    result = player_summary([row])
    assert result["metrics"]["pass_accuracy_pct"]["value"] is None  # 0/0, not zero percent
    assert result["frequencies"]["scored"] == {"count": 0, "denominator": 1, "unknown": 0}
    team = normalize_research_row("team", dict(status="FT", goals=0, corners=0, opponent_shots_total=1,
                                               expected_goals=None, possession=None))
    assert team["corners"] == 0
    assert team["expected_goals"] is None
    assert team["possession"] is None
    assert team["_stats_observed"] is True


@pytest.mark.parametrize("row", [dict(minutes=0, passes_total=1, goals=None),
                                 dict(minutes=None, goals=None),
                                 dict(minutes=90, status="NS", passes_total=1, goals=None)])
def test_unplayed_or_unfinished_player_counts_are_not_filled(row):
    assert normalize_research_row("player", row)["goals"] is None


def test_empty_team_side_is_independent_of_opponent_and_fixture_score():
    row = dict(status="FT", home_goals=3, away_goals=0, is_home=True,
               goals=3, corners=None, opponent_goals=0, opponent_corners=None, opponent_shots_on=0)
    normalized = normalize_research_row("team", row)
    assert normalized["_stats_available"] == {"subject": False, "opponent": True}
    assert normalized["goals"] is None and normalized["corners"] is None
    assert normalized["opponent_corners"] is None  # both sides are needed for match statistics
    assert normalized["result_goals_for"] == 3
    assert normalized["result_goals_against"] == 0
    not_finished = normalize_research_row("team", dict(status="NS", goals=0, corners=None))
    assert not_finished["corners"] is None


def test_threshold_boundaries():
    s = {"minutes": 899, "matches": 15, "peer_group": "F"}
    assert eligibility("player", s) == (False, "below_minutes_threshold")
    assert eligibility("player", {**s, "minutes": 900}) == (True, None)
    assert eligibility("referee", {**s, "matches": 14}) == (False, "below_matches_threshold")
    assert eligibility("referee", s) == (True, None)


def test_referee_counts_normalize_each_populated_side_before_summing():
    rows = [dict(home_stats_yellow_cards=2, away_stats_yellow_cards=2, home_stats_red_cards=0, away_stats_red_cards=0,
                 home_stats_fouls_committed=15, away_stats_fouls_committed=10),
            dict(home_stats_yellow_cards=2, away_stats_yellow_cards=1, home_stats_red_cards=None, away_stats_red_cards=None,
                 home_stats_fouls_committed=None, away_stats_fouls_committed=5)]
    result = referee_summary(rows)
    assert result["metrics"]["cards"]["value"] == 3.5
    assert result["frequencies"]["cards_4_plus"] == {"count": 1, "denominator": 2, "unknown": 0}
    assert result["metrics"]["cards_per_foul"]["value"] == pytest.approx(7/30)
    empty_side = dict(home_stats_yellow_cards=4, home_stats_red_cards=None, home_stats_fouls_committed=10,
                      away_stats_yellow_cards=None, away_stats_red_cards=None, away_stats_fouls_committed=None)
    with_empty = referee_summary(rows + [empty_side])
    assert with_empty["frequencies"]["cards_4_plus"] == {"count": 1, "denominator": 2, "unknown": 1}


def test_team_lower_is_more_rank_and_ties():
    from data_platform.services.research_profiles import rank_rows
    def summary(value):
        return {"matches": 10, "stats_observed_matches": 10, "minutes": None, "peer_group": "all", "metrics": {
            "goals_against": {"value": value, "higher_is_more": False, "coverage": {"observed_matches": 10}}}}
    rows = rank_rows("team", {"1": summary(.5), "2": summary(.5), "3": summary(2)}, {}, {})
    assert [r["rank"] for r in rows] == [1, 1, 3]
    assert [r["percentile"] for r in rows] == [33, 33, 83]
    assert all(r["league_average"] == 1 for r in rows)


def test_referee_name_aliases_merge_and_key_survives_full_name_arrival():
    before = referee_aliases(["A. Taylor"])
    after = referee_aliases(["Anthony Taylor, England", "A. Taylor", "M. Taylor", " "])
    keys = {r["raw_name"]: r["referee_key"] for r in after}
    assert keys["A. Taylor"] == keys["Anthony Taylor, England"] == before[0]["referee_key"]
    assert keys["M. Taylor"] != keys["A. Taylor"]


def test_headline_extremeness_priority_ties_and_no_advice():
    metrics = {name: {"value": 1.25, "percentile": pct, "rank": 1, "peer_count": 20,
                      "unit": "per_90", "coverage": {"missing_matches": 0}}
               for name, pct in (("goals_per_90", 10), ("assists_per_90", 90), ("shots_per_90", 70))}
    result = headline("player", {"name": "Alex"}, {"name": "League"}, metrics, "F")
    assert result["metric"] == "goals_per_90"
    assert result["subject_type"] == "player"
    assert not any(word in result["text"].lower().split() for word in ("bet", "back", "value", "tip"))
    assert headline("player", {}, {}, {}, "F") is None


@pytest.fixture()
def research(session_factory):
    with session_factory() as s:
        comp = Competition(code="EPL", name="Premier League", api_football_id=39)
        cup = Competition(code="UCL", name="Champions League", api_football_id=2)
        s.add_all([comp, cup]); s.flush()
        seasons = [Season(competition_id=comp.id, year=y, label=f"{y}/{str(y+1)[-2:]}") for y in (2025, 2026, 2027)]
        seasons += [Season(competition_id=cup.id, year=2025, label="2025/26")]
        teams = [Team(name=n, api_football_id=100+i, short_code=n[:3].upper(), country="England",
                      logo_url=f"https://example.invalid/{n}.png") for i, n in enumerate(("Alpha", "Beta", "Gamma"))]
        players = [Player(name=n, api_football_id=200+i) for i, n in enumerate(("Alex", "Alex", "Low minutes"))]
        s.add_all(seasons + teams + players); s.flush()
        start = datetime(2025, 8, 1, tzinfo=timezone.utc)
        for i in range(20):
            year = 2025 if i < 17 else 2026 if i < 19 else 2027
            f = Fixture(api_football_id=1000+i, competition_id=comp.id,
                        season_id=seasons[year-2025].id, home_team_id=teams[i % 2].id,
                        away_team_id=teams[1-i % 2].id, kickoff_utc=start+timedelta(days=i+365*(year-2025)),
                        status="NS" if i in (16, 19) else "AET" if i == 0 else "PEN" if i == 1 else "FT",
                        home_goals=2, away_goals=0,
                        referee="A. Taylor" if i % 2 else "Anthony Taylor, England")
            s.add(f); s.flush()
            for home, tid, opp in ((True, f.home_team_id, f.away_team_id), (False, f.away_team_id, f.home_team_id)):
                if i == 15 and not home:
                    continue  # this fixture must not count for referees
                s.add(FixtureTeamStats(fixture_id=f.id, team_id=tid, opponent_team_id=opp, is_home=home,
                      goals=2 if home else 0, shots_on=5 if home else 1, corners=7 if home else 3,
                      possession=.6 if home else .4, expected_goals=1.5 if home else .3,
                      yellow_cards=2 if home else 1, red_cards=1 if home else 0,
                      fouls_committed=15 if home else 10))
            if i == 15:
                continue
            for j, player in enumerate(players):
                s.add(FixturePlayerStats(fixture_id=f.id, player_id=player.id, team_id=teams[0].id,
                    minutes=60 if j == 0 else 90 if j == 1 else 1, position="F", substitute=None if i == 0 else j != 0,
                    goals=1 if j == 0 else 0, assists=0, shots_total=3, shots_on=1, yellow_cards=0, red_cards=0,
                    passes_total=10, passes_accurate=8, duels_total=5, duels_won=2, fouls_drawn=0, fouls_committed=1, rating=7))
        # Same official in another competition contributes to cross-competition scope.
        f = Fixture(api_football_id=2000, competition_id=cup.id, season_id=seasons[3].id,
                    home_team_id=teams[0].id, away_team_id=teams[2].id, status="FT",
                    kickoff_utc=start, home_goals=1, away_goals=1, referee="Anthony Taylor, England")
        s.add(f); s.flush()
        for tid, opp, home in ((teams[0].id, teams[2].id, True), (teams[2].id, teams[0].id, False)):
            s.add(FixtureTeamStats(fixture_id=f.id, team_id=tid, opponent_team_id=opp, is_home=home,
                  goals=1, yellow_cards=0, red_cards=0, fouls_committed=0))
        ids = {"player": players[0].id, "other_player": players[1].id, "low": players[2].id,
               "team": teams[0].id, "competition": comp.id}
    service = ResearchProfileService(ResearchRepository(session_factory))
    service.build()
    key = service.search("referees", "Taylor")["results"][0]["referee_key"]
    return service, {**ids, "referee": key}


def test_profiles_defaults_early_season_and_prior_ranks(research):
    service, ids = research
    previous = service.profile("player", ids["player"], competition="EPL", season=2025)
    assert previous["totals"]["minutes"] == 900
    assert previous["ranking"]["ranked"] is True
    assert previous["metrics"]["goals_per_90"]["peer_count"] == 2
    assert previous["metrics"]["goals_per_90"]["percentile"] == 75
    assert previous["starts_basis"]["method"] == "mixed"
    assert previous["starts_basis"]["minutes_fallback_matches"] == 1
    current = service.profile("player", ids["player"])
    assert current["season"]["year"] == 2026  # ignore empty/unplayed 2027
    assert current["ranking"]["reason"] == "below_minutes_threshold"
    assert current["previous_season"]["season"]["year"] == 2025
    assert current["previous_season"]["ranking"]["ranked"] is True
    assert current["headline"] is None
    assert len(previous["last_10"]) == 10
    assert [m["date"] for m in previous["last_10"]] == sorted(m["date"] for m in previous["last_10"])


def test_opponent_join_direction_league_average_and_team_frequencies(research):
    service, ids = research
    profile = service.profile("team", ids["team"], competition="EPL", season=2025)
    match = profile["last_10"][-2]  # a complete fixture
    own = match["venue"] == "home"
    assert match["values"]["xg_against"] == (.3 if own else 1.5)
    assert match["values"]["corners_against"] == (3 if own else 7)
    assert profile["metrics"]["goals_against"]["higher_is_more"] is False
    assert profile["metrics"]["goals_against"]["rank"] in (1, 2)
    assert profile["metrics"]["possession"]["value"] > 1  # display percentage
    assert profile["metrics"]["possession"]["league_average"] is not None
    assert profile["totals"]["corners_for"] == 7 * 8 + 3 * 7
    assert profile["frequencies"]["corners_10_plus"] == {"count": 15, "denominator": 15, "unknown": 1}
    assert profile["totals"]["matches"] == 16
    assert profile["totals"]["stats_observed_matches"] == 15
    assert profile["frequencies"]["cards_4_plus"]["count"] == 15
    assert sum(x["matches"] for x in profile["home_away"].values()) == profile["totals"]["matches"]


def test_referee_pair_requirement_cross_competition_and_previous_season(research):
    service, ids = research
    profile = service.profile("referee", ids["referee"], season=2025)
    assert profile["totals"]["matches"] == 16
    assert profile["ranking"]["ranked"] is True
    assert profile["totals"]["finished_matches"] == 17
    assert profile["totals"]["excluded_matches"] == 1
    assert profile["frequencies"]["cards_4_plus"] == {"count": 15, "denominator": 16, "unknown": 1}
    assert profile["metrics"]["cards_per_foul"]["value"] == pytest.approx(4 / 25)
    assert {b["competition"] for b in profile["competition_breakdown"]} == {"EPL", "UCL"}
    early = service.profile("referee", ids["referee"])
    assert early["totals"]["matches"] == 2
    assert early["ranking"]["reason"] == "below_matches_threshold"
    assert early["previous_season"]["ranking"]["ranked"] is True


def test_rebuild_idempotency_scoped_replacement_and_rollback(research, session_factory, monkeypatch):
    service, ids = research
    with session_factory() as s:
        count = s.scalar(select(func.count()).select_from(ResearchPeerStat))
        untouched = s.get(ResearchProfileBuild, "referee:2026").build_id
    first = service.build(season=2025, competition="EPL")
    second = service.build(season=2025, competition="EPL")
    assert first["peer_rows"] == second["peer_rows"]
    with session_factory() as s:
        assert s.scalar(select(func.count()).select_from(ResearchPeerStat)) == count
        assert s.get(ResearchProfileBuild, "referee:2026").build_id == untouched
        before = s.get(ResearchProfileBuild, "referee:2025").build_id
    original = service.repo.replace
    def fail_after_replace(*args):
        original(*args)
        raise RuntimeError("simulated interrupted build")
    monkeypatch.setattr(service.repo, "replace", fail_after_replace)
    with pytest.raises(RuntimeError, match="interrupted"):
        service.build(season=2025)
    with session_factory() as s:
        assert s.get(ResearchProfileBuild, "referee:2025").build_id == before
        assert s.scalar(select(func.count()).select_from(ResearchPeerStat)) == count


def test_stale_ranks_are_not_attached_to_new_totals(research, session_factory):
    service, ids = research
    with session_factory() as s:
        row = s.scalar(select(FixturePlayerStats).where(FixturePlayerStats.player_id == ids["player"]))
        row.goals = 100
    result = service.profile("player", ids["player"], competition="EPL", season=2025)
    assert result["ranking"]["reason"] == "ranks_stale_or_missing"
    assert result["headline"] is None


def test_old_null_policy_ranks_require_rebuild_even_when_subject_value_matches(research, session_factory):
    service, ids = research
    with session_factory() as s:
        build = s.get(ResearchProfileBuild, f"competition:{ids['competition']}:2025")
        build.build_id = "old-unversioned-build"
    result = service.profile("player", ids["player"], competition="EPL", season=2025)
    assert result["ranking"]["reason"] == "rank_policy_changed_rebuild_required"
    assert result["headline"] is None


def test_queries_charts_peer_builds_and_frequencies_share_count_policy(research, session_factory):
    service, ids = research
    from data_platform.research_stats import (PLAYER_COUNT_FIELDS, PLAYER_MEASUREMENT_FIELDS,
                                              TEAM_COUNT_FIELDS, TEAM_MEASUREMENT_FIELDS)
    with session_factory() as s:
        season = s.scalar(select(Season).where(Season.competition_id == ids["competition"], Season.year == 2025))
        fixture_ids = list(s.scalars(select(Fixture.id).where(Fixture.season_id == season.id, Fixture.status == "FT").order_by(Fixture.id)))
        populated_id, empty_id = fixture_ids[-4:-2]
        for fid, populated in ((populated_id, True), (empty_id, False)):
            player = s.scalar(select(FixturePlayerStats).where(FixturePlayerStats.fixture_id == fid,
                                                               FixturePlayerStats.player_id == ids["player"]))
            assert player is not None
            for field in PLAYER_COUNT_FIELDS + PLAYER_MEASUREMENT_FIELDS:
                setattr(player, field, None)
            if populated:
                player.interceptions = 1  # outside the displayed metric selection
            for team in s.scalars(select(FixtureTeamStats).where(FixtureTeamStats.fixture_id == fid)):
                for field in TEAM_COUNT_FIELDS + TEAM_MEASUREMENT_FIELDS:
                    setattr(team, field, None)
                if populated:
                    team.shots_total = 1  # one of the eight explicit observation fields
    service.build(season=2025)
    for kind, key in (("player", ids["player"]), ("team", ids["team"]), ("referee", ids["referee"])):
        result = service.profile(kind, key, season=2025)
        assert result["ranking"]["ranked"] is True
        assert result["data_basis"]["count_null_policy_version"] == COUNT_NULL_POLICY
        charts = {r["fixture_id"]: r for r in result["last_10"]}
        field = "goals" if kind == "player" else "cards"
        assert charts[populated_id]["values"][field] == 0
        assert charts[empty_id]["values"][field] == (0 if kind == "player" else None)
        frequency_name = "scored" if kind == "player" else "cards_4_plus"
        assert result["frequencies"][frequency_name]["unknown"] == (0 if kind == "player" else 2)
    with session_factory() as s:
        player = s.scalar(select(FixturePlayerStats).where(FixturePlayerStats.fixture_id == populated_id,
                                                          FixturePlayerStats.player_id == ids["player"]))
        assert player.goals is None  # display policy never rewrites canonical values


def test_router_isolated_errors_search_ids_and_read_only(research, session_factory, monkeypatch):
    from web_app.routers import research as router_module
    service, ids = research
    monkeypatch.setattr(router_module, "_get_research_service", lambda: service)
    app = FastAPI(); app.include_router(router_module.router)
    client = TestClient(app)
    with session_factory() as s:
        before = s.scalar(select(func.count()).select_from(ResearchPeerStat))
    base = "/api/research"
    good = client.get(f"{base}/players/{ids['player']}?competition=EPL&season=2025")
    assert good.status_code == 200
    for route in (f"teams/{ids['team']}", f"referees/{ids['referee']}"):
        assert client.get(f"{base}/{route}").status_code == 200
    for route in ("players/99999", "teams/99999", "referees/no-such-ref", f"players/{ids['player']}?season=1990"):
        assert client.get(f"{base}/{route}").status_code == 404
    for route in ("players/0", "players/no", "players/1?season=bad", "players/1?season=1800",
                  "players/1?competition=BAD", "search?type=other&q=a", "search?type=players&q=%20",
                  "search?type=players&q=a&limit=101", "search?type=players&q=a&competition=BAD"):
        assert client.get(f"{base}/{route}").status_code == 422
    results = client.get(f"{base}/search?type=players&q=Alex&competition=EPL").json()["results"]
    assert {r["id"] for r in results} == {ids["player"], ids["other_player"]}
    assert client.get(f"{base}/search?type=players&q=%25").json()["results"] == []
    assert len(client.get(f"{base}/search?type=referees&q=Taylor&competition=UCL").json()["results"]) == 1
    assert client.post(f"{base}/players/1").status_code == 405
    with session_factory() as s:
        assert s.scalar(select(func.count()).select_from(ResearchPeerStat)) == before


def test_default_competition_uses_minutes_or_matches_then_code(research, session_factory):
    service, ids = research
    # The player competes in two competitions in the same season. The most
    # minutes must win even when the smaller competition has a later fixture.
    with session_factory() as s:
        cup = s.scalar(select(Competition).where(Competition.code == "UCL"))
        fixture = s.scalar(select(Fixture).where(Fixture.competition_id == cup.id))
        s.add(FixturePlayerStats(fixture_id=fixture.id, player_id=ids["player"], team_id=ids["team"],
                                position="F", minutes=90, goals=0, substitute=False))
    player = service.profile("player", ids["player"], season=2025)
    team = service.profile("team", ids["team"], season=2025)
    assert player["competition"]["code"] == "EPL"
    assert team["competition"]["code"] == "EPL"
    low = service.profile("player", ids["low"], season=2025)
    assert low["previous_season"] is None
    assert low["ranking"]["reason"] == "below_minutes_threshold"


def test_missing_build_returns_counts_without_manufacturing_ranks(session_factory):
    # Read endpoints must never build ranks as a side effect.
    with session_factory() as s:
        c = Competition(code="EPL", name="League", api_football_id=39)
        s.add(c); s.flush()
        season = Season(competition_id=c.id, year=2026, label="2026/27")
        a, b = Team(name="A", api_football_id=1), Team(name="B", api_football_id=2)
        s.add_all([season, a, b]); s.flush()
        f = Fixture(api_football_id=1, competition_id=c.id, season_id=season.id,
                    home_team_id=a.id, away_team_id=b.id, status="FT", home_goals=1, away_goals=0)
        s.add(f); s.flush()
        s.add(FixtureTeamStats(fixture_id=f.id, team_id=a.id, opponent_team_id=b.id, is_home=True, goals=1))
        key = a.id
    service = ResearchProfileService(ResearchRepository(session_factory))
    result = service.profile("team", key)
    assert result["totals"]["matches"] == 1
    # One match is below the team minimum, so it is unranked whatever the build state.
    assert result["ranking"]["ranked"] is False
    assert result["ranking"]["reason"] == "below_matches_threshold"
    assert result["data_basis"]["build_id"] is None
    assert result["frequencies"]["over_2_5_goals"]["denominator"] == 1
    assert result["frequencies"]["clean_sheet"]["count"] == 1
    assert result["frequencies"]["corners_10_plus"]["denominator"] == 0
    with session_factory() as s:
        assert s.scalar(select(func.count()).select_from(ResearchProfileBuild)) == 0


def test_migration_round_trip_keeps_legacy_tables(settings):
    from pathlib import Path
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect
    root = Path(__file__).resolve().parents[2] / "data_platform"
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "alembic"))
    command.upgrade(cfg, "0008_match_read_cycle")
    db = create_engine(settings.database_url)
    before = set(inspect(db).get_table_names())
    command.upgrade(cfg, "head")
    assert set(inspect(db).get_table_names()) - before == {
        "research_peer_stats", "research_profile_builds", "research_referee_aliases"}
    command.downgrade(cfg, "0008_match_read_cycle")
    assert set(inspect(db).get_table_names()) == before
    command.upgrade(cfg, "head")
    assert "ix_research_player_fixture" in {i["name"] for i in inspect(db).get_indexes("fixture_player_stats")}
    db.dispose()


@pytest.mark.parametrize("starts,role", [(0, "mostly_substitute"), (2, "mostly_substitute"),
                                         (3, "rotation"), (6, "rotation"), (7, "regular_starter"),
                                         (10, "regular_starter")])
def test_role_boundaries(starts, role):
    summary = player_role(starts, 10, 850)
    assert summary == {"role": role, "starts": starts, "appearances": 10, "start_rate": starts / 10,
                       "average_minutes_per_appearance": 85}
    assert player_role(0, 0, 0)["role"] is None


def test_team_score_only_and_one_empty_side_preserve_results_not_statistics():
    base = dict(status="FT", team_id=1, home_team_id=1, away_team_id=2, home_goals=2, away_goals=0)
    missing = {**base, "goals": 99, "opponent_goals": 99, "expected_goals": 9.9,
               "shots_blocked": 3, "opponent_corners": 1}  # none of the eight subject evidence fields
    observed = {**base, "shots_total": 1, "opponent_shots_on": 0,
                "corners": None, "yellow_cards": 2, "opponent_yellow_cards": 2}
    result = team_summary([missing, observed])
    assert result["totals"]["matches"] == 2
    assert result["totals"]["wins"] == 2
    assert result["totals"]["goals_for"] == 4  # fixture scores, not bogus stored team goals
    assert result["totals"]["goals_against"] == 0
    assert result["totals"]["clean_sheets"] == 2
    assert result["totals"]["stats_observed_matches"] == 1
    assert result["metrics"]["cards"]["value"] == 2
    assert result["metrics"]["xg_for"]["value"] is None
    assert result["metrics"]["possession"]["value"] is None
    assert result["frequencies"]["clean_sheet"] == {"count": 2, "denominator": 2, "unknown": 0}
    assert result["frequencies"]["cards_4_plus"] == {"count": 1, "denominator": 1, "unknown": 1}


def test_rating_and_shooting_ratios_use_the_right_denominators():
    summary = player_summary([dict(minutes=90, goals=1, shots_total=2, shots_on=1, rating=8),
                              dict(minutes=90, goals=None, shots_total=8, shots_on=2, rating=None)])
    assert summary["metrics"]["average_rating"]["value"] == 8
    assert summary["metrics"]["average_rating"]["coverage"] == {"observed_matches": 1, "missing_matches": 1}
    assert summary["metrics"]["shooting_accuracy_pct"]["value"] == 30
    assert summary["metrics"]["conversion_pct"]["value"] == 10


def test_referee_empty_sides_do_not_meet_ranking_threshold():
    good = dict(home_stats_shots_on=0, away_stats_shots_on=0, home_stats_yellow_cards=2, away_stats_yellow_cards=2)
    empty = dict(home_goals=1, away_goals=0, home_stats_goals=1, away_stats_goals=0)
    result = referee_summary([good] * 14 + [empty] * 2)
    assert result["matches"] == 14
    assert result["totals"]["finished_matches"] == 16
    assert result["totals"]["excluded_matches"] == 2
    assert result["metrics"]["cards"]["value"] == 4
    assert eligibility("referee", result) == (False, "below_matches_threshold")


@pytest.mark.parametrize("kind,name,pct,higher,rank,phrase", [
    ("player", "goals_per_90", 99, True, 1, "more than 99% of Premier League forwards"),
    ("referee", "fouls", 15, True, 17, "fewer than 85% of referees"),
    ("team", "goals_against", 2, False, 1, "the fewest of 20 teams"),
    ("team", "goals_for", 93, True, 2, "the 2nd most of 20 teams"),
    ("team", "goals_for", 98, True, 1, "the most of 20 teams"),
])
def test_structured_headline_words_and_rank_phrases(kind, name, pct, higher, rank, phrase):
    import re
    result = headline(kind, {"name": "Person"}, {"name": "Premier League"},
                      {name: {"value": .8215, "percentile": pct, "rank": rank, "peer_count": 20,
                              "unit": "per_90" if kind == "player" else "per_match", "higher_is_more": higher,
                              "coverage": {"missing_matches": 0}}}, "F")
    assert result["display_value"] == "0.82"
    assert phrase in result["text"]
    assert not re.search(r"\b(percentile|records|bet|tip|value)\b", result["text"], re.I)
    assert result["rank_phrase"] == (phrase if kind == "team" else None)
    assert result["percentile"] == (None if kind == "team" else pct)
    assert result["comparison_share"] == (None if kind == "team" else pct if pct >= 50 else 100 - pct)
    assert "missing" not in result["text"]


def test_headline_updated_priorities_on_equal_extremeness():
    def m(pct):
        return dict(value=1, percentile=pct, rank=1, peer_count=20, unit="per_match",
                    higher_is_more=True, coverage={"missing_matches": 0})
    for kind, names, expected in [
        ("player", ("assists_per_90", "shots_on_target_per_90"), "shots_on_target_per_90"),
        ("team", ("goals_for", "goals_against"), "goals_against"),
        ("referee", ("fouls", "cards_per_foul"), "cards_per_foul"),
    ]:
        result = headline(kind, {}, {"name": "League"}, {names[0]: m(10), names[1]: m(90)}, "F")
        assert result["metric"] == expected


def test_squad_leader_ties_minutes_name_then_identity():
    def row(pid, name, minutes, goals):
        return dict(player_id=pid, name=name, minutes=minutes, goals=goals,
                    assists=0, shots_on=0, yellow_cards=0, appearances=2)
    result = squad_leaders([row(1, "Zed", 100, 2), row(2, "Amy", 90, 2),
                            row(3, "Aaron", 90, 2), row(4, "Top", 180, 3)])
    assert [r["player_id"] for r in result["goals"]] == [4, 3, 2]
    assert result["goals"][0] == {"player_id": 4, "name": "Top", "value": 3, "appearances": 2}


def test_context_scopes_and_chart_fields(research):
    service, ids = research
    result = service.profile("player", ids["player"], season=2026)
    assert result["subject"]["club"]["name"] == "Alpha"
    assert result["subject"]["position"] == {"code": "F", "label": "Forward"}
    assert result["subject"]["photo_url"].endswith("/200.png")
    assert result["role_summary"]["role"] == "regular_starter"
    assert [(r["season"]["year"], r["matches"], r["ranked"]) for r in result["available_scopes"]] == [(2026, 2, False), (2025, 15, True)]
    assert result["available_scopes"][0]["minutes"] == 120
    assert result["last_10"][0]["values"]["shots_off_target"] == 2
    assert result["last_10"][0]["opponent"]["abbreviation"] == "BET"
    assert result["shooting_funnel"] == {"shots": 6, "shots_on_target": 2, "goals": 2}
    referee = service.profile("referee", ids["referee"], season=2025)
    assert referee["subject"]["country"] == "England"
    assert {r["competition"]: r["small_sample"] for r in referee["competition_breakdown"]} == {"EPL": False, "UCL": True}
    assert referee["available_scopes"][0]["competition"]["code"] == "ALL"
    assert referee_country(["A. Taylor"]) is None


def test_latest_club_scopes_and_squad_leaders_follow_actual_team(research, session_factory):
    service, ids = research
    with session_factory() as s:
        season = s.scalar(select(Season).where(Season.year == 2025, Season.competition_id == ids["competition"]))
        last = s.scalar(select(FixturePlayerStats).join(Fixture, Fixture.id == FixturePlayerStats.fixture_id).where(
            FixturePlayerStats.player_id == ids["player"], Fixture.season_id == season.id,
            Fixture.status == "FT").order_by(Fixture.kickoff_utc.desc()).limit(1))
        beta = s.scalar(select(Team).where(Team.name == "Beta"))
        last.team_id = beta.id
        cup_fixture = s.scalar(select(Fixture).join(Competition).where(Competition.code == "UCL"))
        s.add(FixturePlayerStats(fixture_id=cup_fixture.id, team_id=ids["team"], player_id=ids["player"],
                                minutes=90, position="F", goals=100))
    result = service.profile("player", ids["player"], competition="EPL", season=2025)
    assert result["subject"]["club"]["name"] == "Beta"
    assert any(r["competition"]["code"] == "UCL" for r in result["available_scopes"])
    team = service.profile("team", ids["team"], competition="EPL", season=2025)
    assert team["squad_leaders"]["goals"][0]["value"] == 14  # excludes other-club game and the cup goal tally
    assert team["subject"]["short_code"] == "ALP"


def test_search_context_and_recency_order_before_limit(research, session_factory):
    service, ids = research
    with session_factory() as s:
        old_player = s.scalar(select(Player).where(Player.id == ids["player"]))
        old_player.name = "Zulu Match"
        another = s.scalar(select(Player).where(Player.id == ids["other_player"]))
        another.name = "Alpha Match"
        # Remove the latter's latest-season appearances, leaving a name that
        # sorts first alphabetically but last by real activity.
        for row in s.scalars(select(FixturePlayerStats).join(Fixture).join(Season).where(
            FixturePlayerStats.player_id == another.id, Season.year == 2026)):
            row.minutes = 0
    result = service.search("players", "Match", limit=1)["results"][0]
    assert result["name"] == "Zulu Match"
    assert result["club"]["name"] == "Alpha"
    assert result["position"] == {"code": "F", "label": "Forward"}
    assert (result["latest_competition"], result["latest_season"]) == ("EPL", 2026)
    team = service.search("teams", "Alpha")["results"][0]
    assert (team["country"], team["latest_competition"]) == ("England", "EPL")
    ref = service.search("referees", "Taylor")["results"][0]
    assert ref["country"] == "England"
    assert ref["matches"] == 19  # finished across both seasons, all aliases, excluding NS


# --- Comparable rounds, team minimum, rank wording and position-aware headlines ---

@pytest.mark.parametrize("rank, higher, expected", [
    (1, True, "the most of 20 teams"), (2, True, "the 2nd most of 20 teams"),
    (19, True, "the 2nd fewest of 20 teams"), (20, True, "the fewest of 20 teams"),
    (1, False, "the fewest of 20 teams"), (20, False, "the most of 20 teams"),
    (10, True, "the 10th most of 20 teams"), (11, True, "the 10th fewest of 20 teams"),
])
def test_rank_phrase_counts_from_the_nearer_end(rank, higher, expected):
    from data_platform.services.research_profiles import rank_phrase
    assert rank_phrase(rank, 20, higher) == expected


def test_teams_below_match_minimum_are_not_ranked():
    from data_platform.services.research_profiles import TEAM_MATCHES, rank_rows
    def summary(observed, value):
        return {"matches": observed, "stats_observed_matches": observed, "minutes": None, "peer_group": "all",
                "metrics": {"goals_for": {"value": value, "higher_is_more": True,
                                          "coverage": {"observed_matches": observed}}}}
    rows = rank_rows("team", {"league": summary(34, 1.5), "playoff_side": summary(2, 3.0),
                              "edge": summary(TEAM_MATCHES, 1.0)}, {}, {})
    assert {r["subject_key"] for r in rows} == {"league", "edge"}
    assert all(r["peer_count"] == 2 for r in rows)


@pytest.mark.parametrize("group, metrics, expected", [
    # A goalkeeper never leads with attacking output, even when it is the most extreme.
    ("G", {"assists_per_90": (0.05, 99), "average_rating": (6.9, 80), "pass_accuracy_pct": (71.0, 60)}, "average_rating"),
    ("D", {"goals_per_90": (0.3, 99), "duel_win_pct": (64.0, 85), "cards_per_90": (0.1, 30)}, "duel_win_pct"),
    ("M", {"assists_per_90": (0.31, 95), "pass_accuracy_pct": (88.0, 70)}, "assists_per_90"),
    ("F", {"goals_per_90": (0.82, 99), "average_rating": (7.3, 100)}, "goals_per_90"),
])
def test_headline_uses_position_specific_metrics(group, metrics, expected):
    data = {name: {"value": value, "percentile": pct, "rank": 1, "peer_count": 40,
                   "unit": "percent" if name.endswith("_pct") else "rating" if name == "average_rating" else "per_90",
                   "coverage": {"missing_matches": 0}} for name, (value, pct) in metrics.items()}
    result = headline("player", {"name": "Sample"}, {"name": "League"}, data, group)
    assert result["metric"] == expected


def test_headline_never_leads_with_a_zero():
    data = {"average_rating": {"value": 0, "percentile": 2, "rank": 40, "peer_count": 40, "unit": "rating",
                               "coverage": {"missing_matches": 0}},
            "pass_accuracy_pct": {"value": 74.4, "percentile": 55, "rank": 18, "peer_count": 40, "unit": "percent",
                                  "coverage": {"missing_matches": 0}}}
    result = headline("player", {"name": "Keeper"}, {"name": "La Liga"}, data, "G")
    assert result["metric"] == "pass_accuracy_pct"
    assert result["text"].startswith("74% pass accuracy")
    assert headline("player", {}, {"name": "League"}, {"assists_per_90": {"value": 0.0, "percentile": 40, "rank": 9,
                    "peer_count": 40, "unit": "per_90", "coverage": {"missing_matches": 0}}}, "M") is None


def test_comparable_rounds_drop_playoffs_and_qualifiers(session_factory):
    from data_platform.repositories.research import comparable_rounds
    with session_factory() as s:
        league = Competition(code="BL", name="League", api_football_id=78, competition_type="domestic_league")
        cup = Competition(code="CUP", name="Cup", api_football_id=2, competition_type="continental_cup")
        s.add_all([league, cup]); s.flush()
        rounds = {league.id: ["Regular Season - 1", "Relegation Round", "Promotion Play-offs - Final"],
                  cup.id: ["1st Qualifying Round", "Play-offs", "Playoff round", "Preliminary Round",
                           "League Stage - 1", "Group A - 2", "Knockout Round Play-offs", "Round of 16",
                           "16th Finals", "Quarter-finals", "Semi-finals", "Final"]}
        for i, (comp_id, names) in enumerate(rounds.items()):
            season = Season(competition_id=comp_id, year=2025, label="2025/26"); s.add(season); s.flush()
            home, away = Team(name=f"H{i}", api_football_id=900 + i), Team(name=f"A{i}", api_football_id=950 + i)
            s.add_all([home, away]); s.flush()
            for j, name in enumerate(names):
                s.add(Fixture(api_football_id=10000 + i * 100 + j, competition_id=comp_id, season_id=season.id,
                              home_team_id=home.id, away_team_id=away.id, status="FT", round=name))
        s.flush()
        kept = set(s.scalars(select(Fixture.round).join(Competition, Competition.id == Fixture.competition_id)
                             .where(comparable_rounds())))
    assert kept == {"Regular Season - 1", "League Stage - 1", "Group A - 2", "Knockout Round Play-offs",
                    "Round of 16", "16th Finals", "Quarter-finals", "Semi-finals", "Final"}


def test_match_log_covers_whole_scope_for_charts(research):
    service, ids = research
    player = service.profile("player", ids["player"], competition="EPL", season=2025)
    assert len(player["match_log"]) == player["totals"]["appearances"] == 15
    assert player["match_log"][-10:][-1]["fixture_id"] == player["last_10"][-1]["fixture_id"]
    assert [m["date"] for m in player["match_log"]] == sorted(m["date"] for m in player["match_log"])
    first = player["match_log"][0]
    assert {"assists", "passes", "fouls_drawn", "cards", "rating"} <= set(first["values"])
    assert isinstance(first["started"], bool)
    assert sum(m["values"]["goals"] for m in player["match_log"]) == player["totals"]["goals"]

    team = service.profile("team", ids["team"], competition="EPL", season=2025)
    assert len(team["match_log"]) == team["totals"]["matches"]
    observed = [m for m in team["match_log"] if m["stats_observed"]]
    assert all(m["values"]["possession"] > 1 for m in observed if m["values"]["possession"] is not None)
    # A fixture without both teams' statistics keeps every stat unknown, never zero.
    missing = [m for m in team["match_log"] if not m["stats_observed"]]
    assert missing and all(m["values"]["corners_for"] is None for m in missing)

    referee = service.profile("referee", ids["referee"], season=2025)
    for match in referee["match_log"]:
        v = match["values"]
        if v["cards"] is not None:
            assert v["home_cards"] + v["away_cards"] == v["cards"]
            assert v["home_fouls"] + v["away_fouls"] == v["fouls"]
    # Rank-only payloads (previous season) do not carry a match log.
    current = service.profile("player", ids["player"])
    assert "match_log" not in current["previous_season"]
