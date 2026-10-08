"""Narrow canonical SQLAlchemy queries for research; no CSVs or provider calls."""
from decimal import Decimal

from sqlalchemy import and_, case, delete, func, insert, or_, select
from sqlalchemy.orm import aliased

from ..db import session_scope
from ..models import (Competition, Fixture, FixturePlayerStats, FixtureTeamStats,
                      Player, ResearchPeerStat, ResearchProfileBuild,
                      ResearchRefereeAlias, Season, Team)
from ..research_stats import (PLAYER_COUNT_FIELDS, PLAYER_MEASUREMENT_FIELDS,
                              TEAM_COUNT_FIELDS, TEAM_MEASUREMENT_FIELDS, TEAM_OBSERVATION_FIELDS,
                              RANK_BUILD_PREFIX)

FINISHED = ("FT", "AET", "PEN")
COMPETITION_PHASE = ("domestic leagues: regular season only; continental cups: league/group stage and "
                     "knockouts, excluding preliminary, qualifying and pre-group play-off rounds")


def comparable_rounds(fixture=Fixture, competition=Competition):
    """Rounds that every team in a competition-season plays on comparable terms.

    Domestic play-offs (relegation, promotion, split-season groups) and European
    qualifiers would otherwise put sides with one or two matches into league peer
    groups. Unknown competition types and missing round labels are kept.
    """
    r = fixture.round
    main_stage = or_(r.like("League Stage%"), r.like("Group%"), r.like("Round of%"), r.like("Knockout%"),
                     r.like("Quarter%"), r.like("Semi%"), r.like("Final%"), r.like("%th Finals"))
    return or_(r.is_(None),
               and_(competition.competition_type == "domestic_league", r.like("Regular Season%")),
               and_(competition.competition_type == "continental_cup", main_stage),
               competition.competition_type.is_(None),
               competition.competition_type.notin_(("domestic_league", "continental_cup")))
# Keep source stats separate from the fixture score and availability policy.
PLAYER_FIELDS = (("player_id", "team_id", "position", "minutes", "substitute")
                 + PLAYER_COUNT_FIELDS + PLAYER_MEASUREMENT_FIELDS)
TEAM_FIELDS = TEAM_COUNT_FIELDS + TEAM_MEASUREMENT_FIELDS


def records(session, query):
    return [{k: float(v) if isinstance(v, Decimal) else v for k, v in row.items()}
            for row in session.execute(query).mappings()]


class ResearchRepository:
    def __init__(self, session_factory=session_scope):
        self.session_factory = session_factory

    def competition(self, session, code):
        if code is None:
            return None
        result = session.scalar(select(Competition).where(Competition.code == code))
        if result is None:
            raise ValueError("Unknown competition code")
        return result.id

    def scopes(self, session, *, year=None, competition_id=None):
        q = select(Season.id.label("season_id"), Season.year, Season.label,
                   Competition.id.label("competition_id"), Competition.code, Competition.name
                   ).join(Competition, Competition.id == Season.competition_id)
        if year is not None:
            q = q.where(Season.year == year)
        if competition_id is not None:
            q = q.where(Competition.id == competition_id)
        return records(session, q.order_by(Season.year, Competition.code))

    def subject(self, session, kind, key):
        if kind == "referee":
            rows = records(session, select(ResearchRefereeAlias).with_only_columns(
                ResearchRefereeAlias.raw_name, ResearchRefereeAlias.name,
                ResearchRefereeAlias.referee_key).where(ResearchRefereeAlias.referee_key == key))
            return ({"referee_key": key, "name": rows[0]["name"],
                     "aliases": sorted(r["raw_name"] for r in rows)} if rows else None)
        model = Player if kind == "player" else Team
        cols = [model.id, model.name, model.api_football_id]
        cols += [Player.nationality] if kind == "player" else [Team.logo_url, Team.short_code, Team.country]
        rows = records(session, select(*cols).where(model.id == key))
        return rows[0] if rows else None

    def select_scope(self, session, kind, key, *, year=None, competition_id=None):
        f = Fixture
        q = select(Season.id.label("season_id"), Season.year, Season.label,
                   Competition.id.label("competition_id"), Competition.code, Competition.name)
        if kind == "player":
            p = FixturePlayerStats
            q = q.add_columns(func.sum(p.minutes).label("weight")).select_from(p).join(f, f.id == p.fixture_id)
            q = q.where(p.player_id == key, p.minutes > 0)
        else:
            q = q.add_columns(func.count().label("weight")).select_from(f)
            q = q.where(or_(f.home_team_id == key, f.away_team_id == key))
        q = q.join(Season, Season.id == f.season_id).join(Competition, Competition.id == f.competition_id)
        q = q.where(f.status.in_(FINISHED), comparable_rounds())
        if year is not None:
            q = q.where(Season.year == year)
        if competition_id is not None:
            q = q.where(f.competition_id == competition_id)
        q = q.group_by(Season.id, Season.year, Season.label, Competition.id, Competition.code, Competition.name)
        result = records(session, q.order_by(Season.year.desc(), q.selected_columns.weight.desc(), Competition.code).limit(1))
        return result[0] if result else None

    def player_rows(self, session, season_id, player_id=None):
        p, f = FixturePlayerStats, Fixture
        q = select(*[getattr(p, c) for c in PLAYER_FIELDS],
                   f.id.label("fixture_id"), f.kickoff_utc, f.status, f.home_team_id,
                   f.away_team_id, f.home_goals, f.away_goals).join(f, f.id == p.fixture_id)
        q = q.join(Competition, Competition.id == f.competition_id)
        q = q.where(f.season_id == season_id, f.status.in_(FINISHED), p.minutes > 0, comparable_rounds(),
                    or_(p.team_id == f.home_team_id, p.team_id == f.away_team_id))
        if player_id is not None:
            q = q.where(p.player_id == player_id)
        return records(session, q.order_by(f.kickoff_utc.asc().nullsfirst(), f.id))

    def team_rows(self, session, season_id, team_id=None):
        t, f, opp = FixtureTeamStats, Fixture, aliased(FixtureTeamStats)
        opponent_id = case((Team.id == f.home_team_id, f.away_team_id), else_=f.home_team_id)
        q = select(Team.id.label("team_id"), (Team.id == f.home_team_id).label("is_home"),
                   opponent_id.label("opponent_team_id"),
                   *[getattr(t, c) for c in TEAM_FIELDS],
                   *[getattr(opp, c).label("opponent_" + c) for c in TEAM_FIELDS],
                   f.id.label("fixture_id"), f.kickoff_utc, f.status, f.home_team_id,
                   f.away_team_id, f.home_goals, f.away_goals).select_from(f)
        q = q.join(Team, or_(Team.id == f.home_team_id, Team.id == f.away_team_id))
        q = q.outerjoin(t, and_(t.fixture_id == f.id, t.team_id == Team.id))
        q = q.outerjoin(opp, and_(opp.fixture_id == f.id, opp.team_id == opponent_id))
        q = q.join(Competition, Competition.id == f.competition_id)
        q = q.where(f.season_id == season_id, f.status.in_(FINISHED), comparable_rounds())
        if team_id is not None:
            q = q.where(Team.id == team_id)
        return records(session, q.order_by(f.kickoff_utc.asc().nullsfirst(), f.id))

    def referee_rows(self, session, year=None, aliases=None):
        f, home, away = Fixture, aliased(FixtureTeamStats), aliased(FixtureTeamStats)
        q = select(f.id.label("fixture_id"), f.referee, f.kickoff_utc, f.status,
                   f.home_team_id, f.away_team_id, f.home_goals, f.away_goals,
                   Season.year, Competition.code.label("competition"), Competition.name.label("competition_name"),
                   *[getattr(home, c).label("home_stats_" + c) for c in TEAM_FIELDS],
                   *[getattr(away, c).label("away_stats_" + c) for c in TEAM_FIELDS])
        q = q.join(Season, Season.id == f.season_id).join(Competition, Competition.id == f.competition_id)
        q = q.outerjoin(home, and_(home.fixture_id == f.id, home.team_id == f.home_team_id))
        q = q.outerjoin(away, and_(away.fixture_id == f.id, away.team_id == f.away_team_id))
        q = q.where(f.status.in_(FINISHED), f.referee.is_not(None))
        if year is not None:
            q = q.where(Season.year == year)
        if aliases is not None:
            q = q.where(f.referee.in_(aliases))
        return records(session, q.order_by(f.kickoff_utc.asc().nullsfirst(), f.id))

    def team_details(self, session, ids):
        if not ids:
            return {}
        rows = records(session, select(Team.id, Team.name, Team.short_code, Team.logo_url).where(Team.id.in_(ids)))
        return {r["id"]: r for r in rows}

    def available_scopes(self, session, kind, key, aliases=()):
        f, p = Fixture, FixturePlayerStats
        if kind == "referee":
            home, away = aliased(FixtureTeamStats), aliased(FixtureTeamStats)
            observed = and_(*[or_(*[getattr(side, field).is_not(None) for field in TEAM_OBSERVATION_FIELDS])
                              for side in (home, away)])
            q = select(Season.year, func.count().label("finished_matches"),
                       func.sum(case((observed, 1), else_=0)).label("matches")).select_from(f)
            q = q.join(Season, Season.id == f.season_id)
            q = q.outerjoin(home, and_(home.fixture_id == f.id, home.team_id == f.home_team_id))
            q = q.outerjoin(away, and_(away.fixture_id == f.id, away.team_id == f.away_team_id))
            q = q.where(f.status.in_(FINISHED), f.referee.in_(aliases)).group_by(Season.year)
            rows = records(session, q.order_by(Season.year.desc()))
            for row in rows:
                row.update(code="ALL", name="All competitions", label=f"{row['year']}/{str(row['year']+1)[-2:]}",
                           competition_id=None, minutes=None)
        else:
            cols = [Season.year, Season.label, Season.id.label("season_id"), Competition.id.label("competition_id"),
                    Competition.code, Competition.name, func.count().label("matches")]
            if kind == "player":
                q = select(*cols, func.sum(p.minutes).label("minutes")).select_from(p).join(f, f.id == p.fixture_id)
                q = q.where(p.player_id == key, p.minutes > 0)
            else:
                q = select(*cols).select_from(f).where(or_(f.home_team_id == key, f.away_team_id == key))
            q = q.join(Season, Season.id == f.season_id).join(Competition, Competition.id == f.competition_id)
            q = q.where(f.status.in_(FINISHED), comparable_rounds()).group_by(Season.year, Season.label, Season.id,
                                                       Competition.id, Competition.code, Competition.name)
            rows = records(session, q.order_by(Season.year.desc(), Competition.code))
        stored = records(session, select(ResearchPeerStat.scope_key, ResearchPeerStat.sample_matches,
                                         ResearchPeerStat.sample_minutes).join(
            ResearchProfileBuild, ResearchProfileBuild.scope_key == ResearchPeerStat.scope_key).where(
                ResearchPeerStat.subject_type == kind, ResearchPeerStat.subject_key == str(key),
                ResearchProfileBuild.build_id.startswith(RANK_BUILD_PREFIX)).distinct())
        ranks = {(r["scope_key"], r["sample_matches"], r["sample_minutes"]) for r in stored}
        for row in rows:
            scope = f"referee:{row['year']}" if kind == "referee" else f"competition:{row['competition_id']}:{row['year']}"
            row["ranked"] = (scope, row["matches"], row.get("minutes")) in ranks
        return rows

    def squad_leaders(self, session, team_id, season_id):
        p, f = FixturePlayerStats, Fixture
        fields = ("goals", "assists", "shots_on", "yellow_cards")
        q = select(p.player_id, Player.name, func.count().label("appearances"),
                   func.sum(p.minutes).label("minutes"),
                   *[func.sum(func.coalesce(getattr(p, field), 0)).label(field) for field in fields])
        q = q.join(Player, Player.id == p.player_id).join(f, f.id == p.fixture_id)
        q = q.join(Competition, Competition.id == f.competition_id)
        q = q.where(p.team_id == team_id, p.minutes > 0, f.season_id == season_id, f.status.in_(FINISHED),
                    comparable_rounds())
        return records(session, q.group_by(p.player_id, Player.name))

    def referee_names(self, session):
        return list(session.scalars(select(Fixture.referee).where(Fixture.referee.is_not(None)).distinct().order_by(Fixture.referee)))

    def ranks(self, session, scope_key, kind, key):
        build = session.get(ResearchProfileBuild, scope_key)
        rows = records(session, select(*ResearchPeerStat.__table__.columns).where(
            ResearchPeerStat.scope_key == scope_key, ResearchPeerStat.subject_type == kind,
            ResearchPeerStat.subject_key == str(key)))
        return build, rows

    def replace(self, session, builds, rows, aliases):
        """Caller commits once: failures retain the previous complete build scope."""
        keys = [b["scope_key"] for b in builds]
        session.execute(delete(ResearchPeerStat).where(ResearchPeerStat.scope_key.in_(keys)))
        session.execute(delete(ResearchProfileBuild).where(ResearchProfileBuild.scope_key.in_(keys)))
        if builds:
            session.execute(insert(ResearchProfileBuild), builds)
        for start in range(0, len(rows), 2000):
            session.execute(insert(ResearchPeerStat), rows[start:start + 2000])
        session.execute(delete(ResearchRefereeAlias))
        if aliases:
            session.execute(insert(ResearchRefereeAlias), aliases)

    def search(self, session, kind, q, competition_id, limit):
        # Literal substring, latest activity before LIMIT, deterministic tie breaks.
        term = "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        f = Fixture
        if kind == "referees":
            a = ResearchRefereeAlias
            matched_keys = select(a.referee_key).where(or_(a.name.ilike(term, escape="\\"), a.raw_name.ilike(term, escape="\\")))
            query = select(a.referee_key, a.name, func.count().label("matches"),
                           func.max(f.kickoff_utc).label("last_activity_at")).select_from(a).join(f, f.referee == a.raw_name)
            query = query.where(a.referee_key.in_(matched_keys), f.status.in_(FINISHED))
            if competition_id is not None:
                query = query.where(f.competition_id == competition_id)
            query = query.group_by(a.referee_key, a.name).order_by(
                func.max(f.kickoff_utc).desc().nullslast(), a.name, a.referee_key).limit(limit)
            rows = records(session, query)
            aliases = records(session, select(a.referee_key, a.raw_name).where(a.referee_key.in_([r["referee_key"] for r in rows])))
            for row in rows:
                row["_aliases"] = [a["raw_name"] for a in aliases if a["referee_key"] == row["referee_key"]]
            return rows
        model = Player if kind == "players" else Team
        order = (f.kickoff_utc.desc().nullslast(), f.id.desc())
        common = [model.id, model.name, model.api_football_id, f.kickoff_utc.label("last_activity_at"),
                  Competition.code.label("latest_competition"), Season.year.label("latest_season"),
                  func.row_number().over(partition_by=model.id, order_by=order).label("row_number")]
        if kind == "players":
            p = FixturePlayerStats
            query = select(*common, p.position, Team.id.label("club_id"), Team.name.label("club_name"),
                           Team.logo_url.label("club_logo_url")).select_from(p).join(Player, Player.id == p.player_id)
            query = query.join(f, f.id == p.fixture_id).join(Team, Team.id == p.team_id).where(p.minutes > 0)
            # Restrict appearances by IDs before the window, so the existing
            # (player_id, fixture_id) index can avoid scanning all player stats.
            query = query.where(p.player_id.in_(select(Player.id).where(Player.name.ilike(term, escape="\\"))))
        else:
            query = select(*common, Team.country, Team.short_code, Team.logo_url).select_from(Team)
            query = query.join(f, or_(f.home_team_id == Team.id, f.away_team_id == Team.id))
        query = query.join(Season, Season.id == f.season_id).join(Competition, Competition.id == f.competition_id)
        query = query.where(model.name.ilike(term, escape="\\"), f.status.in_(FINISHED))
        if competition_id is not None:
            query = query.where(f.competition_id == competition_id)
        latest = query.subquery()
        return records(session, select(*[c for c in latest.c if c.name != "row_number"]).where(latest.c.row_number == 1)
                       .order_by(latest.c.last_activity_at.desc().nullslast(), latest.c.name, latest.c.id).limit(limit))
