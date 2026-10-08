"""Research Area landing data: this weekend's matches and season leaders.

Read-only and built from the canonical database. Season figures follow the
profile rules (finished, comparable-round matches; played player count nulls
are zero; team statistics need both sides observed). Referee card rates reuse
the ranked peer stats, so the landing page and profiles always agree.
"""
from collections import defaultdict
from datetime import timedelta
import time

from sqlalchemy import func, select
from sqlalchemy.orm import aliased

from ..db import session_scope
from ..models import (Competition, Fixture, FixturePlayerStats, Player, ResearchPeerStat,
                      ResearchProfileBuild, ResearchRefereeAlias, Season, Team)
from ..repositories.research import FINISHED, ResearchRepository, comparable_rounds
from ..research_stats import RANK_BUILD_PREFIX, normalize_research_row
from .research_profiles import REFEREE_MATCHES, add_known, iso, utcnow

TOP_FIVE = ("EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1")
EUROPE = ("UCL", "UEL", "UECL")
LEAGUE_GROUPS = {"top5": ("Top five leagues", TOP_FIVE), "europe": ("European cups", EUROPE)}
UPCOMING = ("NS", "TBD")
WEEKEND_DAYS = 7
WEEKEND_LIMIT = 12
LEADER_LIMIT = 10
TEAM_MIN_MATCHES = 3
REFEREE_MIN_RANKED = 10
CACHE_SECONDS = 600

PLAYER_LEADERS = {
    "goals": ("Goals", lambda p: func.coalesce(p.goals, 0)),
    "shots_on_target": ("Shots on target", lambda p: func.coalesce(p.shots_on, 0)),
    "assists": ("Assists", lambda p: func.coalesce(p.assists, 0)),
    "cards": ("Cards", lambda p: func.coalesce(p.yellow_cards, 0) + func.coalesce(p.red_cards, 0)),
}
# name: (label, row value, higher first)
TEAM_LEADERS = {
    "goals_for": ("Goals scored per match", lambda r: r.get("result_goals_for"), True),
    "goals_against": ("Fewest conceded per match", lambda r: r.get("result_goals_against"), False),
    "corners": ("Corners per match", lambda r: r.get("corners"), True),
    "cards": ("Cards per match", lambda r: add_known(r.get("yellow_cards"), r.get("red_cards")), True),
}


def photo(api_football_id):
    return f"https://media.api-sports.io/football/players/{api_football_id}.png" if api_football_id else None


def season_label(year):
    return f"{year}/{str(year + 1)[-2:]}"


def resolve_league(session, league):
    """Map a filter key to competition codes known to the database."""
    known = dict(session.execute(select(Competition.code, Competition.name)).all())
    key = league or "top5"
    if key in LEAGUE_GROUPS:
        label, codes = LEAGUE_GROUPS[key]
    elif key == "all":
        label, codes = "All competitions", tuple(sorted(known))
    elif key in known:
        label, codes = known[key], (key,)
    else:
        raise ValueError("Unknown league filter")
    codes = tuple(c for c in codes if c in known)
    return {"key": key, "label": label, "codes": list(codes)}


def standings(rows):
    """League table from finished fixture scores: points, goal difference, goals for, name."""
    table = defaultdict(lambda: {"played": 0, "won": 0, "drawn": 0, "lost": 0, "goals_for": 0, "goals_against": 0})
    names = {}
    for r in rows:
        if r["home_goals"] is None or r["away_goals"] is None:
            continue
        for team, name, gf, ga in ((r["home_team_id"], r["home_name"], r["home_goals"], r["away_goals"]),
                                   (r["away_team_id"], r["away_name"], r["away_goals"], r["home_goals"])):
            t = table[team]
            names[team] = name
            t["played"] += 1
            t["goals_for"] += gf
            t["goals_against"] += ga
            t["won" if gf > ga else "lost" if gf < ga else "drawn"] += 1
    ordered = sorted(table, key=lambda team: (-(3 * table[team]["won"] + table[team]["drawn"]),
                                              -(table[team]["goals_for"] - table[team]["goals_against"]),
                                              -table[team]["goals_for"], names[team]))
    return {team: {**table[team], "points": 3 * table[team]["won"] + table[team]["drawn"],
                   "position": i + 1, "teams": len(ordered)} for i, team in enumerate(ordered)}


def best_scorer(candidates):
    """Most goals, then shots on target, then fewer appearances, then name."""
    return min(candidates, key=lambda c: (-c["goals"], -c["shots_on_target"], c["appearances"], c["name"]))


def team_leader_rows(rows, limit=LEADER_LIMIT):
    """Per-match averages over observed matches, normalised like team profiles."""
    by_team = defaultdict(list)
    for row in rows:
        by_team[row["team_id"]].append(normalize_research_row("team", row))
    result = {}
    for name, (label, getter, higher) in TEAM_LEADERS.items():
        entries = []
        for team_id, team_rows in by_team.items():
            values = [getter(r) for r in team_rows]
            known = [v for v in values if v is not None]
            if len(known) < TEAM_MIN_MATCHES:
                continue
            entries.append({"team_id": team_id, "value": sum(known) / len(known), "matches": len(known)})
        entries.sort(key=lambda e: (-e["value"] if higher else e["value"], -e["matches"], e["team_id"]))
        result[name] = {"label": label, "unit": "per_match", "higher_first": higher, "rows": entries[:limit]}
    return result


class ResearchHomeService:
    _cache = {}

    def __init__(self, repo=None, clock=utcnow, cache_seconds=CACHE_SECONDS):
        self.repo = repo or ResearchRepository()
        self.clock = clock
        self.cache_seconds = cache_seconds

    def home(self, kind, *, league=None):
        if kind not in ("players", "teams", "referees"):
            raise ValueError("Unknown research area")
        now = self.clock()
        key = (kind, league or "top5", now.strftime("%Y-%m-%d %H"))
        cached = self._cache.get(key) if self.cache_seconds else None
        if cached and time.monotonic() - cached[0] < self.cache_seconds:
            return cached[1]
        with self.repo.session_factory() as session:
            scope = resolve_league(session, league)
            result = self._build(session, kind, scope, now)
        if self.cache_seconds:
            self._cache[key] = (time.monotonic(), result)
        return result

    # ----- shared lookups -----
    def _competitions(self, session, codes):
        rows = session.execute(select(Competition.id, Competition.code, Competition.name, Competition.competition_type)
                               .where(Competition.code.in_(codes))).mappings().all()
        return {r["id"]: dict(r) for r in rows}

    def _latest_finished(self, session, comp_ids):
        return session.scalar(select(func.max(Fixture.kickoff_utc)).where(
            Fixture.competition_id.in_(comp_ids), Fixture.status.in_(FINISHED)))

    def _current_year(self, session, comp_ids):
        return session.scalar(select(func.max(Season.year)).join(Fixture, Fixture.season_id == Season.id).join(
            Competition, Competition.id == Fixture.competition_id).where(
            Fixture.competition_id.in_(comp_ids), Fixture.status.in_(FINISHED), comparable_rounds()))

    def _seasons(self, session, comp_ids, year):
        rows = session.execute(select(Season.id, Season.competition_id).where(
            Season.competition_id.in_(comp_ids), Season.year == year)).all()
        return {competition_id: season_id for season_id, competition_id in rows}

    def _teams(self, session, ids):
        if not ids:
            return {}
        rows = session.execute(select(Team.id, Team.name, Team.short_code, Team.logo_url).where(Team.id.in_(ids))).mappings()
        return {r["id"]: dict(r) for r in rows}

    def _referees(self, session, raw_names):
        """Raw fixture names -> canonical key, display name and latest ranked card rate."""
        raw_names = {n for n in raw_names if n}
        if not raw_names:
            return {}
        aliases = session.execute(select(ResearchRefereeAlias.raw_name, ResearchRefereeAlias.referee_key,
                                         ResearchRefereeAlias.name).where(ResearchRefereeAlias.raw_name.in_(raw_names))).all()
        keys = {key for _, key, _ in aliases}
        rates = {}
        if keys:
            stats = session.execute(select(ResearchPeerStat.subject_key, ResearchPeerStat.season_year, ResearchPeerStat.value,
                                           ResearchPeerStat.percentile, ResearchPeerStat.peer_count,
                                           ResearchPeerStat.sample_matches)
                                    .join(ResearchProfileBuild, ResearchProfileBuild.scope_key == ResearchPeerStat.scope_key)
                                    .where(ResearchPeerStat.subject_type == "referee", ResearchPeerStat.metric == "cards",
                                           ResearchPeerStat.subject_key.in_(keys),
                                           ResearchProfileBuild.build_id.startswith(RANK_BUILD_PREFIX))).all()
            for key, year, value, pct, peers, matches in stats:
                if key not in rates or year > rates[key]["season_year"]:
                    rates[key] = {"season_year": year, "season_label": season_label(year), "cards_per_match": value,
                                  "percentile": pct, "peer_count": peers, "matches": matches}
        return {raw: {"referee_key": key, "name": name, "cards": rates.get(key)} for raw, key, name in aliases}

    # ----- weekend -----
    def _weekend(self, session, kind, comps, now):
        end = now + timedelta(days=WEEKEND_DAYS)
        q = (select(Fixture.id, Fixture.kickoff_utc, Fixture.round, Fixture.referee, Fixture.competition_id,
                    Fixture.season_id, Season.year, Fixture.home_team_id, Fixture.away_team_id)
             .join(Season, Season.id == Fixture.season_id)
             .where(Fixture.competition_id.in_(list(comps)), Fixture.status.in_(UPCOMING),
                    Fixture.kickoff_utc >= now, Fixture.kickoff_utc < end))
        if kind == "referees":
            q = q.where(Fixture.referee.is_not(None))
        fixtures = [dict(r) for r in session.execute(q.order_by(Fixture.kickoff_utc, Fixture.id).limit(WEEKEND_LIMIT)).mappings()]
        if not fixtures:
            return []
        team_ids = {f[k] for f in fixtures for k in ("home_team_id", "away_team_id")}
        teams = self._teams(session, team_ids)
        referees = self._referees(session, {f["referee"] for f in fixtures})
        tables = self._tables(session, {(f["competition_id"], f["season_id"]) for f in fixtures
                                        if comps[f["competition_id"]]["competition_type"] == "domestic_league"})
        scorers = self._top_scorers(session, team_ids, {f["year"] for f in fixtures}) if kind == "players" else {}
        out = []
        for f in fixtures:
            comp = comps[f["competition_id"]]
            table = tables.get((f["competition_id"], f["season_id"]), {})
            sides = {}
            for side in ("home", "away"):
                team_id = f[f"{side}_team_id"]
                entry = {"id": team_id, **{k: teams.get(team_id, {}).get(k) for k in ("name", "short_code", "logo_url")}}
                if kind == "teams":
                    entry["standing"] = table.get(team_id)
                if kind == "players":
                    entry["top_scorer"] = scorers.get((team_id, f["year"]))
                sides[side] = entry
            referee = referees.get(f["referee"]) if f["referee"] else None
            out.append({"fixture_id": f["id"], "kickoff": iso(f["kickoff_utc"]), "round": f["round"],
                        "competition": {"code": comp["code"], "name": comp["name"]}, "season": season_label(f["year"]),
                        "scope": f"{comp['code']}-{f['year']}",
                        **sides, "referee": referee if referee else ({"referee_key": None, "name": f["referee"], "cards": None}
                                                                     if f["referee"] else None)})
        return out

    def _tables(self, session, scopes):
        result = {}
        home, away = aliased(Team), aliased(Team)
        for competition_id, season_id in scopes:
            rows = session.execute(select(Fixture.home_team_id, Fixture.away_team_id, Fixture.home_goals, Fixture.away_goals,
                                          home.name.label("home_name"), away.name.label("away_name"))
                                   .join(home, home.id == Fixture.home_team_id).join(away, away.id == Fixture.away_team_id)
                                   .join(Competition, Competition.id == Fixture.competition_id)
                                   .where(Fixture.competition_id == competition_id, Fixture.season_id == season_id,
                                          Fixture.status.in_(FINISHED), comparable_rounds())).mappings().all()
            result[(competition_id, season_id)] = standings(rows)
        return result

    def _top_scorers(self, session, team_ids, years):
        """Each club's top scorer: league matches first, then any competition that season."""
        p, f = FixturePlayerStats, Fixture
        q = (select(p.team_id, Season.year, Competition.competition_type, p.player_id, Player.name, Player.api_football_id,
                    func.count().label("appearances"), func.sum(func.coalesce(p.goals, 0)).label("goals"),
                    func.sum(func.coalesce(p.shots_on, 0)).label("shots_on_target"),
                    Competition.code.label("competition"))
             .join(f, f.id == p.fixture_id).join(Season, Season.id == f.season_id)
             .join(Competition, Competition.id == f.competition_id).join(Player, Player.id == p.player_id)
             .where(p.team_id.in_(team_ids), Season.year.in_(years), p.minutes > 0, f.status.in_(FINISHED),
                    comparable_rounds())
             .group_by(p.team_id, Season.year, Competition.competition_type, Competition.code, p.player_id, Player.name,
                       Player.api_football_id))
        grouped = defaultdict(lambda: defaultdict(list))
        for r in session.execute(q).mappings():
            grouped[(r["team_id"], r["year"])][r["competition_type"] == "domestic_league"].append(dict(r))
        result = {}
        for key, buckets in grouped.items():
            pool = buckets[True] or buckets[False]
            pool = [c for c in pool if c["goals"] or c["shots_on_target"]]
            if pool:
                best = best_scorer(pool)
                result[key] = {"player_id": best["player_id"], "name": best["name"], "photo_url": photo(best["api_football_id"]),
                               "goals": best["goals"],
                               "shots_on_target": best["shots_on_target"], "appearances": best["appearances"],
                               "competition": best["competition"], "season": season_label(best["year"]),
                               "scope": f"{best['competition']}-{best['year']}"}
        return result

    # ----- leaders -----
    def _player_leaders(self, session, comps, year):
        p, f = FixturePlayerStats, Fixture
        result = {}
        for name, (label, expression) in PLAYER_LEADERS.items():
            value = func.sum(expression(p)).label("value")
            q = (select(p.player_id, Player.name, Player.api_football_id, p.team_id, Competition.code.label("competition"), value,
                        func.count().label("appearances"), func.sum(p.minutes).label("minutes"))
                 .join(f, f.id == p.fixture_id).join(Season, Season.id == f.season_id)
                 .join(Competition, Competition.id == f.competition_id).join(Player, Player.id == p.player_id)
                 .where(f.competition_id.in_(list(comps)), Season.year == year, p.minutes > 0,
                        f.status.in_(FINISHED), comparable_rounds())
                 .group_by(p.player_id, Player.name, Player.api_football_id, p.team_id, Competition.code)
                 .having(value > 0)
                 .order_by(value.desc(), func.count(), Player.name, p.player_id).limit(LEADER_LIMIT))
            rows = [dict(r) for r in session.execute(q).mappings()]
            teams = self._teams(session, {r["team_id"] for r in rows})
            for r in rows:
                r["team"] = {"id": r["team_id"], **{k: teams.get(r["team_id"], {}).get(k) for k in ("name", "short_code", "logo_url")}}
                r["scope"] = f"{r['competition']}-{year}"
                r["photo_url"] = photo(r.pop("api_football_id"))
            result[name] = {"label": label, "unit": "count", "higher_first": True, "rows": rows}
        return result

    def _team_leaders(self, session, comps, year):
        seasons = self._seasons(session, list(comps), year)
        rows = []
        for competition_id, season_id in seasons.items():
            for row in self.repo.team_rows(session, season_id):
                rows.append({**row, "_competition": comps[competition_id]["code"]})
        result = team_leader_rows(rows)
        competition_of = {r["team_id"]: r["_competition"] for r in rows}
        teams = self._teams(session, {e["team_id"] for board in result.values() for e in board["rows"]})
        for board in result.values():
            for e in board["rows"]:
                e.update({k: teams.get(e["team_id"], {}).get(k) for k in ("name", "short_code", "logo_url")},
                         competition=competition_of.get(e["team_id"]), scope=f"{competition_of.get(e['team_id'])}-{year}")
        return result

    def _referee_leaders(self, session, comps):
        """Most and fewest cards per match among ranked officials who worked in these competitions."""
        years = [y for (y,) in session.execute(select(ResearchPeerStat.season_year).where(
            ResearchPeerStat.subject_type == "referee").distinct().order_by(ResearchPeerStat.season_year.desc()))]
        for year in years:
            raw = select(Fixture.referee).join(Season, Season.id == Fixture.season_id).where(
                Fixture.competition_id.in_(list(comps)), Season.year == year, Fixture.status.in_(FINISHED),
                Fixture.referee.is_not(None))
            keys = select(ResearchRefereeAlias.referee_key).where(ResearchRefereeAlias.raw_name.in_(raw))
            rows = session.execute(select(ResearchPeerStat.subject_key, ResearchPeerStat.value, ResearchPeerStat.percentile,
                                          ResearchPeerStat.peer_count, ResearchPeerStat.sample_matches)
                                   .join(ResearchProfileBuild, ResearchProfileBuild.scope_key == ResearchPeerStat.scope_key)
                                   .where(ResearchPeerStat.subject_type == "referee", ResearchPeerStat.metric == "cards",
                                          ResearchPeerStat.season_year == year, ResearchPeerStat.subject_key.in_(keys),
                                          ResearchProfileBuild.build_id.startswith(RANK_BUILD_PREFIX))).all()
            if len(rows) < REFEREE_MIN_RANKED:
                continue
            names = dict(session.execute(select(ResearchRefereeAlias.referee_key, func.min(ResearchRefereeAlias.name))
                                         .where(ResearchRefereeAlias.referee_key.in_([r[0] for r in rows]))
                                         .group_by(ResearchRefereeAlias.referee_key)).all())
            entries = [{"referee_key": key, "name": names.get(key, key), "value": value, "percentile": pct,
                        "peer_count": peers, "matches": matches} for key, value, pct, peers, matches in rows]
            most = sorted(entries, key=lambda e: (-e["value"], -e["matches"], e["name"]))[:LEADER_LIMIT]
            fewest = sorted(entries, key=lambda e: (e["value"], -e["matches"], e["name"]))[:LEADER_LIMIT]
            return year, {
                "most_cards": {"label": "Most cards per match", "unit": "per_match", "higher_first": True, "rows": most},
                "fewest_cards": {"label": "Fewest cards per match", "unit": "per_match", "higher_first": False, "rows": fewest},
            }
        return None, {}

    def _build(self, session, kind, scope, now):
        comps = self._competitions(session, scope["codes"])
        latest = self._latest_finished(session, list(comps)) if comps else None
        weekend = self._weekend(session, kind, comps, now) if comps else []
        year, leaders = None, {}
        if comps and kind == "referees":
            year, leaders = self._referee_leaders(session, comps)
        elif comps:
            year = self._current_year(session, list(comps))
            if year is not None:
                leaders = (self._player_leaders if kind == "players" else self._team_leaders)(session, comps, year)
        return {"schema_version": "research-home.v1", "type": kind,
                "league": scope,
                "weekend": {"from": iso(now), "to": iso(now + timedelta(days=WEEKEND_DAYS)), "fixtures": weekend},
                "leaders": {"season": {"year": year, "label": season_label(year)} if year is not None else None,
                            "boards": leaders},
                "data_basis": {"source": "canonical_database", "latest_finished_kickoff": iso(latest),
                               "generated_at": iso(now),
                               "leaders_rule": ("ranked referees, 15+ matches, cards per match" if kind == "referees"
                                                else "league and cup matches in the comparable phase; counts until ranks apply"
                                                if kind == "players" else f"per match, {TEAM_MIN_MATCHES}+ matches with statistics"),
                               "referee_minimum_matches": REFEREE_MATCHES}}
