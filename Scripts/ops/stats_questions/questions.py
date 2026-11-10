"""Seven tested question types over the stats file.

The model only picks a type and fills its parameters. Everything here checks those parameters against fixed lists,
does the counting in SQL, and words the answer from fixed templates, so no number ever comes from the model.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from datetime import datetime
from statistics import mean

VERSION = "stats_answer.v1"
SMALL_SAMPLE = 5
FIRST_SEASON = 2019
TEAM_MATCHES, PLAYER_MINUTES, REFEREE_MATCHES = 3, 450, 5  # leader tables leave out smaller samples

# stat: (noun, for, against, in the match). Column names are these keys plus _for / _against.
TEAM_STATS = {
    "goals": ("goals", "goals scored", "goals conceded", "goals in the match"),
    "xg": ("expected goals", "expected goals", "expected goals against", "expected goals in the match"),
    "shots": ("shots", "shots", "shots faced", "shots in the match"),
    "shots_on_target": ("shots on target", "shots on target", "shots on target faced",
                        "shots on target in the match"),
    "corners": ("corners", "corners won", "corners conceded", "corners in the match"),
    "cards": ("cards", "cards (yellow and red)", "opponents' cards", "cards in the match"),
    "yellow_cards": ("yellow cards", "yellow cards", "opponents' yellow cards", "yellow cards in the match"),
    "red_cards": ("red cards", "red cards", "opponents' red cards", "red cards in the match"),
    "fouls": ("fouls", "fouls committed", "fouls suffered", "fouls in the match"),
    "offsides": ("offsides", "offsides", "opponents' offsides", "offsides in the match"),
    "saves": ("saves", "saves", "opponents' saves", "saves in the match"),
    "possession": ("possession", "possession", "opponents' possession", None),
}
PLAYER_STATS = {"goals": "goals", "assists": "assists", "shots": "shots", "shots_on_target": "shots on target",
                "fouls_committed": "fouls committed", "fouls_drawn": "fouls won", "cards": "cards (yellow and red)",
                "yellow_cards": "yellow cards", "red_cards": "red cards", "tackles": "tackles",
                "interceptions": "interceptions", "passes": "passes", "pass_accuracy": "pass accuracy",
                "rating": "rating"}
PLAYER_ONE = {"goals": "goal", "assists": "assist", "shots": "shot", "shots_on_target": "shot on target",
              "fouls_committed": "foul committed", "fouls_drawn": "foul won", "cards": "card (yellow and red)",
              "yellow_cards": "yellow card", "red_cards": "red card", "tackles": "tackle",
              "interceptions": "interception", "passes": "pass"}  # the noun for a total of exactly 1
AVERAGED = {"rating", "pass_accuracy"}
REFEREE_STATS = {"cards": "cards (yellow and red)", "yellow_cards": "yellow cards", "red_cards": "red cards",
                 "fouls": "fouls"}
TEAM_ALIASES = {"spurs": "tottenham", "tottenham hotspur": "tottenham", "man utd": "manchester united",
                "man united": "manchester united", "man u": "manchester united", "man city": "manchester city",
                "psg": "paris saint germain", "barca": "barcelona", "atleti": "atletico madrid",
                "bayern": "bayern munchen", "bayern munich": "bayern munchen", "gladbach": "borussia monchengladbach",
                "dortmund": "borussia dortmund", "bvb": "borussia dortmund", "leverkusen": "bayer leverkusen",
                "forest": "nottingham forest", "villa": "aston villa", "palace": "crystal palace",
                "inter milan": "inter", "milan": "ac milan", "roma": "as roma", "juve": "juventus",
                "wolverhampton": "wolves", "newcastle united": "newcastle", "west ham united": "west ham"}

_S = {"type": "string"}
_NS, _NI, _NN = {"type": ["string", "null"]}, {"type": ["integer", "null"]}, {"type": ["number", "null"]}
FIELDS = {
    "team": dict(_S, description="The club's full name, e.g. 'Tottenham' for Spurs, 'Manchester United' for Man Utd."),
    "opponent": dict(_S, description="The other club's full name."),
    "player": dict(_S, description="The player's name as written, e.g. 'Bukayo Saka'."),
    "referee": dict(_S, description="The referee's name, e.g. 'Anthony Taylor'."),
    "side": {"type": "string", "enum": ["for", "against", "match"],
             "description": "for: the team's own count; against: its opponents'; match: both teams added."},
    "venue": {"type": "string", "enum": ["home", "away", "all"]},
    "competition": dict(_NS, description="A competition code from the list, 'all', or null for the default."),
    "season": dict(_NI, description="Starting year, e.g. 2025 for 2025/26. Null means this season."),
    "last": dict(_NI, description="Only the last N matches, across seasons. Null to use the season."),
    "line": dict(_NN, description="The betting line, e.g. 9.5."),
    "direction": {"type": "string", "enum": ["over", "under"]},
    "at_least": dict(_NI, description="Count matches with at least this many, e.g. 2 for '2+ shots on target'."),
    "starts_only": {"type": "boolean"},
    "kind": {"type": "string", "enum": ["team", "player", "referee"]},
    "order": {"type": "string", "enum": ["most", "fewest"]},
    "top": dict(_NI, description="How many to list, at most 10."),
    "team_filter": dict(_NS, description="Only this referee's matches with this club, or null."),
}
DEFAULTS = {"side": "for", "venue": "all", "direction": "over", "starts_only": False, "order": "most"}


def _stat(names):
    return {"type": "string", "enum": sorted(names)}


TYPES = {
    "team_average": {
        "description": "A team's average for one stat per match, e.g. 'How many corners do Brentford get away?'",
        "params": {"team": FIELDS["team"], "stat": _stat(TEAM_STATS), "side": FIELDS["side"],
                   "venue": FIELDS["venue"], "competition": FIELDS["competition"], "season": FIELDS["season"],
                   "last": FIELDS["last"]}},
    "line_record": {
        "description": "How often a team's matches went over or under a line, e.g. 'How often do Brentford's away "
                       "games go over 9.5 corners?'",
        "params": {"team": FIELDS["team"], "stat": _stat(set(TEAM_STATS) - {"possession"}), "side": FIELDS["side"],
                   "line": FIELDS["line"], "direction": FIELDS["direction"], "venue": FIELDS["venue"],
                   "competition": FIELDS["competition"], "season": FIELDS["season"], "last": FIELDS["last"]}},
    "recent_form": {
        "description": "A team's recent results with goals, expected goals, corners and cards, e.g. 'How have Spurs "
                       "been doing lately?' Defaults to the last 6 matches in all competitions.",
        "params": {"team": FIELDS["team"], "venue": FIELDS["venue"], "competition": FIELDS["competition"],
                   "last": FIELDS["last"]}},
    "head_to_head": {
        "description": "Past meetings between two clubs since 2019/20, e.g. 'How do Arsenal v Chelsea games go?'",
        "params": {"team": FIELDS["team"], "opponent": FIELDS["opponent"], "last": FIELDS["last"]}},
    "referee": {
        "description": "A referee's cards or fouls per match, optionally how often over a line or only with one "
                       "club, e.g. 'How many cards does Anthony Taylor give?'",
        "params": {"referee": FIELDS["referee"], "stat": _stat(REFEREE_STATS), "competition": FIELDS["competition"],
                   "season": FIELDS["season"], "last": FIELDS["last"], "line": FIELDS["line"],
                   "team_filter": FIELDS["team_filter"]}},
    "player": {
        "description": "A player's numbers for one stat, e.g. 'How often does Saka have 2+ shots on target?'",
        "params": {"player": FIELDS["player"], "stat": _stat(PLAYER_STATS), "competition": FIELDS["competition"],
                   "season": FIELDS["season"], "last": FIELDS["last"], "at_least": FIELDS["at_least"],
                   "starts_only": FIELDS["starts_only"]}},
    "leaders": {
        "description": "The top teams, players or referees in a competition for one stat, e.g. 'Which La Liga team "
                       "wins the most corners?' Teams use team stats, players player stats, referees referee stats.",
        "params": {"kind": FIELDS["kind"], "stat": _stat(set(TEAM_STATS) | set(PLAYER_STATS)),
                   "side": FIELDS["side"], "venue": FIELDS["venue"], "competition": FIELDS["competition"],
                   "season": FIELDS["season"], "order": FIELDS["order"], "top": FIELDS["top"]}},
}
RANGES = {"season": (FIRST_SEASON, 2100), "last": (1, 50), "top": (1, 10), "at_least": (1, 20)}


class Invalid(Exception):
    pass


class NotFound(Exception):
    pass


class Which(Exception):
    def __init__(self, kind, field, query, rows):
        super().__init__(query)
        self.kind, self.field, self.query, self.rows = kind, field, query, rows


def blank(kind: str) -> dict:
    return {name: DEFAULTS.get(name) for name in TYPES.get(kind, {}).get("params", {})}


# -- formatting ---------------------------------------------------------------------------------------------------

def season_label(year: int) -> str:
    return f"{year}/{(year + 1) % 100:02d}"


def plural(n, word):
    return f"{n} {word}" if n == 1 else f"{n} {word}{'es' if word.endswith(('ch', 's', 'x')) else 's'}"


def _fmt(value, stat=None):
    if value is None:
        return "–"
    if stat == "possession":
        return f"{value:.0f}%"
    if stat == "rating":
        return f"{value:.2f}"
    return f"{value:.1f}"


def _num(value):
    return f"{value:g}" if isinstance(value, (int, float)) else "–"


def _val(value, stat=None):
    """One match's figure: counts as whole numbers, expected goals to one decimal, possession in percent."""
    if value is None or stat in ("possession", "rating"):
        return _fmt(value, stat)
    return f"{value:g}" if float(value).is_integer() else f"{value:.1f}"


def _day(iso):
    d = datetime.fromisoformat(iso)
    return f"{d.day} {d:%b %Y}"


def data_up_to(conn):
    row = conn.execute("SELECT value FROM build WHERE key = 'data_up_to'").fetchone()
    if not row or not row[0]:
        return None
    d = datetime.fromisoformat(row[0])
    return f"{d:%a} {d.day} {d:%b %Y}"


def _card(conn, kind, headline, *, scope=None, detail=(), table=None, matches=None, n=None, note=None, **extra):
    if note is None and n is not None and 0 < n < SMALL_SAMPLE:
        note = f"Small sample: only {plural(n, 'match')}."
    return {"version": VERSION, "type": kind, "scope": scope, "headline": headline, "detail": list(detail),
            "table": table, "matches": matches, "note": note, "data_up_to": data_up_to(conn), **extra}


# -- names --------------------------------------------------------------------------------------------------------

def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    return " ".join(t for t in re.sub(r"[^a-z0-9]+", " ", text).split() if t not in ("fc", "cf", "afc"))


def _tokens_match(query, name):
    words = name.split()
    return all(any(w.startswith(t) for w in words) for t in query.split())


FIND = {"team": "SELECT id, name, league, NULL AS club FROM teams",
        "player": "SELECT p.id, p.name, t.league, t.name AS club, p.team_id FROM players p "
                  "LEFT JOIN teams t ON t.id = p.team_id",
        "referee": "SELECT referee_key AS id, name, NULL AS league, NULL AS club FROM referees"}


def find(conn, kind, query, field=None):
    field = field or kind
    query = (query or "").strip()
    if not query:
        raise Invalid(f"Which {kind}? The question needs one.")
    if query.startswith("#"):
        key = query[1:]
        if kind != "referee":
            if not key.isdigit():
                raise Invalid(f"“{query}” is not a {kind} id.")
            key = int(key)
        row = conn.execute(f"SELECT * FROM ({FIND[kind]}) WHERE id = ?", (key,)).fetchone()
        if row is None:
            raise NotFound(f"No {kind} with id {key} in Spix's data.")
        return dict(row)
    rows = conn.execute(FIND[kind]).fetchall()
    target = _norm(query)
    if kind == "team":
        target = TEAM_ALIASES.get(target, target)
    pool = [r for r in rows if _norm(r["name"]) == target]
    if len(pool) == 1:
        return dict(pool[0])
    pool = pool or [r for r in rows if _tokens_match(target, _norm(r["name"]))]
    if not pool:
        by_name = {}
        for r in rows:
            by_name.setdefault(_norm(r["name"]), r)
        pool = [by_name[n] for n in difflib.get_close_matches(target, list(by_name), n=6, cutoff=0.8)]
    if not pool:
        raise NotFound(f"No {kind} called “{query}” in Spix's data.")
    whole = [r for r in pool if all(len(t) == 1 or t in _norm(r["name"]).split() for t in target.split())]
    pool = whole or pool  # "Saka" means a Saka before a Sakamoto
    if len(pool) == 1:
        return dict(pool[0])
    pool = _busiest_first(conn, kind, pool)
    if pool[0]["weight"] and pool[0]["weight"] >= 2 * pool[1]["weight"]:
        return pool[0]  # a much busier namesake: Erling Haaland, not M. Haaland of Brann
    raise Which(kind, field, query, pool[:6])


WEIGHT = {"team": "SELECT team_id, count(*) FROM team_matches WHERE season >= ? AND team_id IN ({}) GROUP BY 1",
          "player": "SELECT player_id, sum(minutes) FROM player_matches WHERE season >= ? AND player_id IN ({}) "
                    "GROUP BY 1",
          "referee": "SELECT referee_key, count(*) FROM matches WHERE season >= ? AND referee_key IN ({}) GROUP BY 1"}


def _busiest_first(conn, kind, rows):
    """Matches (or minutes) over this season and the last one, most first."""
    rows = [dict(r) for r in rows[:200]]
    since = (_current(conn, "all") or FIRST_SEASON) - 1
    weights = dict(conn.execute(WEIGHT[kind].format(", ".join("?" * len(rows))),
                                [since, *(r["id"] for r in rows)]).fetchall())
    for r in rows:
        r["weight"] = weights.get(r["id"], 0)
    return sorted(rows, key=lambda r: -r["weight"])


# -- shared pieces ------------------------------------------------------------------------------------------------

def _competition(conn, code, default):
    code = code if code not in (None, "") else default
    if code in (None, "all"):
        return "all", "All competitions"
    row = conn.execute("SELECT code, name FROM competitions WHERE lower(code) = lower(?) OR lower(name) = lower(?)",
                       (code, code)).fetchone()
    if row is None:
        raise Invalid(f"Spix has no competition called “{code}”.")
    return row["code"], row["name"]


def _current(conn, competition, where="", args=()):
    sql = "SELECT max(season) FROM matches m WHERE 1 = 1" + where
    if competition != "all":
        sql, args = sql + " AND m.competition = ?", (*args, competition)
    return conn.execute(sql, args).fetchone()[0]


def _scope(subject, comp, comp_name, season, n, last, noun="match"):
    parts = [subject]
    if last:
        parts += [comp_name, f"last {plural(n, noun)}"]
    elif comp == "all":
        parts += [comp_name, season_label(season), plural(n, noun)]
    else:
        parts += [f"{comp_name} {season_label(season)}", plural(n, noun)]
    return " · ".join(parts)


def _filters(alias, p, comp, season, *, venue=True):
    where, args = [], []
    if comp != "all":
        where.append(f"{alias}.competition = ?"); args.append(comp)
    if venue and p.get("venue", "all") != "all":
        where.append(f"{alias}.is_home = ?"); args.append(int(p["venue"] == "home"))
    if not p.get("last"):
        where.append(f"{alias}.season = ?"); args.append(season)
    return where, args


def _team_expr(stat, side):
    if side == "match":
        return f"(tm.{stat}_for + tm.{stat}_against)"
    return f"tm.{stat}_{side}"


VENUE_PHRASE = {"home": " at home", "away": " away from home", "all": ""}
VENUE_SUBJECT = {"home": " at home", "away": " away", "all": ""}
VENUE_SIDES = {"home": " for home sides", "away": " for away sides", "all": ""}


def _team_rows(conn, team, p, expr, comp, season):
    where, args = _filters("tm", p, comp, season)
    sql = (f"SELECT tm.*, o.name AS opponent, {expr} AS value FROM team_matches tm JOIN teams o ON o.id = "
           f"tm.opponent_id WHERE tm.team_id = ? AND {expr} IS NOT NULL"
           + "".join(" AND " + w for w in where) + " ORDER BY tm.kickoff DESC")
    args = [team["id"], *args]
    if p.get("last"):
        sql += " LIMIT ?"; args.append(p["last"])
    return conn.execute(sql, args).fetchall()


def _team_setup(conn, p, stat):
    team = find(conn, "team", p["team"], "team")
    comp, comp_name = _competition(conn, p.get("competition"), team["league"])
    season = p.get("season") or _current(conn, comp)
    label = TEAM_STATS[stat][{"for": 1, "against": 2, "match": 3}[p["side"]]]
    return team, comp, comp_name, season, label


def _match_rows(rows, value_stat, title):
    return {"columns": ["Date", "Opponent", "Venue", "Score", title[:1].upper() + title[1:]],
            "rows": [[_day(r["kickoff"]), r["opponent"], "H" if r["is_home"] else "A",
                      f"{_num(r['goals_for'])}–{_num(r['goals_against'])}", _val(r["value"], value_stat)]
                     for r in rows]}


# -- the seven question types -------------------------------------------------------------------------------------

def team_average(conn, p):
    stat = p["stat"]
    team, comp, comp_name, season, label = _team_setup(conn, p, stat)
    rows = _team_rows(conn, team, p, _team_expr(stat, p["side"]), comp, season)
    scope = _scope(team["name"] + VENUE_SUBJECT[p["venue"]], comp, comp_name, season, len(rows), p.get("last"))
    if not rows:
        return _card(conn, "team_average", f"No {TEAM_STATS[stat][0]} recorded for these matches.", scope=scope, n=0)
    avg = mean(r["value"] for r in rows)
    if stat == "possession":
        headline = f"{team['name']} had {_fmt(avg, stat)} of the ball on average{VENUE_PHRASE[p['venue']]}."
    else:
        headline = f"{team['name']} averaged {_fmt(avg)} {label} a match{VENUE_PHRASE[p['venue']]}."
    detail, table = [], None
    if comp != "all":
        seasons = sorted({r["season"] for r in rows})
        expr = _team_expr(stat, p["side"])
        where, args = _filters("tm", {**p, "last": None}, comp, seasons[0])
        where[-1] = f"tm.season IN ({', '.join('?' * len(seasons))})"
        args[-1:] = seasons
        league = conn.execute(f"SELECT avg({expr}) FROM team_matches tm WHERE {expr} IS NOT NULL AND "
                              + " AND ".join(where), args).fetchone()[0]
        detail.append(f"{comp_name} average{VENUE_SIDES[p['venue']]}: {_fmt(league, stat)}.")
    if stat != "possession":
        both = [r for r in rows if r[f"{stat}_for"] is not None and r[f"{stat}_against"] is not None]
        if both:
            f, a = mean(r[f"{stat}_for"] for r in both), mean(r[f"{stat}_against"] for r in both)
            table = {"columns": ["", "For", "Against", "In the match"],
                     "rows": [["Average a match", _fmt(f), _fmt(a), _fmt(f + a)]]}
    return _card(conn, "team_average", headline, scope=scope, detail=detail, table=table,
                 matches=_match_rows(rows, stat, label), n=len(rows))


def line_record(conn, p):
    stat, line = p["stat"], p.get("line")
    if stat == "possession":
        raise Invalid("Possession has no over/under lines here.")
    if line is None:
        raise Invalid("Which line? For example over 9.5.")
    team, comp, comp_name, season, _ = _team_setup(conn, p, stat)
    expr = _team_expr(stat, p["side"])
    rows = _team_rows(conn, team, p, expr, comp, season)
    scope = _scope(team["name"] + VENUE_SUBJECT[p["venue"]], comp, comp_name, season, len(rows), p.get("last"))
    noun, over = TEAM_STATS[stat][0], p["direction"] == "over"
    if not rows:
        return _card(conn, "line_record", f"No {noun} recorded for these matches.", scope=scope, n=0)
    hit = (lambda v: v > line) if over else (lambda v: v < line)
    hits, word = sum(hit(r["value"]) for r in rows), "over" if over else "under"
    lead = {"match": word.capitalize(), "for": f"{team['name']} had {word}",
            "against": f"Their opponents had {word}"}[p["side"]]
    headline = f"{lead} {line:g} {noun} in {hits} of {plural(len(rows), 'match')}."
    detail = [f"Average {_fmt(mean(r['value'] for r in rows))}."]
    if comp != "all" and not p.get("last"):
        # A match total compares with every league match once; a side's count with the same venue's sides.
        venue = "all" if p["side"] == "match" else p["venue"]
        where, args = _filters("tm", {**p, "venue": venue}, comp, season)
        if venue == "all" and p["side"] == "match":
            where.append("tm.is_home = 1")
        sign = ">" if over else "<"
        lh, ln = conn.execute(f"SELECT sum({expr} {sign} ?), count(*) FROM team_matches tm WHERE {expr} IS NOT NULL "
                              "AND " + " AND ".join(where), [line, *args]).fetchone()
        detail.append(f"{comp_name}{VENUE_SIDES[venue]}: {word} {line:g} in {lh or 0} of {ln} matches.")
    matches = _match_rows(rows, stat, TEAM_STATS[stat][{"for": 1, "against": 2, "match": 3}[p["side"]]])
    matches["columns"].append(f"{word.capitalize()} {line:g}")
    for row, r in zip(matches["rows"], rows):
        row.append("yes" if hit(r["value"]) else "")
    return _card(conn, "line_record", headline, scope=scope, detail=detail, matches=matches, n=len(rows))


def _result(r):
    if r["goals_for"] is None or r["goals_against"] is None:
        return "–"
    return "W" if r["goals_for"] > r["goals_against"] else "D" if r["goals_for"] == r["goals_against"] else "L"


def _pair(r, stat):
    a, b = r[f"{stat}_for"], r[f"{stat}_against"]
    return "–" if a is None or b is None else f"{_num(a) if stat != 'xg' else f'{a:.1f}'}–" \
                                              f"{_num(b) if stat != 'xg' else f'{b:.1f}'}"


def recent_form(conn, p):
    team = find(conn, "team", p["team"], "team")
    comp, comp_name = _competition(conn, p.get("competition"), "all")
    last = p.get("last") or 6
    where, args = _filters("tm", {**p, "last": last}, comp, None)
    rows = conn.execute("SELECT tm.*, o.name AS opponent FROM team_matches tm JOIN teams o ON o.id = tm.opponent_id "
                        "WHERE tm.team_id = ?" + "".join(" AND " + w for w in where)
                        + " ORDER BY tm.kickoff DESC LIMIT ?", [team["id"], *args, last]).fetchall()
    scope = _scope(team["name"] + VENUE_SUBJECT[p["venue"]], comp, comp_name, None, len(rows), True)
    if not rows:
        return _card(conn, "recent_form", "No matches found.", scope=scope, n=0)
    results = [_result(r) for r in rows]
    w, d, l = (results.count(x) for x in "WDL")
    headline = (f"{team['name']}'s last {len(rows)}: {plural(w, 'win')}, {plural(d, 'draw')}, "
                f"{plural(l, 'defeat')}.")
    scored, conceded = (sum(r[k] or 0 for r in rows) for k in ("goals_for", "goals_against"))
    detail = [f"Scored {scored}, conceded {conceded}."]
    if all(r["xg_for"] is not None and r["xg_against"] is not None for r in rows):
        detail.append(f"Expected goals: {sum(r['xg_for'] for r in rows):.1f} for, "
                      f"{sum(r['xg_against'] for r in rows):.1f} against.")
    matches = {"columns": ["Date", "Opponent", "Venue", "Score", "Result", "xG", "Corners", "Cards"],
               "rows": [[_day(r["kickoff"]), r["opponent"], "H" if r["is_home"] else "A",
                         f"{_num(r['goals_for'])}–{_num(r['goals_against'])}", res, _pair(r, "xg"),
                         _pair(r, "corners"), _pair(r, "cards")] for r, res in zip(rows, results)]}
    return _card(conn, "recent_form", headline, scope=scope, detail=detail, matches=matches, n=len(rows))


def head_to_head(conn, p):
    team = find(conn, "team", p["team"], "team")
    other = find(conn, "team", p["opponent"], "opponent")
    last = p.get("last") or 6
    rows = conn.execute("SELECT tm.*, c.name AS competition_name FROM team_matches tm JOIN competitions c ON "
                        "c.code = tm.competition WHERE tm.team_id = ? AND tm.opponent_id = ? ORDER BY tm.kickoff "
                        "DESC LIMIT ?", (team["id"], other["id"], last)).fetchall()
    n = len(rows)
    scope = f"{team['name']} v {other['name']} · All competitions since {season_label(FIRST_SEASON)} · " \
            f"{plural(n, 'meeting')}"
    if not n:
        return _card(conn, "head_to_head", f"No meetings between {team['name']} and {other['name']} since "
                     f"{season_label(FIRST_SEASON)} in Spix's data.", scope=scope, n=0)
    results = [_result(r) for r in rows]
    w, d, l = (results.count(x) for x in "WDL")
    headline = (f"{team['name']} v {other['name']}, last {plural(n, 'meeting')}: {team['name']} {plural(w, 'win')}, "
                f"{other['name']} {plural(l, 'win')}, {plural(d, 'draw')}.")
    detail = []
    for stat, noun in (("goals", "goals"), ("corners", "corners"), ("cards", "cards (yellow and red)")):
        known = [r[f"{stat}_for"] + r[f"{stat}_against"] for r in rows
                 if r[f"{stat}_for"] is not None and r[f"{stat}_against"] is not None]
        if known:
            detail.append(f"{_fmt(mean(known))} {noun} a match.")
    matches = {"columns": ["Date", "Competition", f"{team['name']} at", "Score", "Corners", "Cards"],
               "rows": [[_day(r["kickoff"]), r["competition_name"], "home" if r["is_home"] else "away",
                         f"{_num(r['goals_for'])}–{_num(r['goals_against'])}", _pair(r, "corners"),
                         _pair(r, "cards")] for r in rows]}
    return _card(conn, "head_to_head", headline, scope=scope, detail=detail, matches=matches, n=n)


def referee(conn, p):
    stat = p["stat"]
    ref = find(conn, "referee", p["referee"], "referee")
    team = find(conn, "team", p["team_filter"], "team_filter") if p.get("team_filter") else None
    comp, comp_name = _competition(conn, p.get("competition"), "all")
    season = p.get("season") or _current(conn, comp, " AND m.referee_key = ?", (ref["id"],))
    expr = f"(tm.{stat}_for + tm.{stat}_against)"
    where, args = _filters("m", p, comp, season, venue=False)
    if team:
        where.append("(m.home_id = ? OR m.away_id = ?)"); args += [team["id"], team["id"]]
    sql = (f"SELECT m.*, h.name AS home, a.name AS away, {expr} AS value FROM matches m JOIN team_matches tm ON "
           "tm.fixture_id = m.fixture_id AND tm.is_home = 1 JOIN teams h ON h.id = m.home_id JOIN teams a ON "
           f"a.id = m.away_id WHERE m.referee_key = ? AND {expr} IS NOT NULL"
           + "".join(" AND " + w for w in where) + " ORDER BY m.kickoff DESC")
    args = [ref["id"], *args]
    if p.get("last"):
        sql += " LIMIT ?"; args.append(p["last"])
    rows = conn.execute(sql, args).fetchall()
    subject = ref["name"] + (f" with {team['name']}" if team else "")
    scope = _scope(subject, comp, comp_name, season, len(rows), p.get("last"))
    label = REFEREE_STATS[stat]
    if not rows:
        return _card(conn, "referee", f"No matches found for {ref['name']}.", scope=scope, n=0)
    avg = mean(r["value"] for r in rows)
    headline, detail = f"{ref['name']}: {_fmt(avg)} {label} a match.", []
    current = _current(conn, comp)
    if not p.get("season") and not p.get("last") and season < current:
        detail.append(f"{ref['name']} has no matches in {season_label(current)} in Spix's data, so this is "
                      f"{season_label(season)}.")
    line = p.get("line")
    if line is not None:
        detail.append(f"Over {line:g} in {sum(r['value'] > line for r in rows)} of {plural(len(rows), 'match')}.")
    comps, seasons = sorted({r["competition"] for r in rows}), sorted({r["season"] for r in rows})
    peer = conn.execute(f"SELECT avg({expr}) FROM matches m JOIN team_matches tm ON tm.fixture_id = m.fixture_id "
                        f"AND tm.is_home = 1 WHERE {expr} IS NOT NULL AND m.competition IN "
                        f"({', '.join('?' * len(comps))}) AND m.season IN ({', '.join('?' * len(seasons))})",
                        [*comps, *seasons]).fetchone()[0]
    detail.append(f"All referees in the same competitions and seasons: {_fmt(peer)}.")
    matches = {"columns": ["Date", "Match", "Score", label.capitalize()],
               "rows": [[_day(r["kickoff"]), f"{r['home']} v {r['away']}",
                         f"{_num(r['home_goals'])}–{_num(r['away_goals'])}", _num(r["value"])] for r in rows]}
    return _card(conn, "referee", headline, scope=scope, detail=detail, matches=matches, n=len(rows))


def player(conn, p):
    stat = p["stat"]
    who = find(conn, "player", p["player"], "player")
    comp, comp_name = _competition(conn, p.get("competition"), who.get("league"))
    season = p.get("season") or _current(conn, comp)
    where, args = _filters("pm", p, comp, season, venue=False)
    if p.get("starts_only"):
        where.append("pm.started = 1")
    sql = ("SELECT pm.*, o.name AS opponent FROM player_matches pm JOIN matches m ON m.fixture_id = pm.fixture_id "
           "JOIN teams o ON o.id = CASE WHEN m.home_id = pm.team_id THEN m.away_id ELSE m.home_id END "
           f"WHERE pm.player_id = ? AND pm.{stat} IS NOT NULL" + "".join(" AND " + w for w in where)
           + " ORDER BY pm.kickoff DESC")
    args = [who["id"], *args]
    if p.get("last"):
        sql += " LIMIT ?"; args.append(p["last"])
    rows = conn.execute(sql, args).fetchall()
    subject = who["name"] + (f" ({who['club']})" if who.get("club") else "") + (
        ", starts only" if p.get("starts_only") else "")
    scope = _scope(subject, comp, comp_name, season, len(rows), p.get("last"))
    noun = PLAYER_STATS[stat]
    if not rows:
        return _card(conn, "player", f"No {noun} recorded for {who['name']} in these matches.", scope=scope, n=0)
    values, minutes = [r[stat] for r in rows], sum(r["minutes"] for r in rows)
    starts = sum(r["started"] for r in rows)
    detail = [f"{plural(starts, 'start')}, {minutes} minutes."]
    if stat in AVERAGED:
        shown = f"{mean(values):.2f}" if stat == "rating" else f"{mean(values):.0f}%"
        headline = f"{who['name']}: average {noun} {shown} in {plural(len(rows), 'match')}."
    elif p.get("at_least"):
        k = p["at_least"]
        headline = f"{who['name']} had {k}+ {noun} in {sum(v >= k for v in values)} of {plural(len(rows), 'match')}."
        detail.append(f"{_fmt(mean(values))} a match on average.")
    else:
        total = sum(values)
        per90 = total * 90 / minutes if minutes else 0
        counted = PLAYER_ONE.get(stat, noun) if total == 1 else noun
        headline = (f"{who['name']}: {_num(total)} {counted} in {plural(len(rows), 'match')}, "
                    f"{_fmt(total / len(rows))} a match ({_fmt(per90)} per 90 minutes).")
    matches = {"columns": ["Date", "Opponent", "Minutes", "Started", noun.capitalize()],
               "rows": [[_day(r["kickoff"]), r["opponent"], r["minutes"], "yes" if r["started"] else "",
                         _val(r[stat], stat)] for r in rows]}
    return _card(conn, "player", headline, scope=scope, detail=detail, matches=matches, n=len(rows))


def leaders(conn, p):
    kind, stat = p["kind"], p["stat"]
    allowed = {"team": TEAM_STATS, "player": PLAYER_STATS, "referee": REFEREE_STATS}[kind]
    if stat not in allowed:
        raise Invalid(f"“{stat}” is not a {kind} stat.")
    if not p.get("competition"):
        raise Invalid("Leaders need a competition, for example the Premier League.")
    comp, comp_name = _competition(conn, p["competition"], None)
    season = p.get("season") or _current(conn, comp)
    top, desc = p.get("top") or 5, "DESC" if p["order"] == "most" else "ASC"
    scope_where = ("" if comp == "all" else " AND {a}.competition = ?") + " AND {a}.season = ?"
    scope_args = ([] if comp == "all" else [comp]) + [season]
    if kind == "team":
        expr = _team_expr(stat, p["side"])
        venue = "" if p["venue"] == "all" else f" AND tm.is_home = {int(p['venue'] == 'home')}"
        rows = conn.execute(f"SELECT t.name, avg({expr}) AS v, count(*) AS n FROM team_matches tm JOIN teams t ON "
                            f"t.id = tm.team_id WHERE {expr} IS NOT NULL{scope_where.format(a='tm')}{venue} GROUP BY "
                            f"tm.team_id HAVING n >= {TEAM_MATCHES} ORDER BY v {desc} LIMIT ?",
                            [*scope_args, top]).fetchall()
        label = TEAM_STATS[stat][{"for": 1, "against": 2, "match": 3}[p["side"]]] or stat
        unit, count_name, note = " a match", "Matches", f"Teams with at least {TEAM_MATCHES} matches."
    elif kind == "player":
        value = f"avg(pm.{stat})" if stat in AVERAGED else f"sum(pm.{stat}) * 90.0 / sum(pm.minutes)"
        rows = conn.execute(f"SELECT p.name || coalesce(' (' || t.name || ')', '') AS name, {value} AS v, "
                            f"sum(pm.minutes) AS n FROM player_matches pm JOIN players p ON p.id = pm.player_id "
                            f"LEFT JOIN teams t ON t.id = p.team_id WHERE pm.{stat} IS NOT NULL"
                            f"{scope_where.format(a='pm')} GROUP BY pm.player_id HAVING n >= {PLAYER_MINUTES} "
                            f"ORDER BY v {desc} LIMIT ?", [*scope_args, top]).fetchall()
        label = PLAYER_STATS[stat]
        unit = "" if stat in AVERAGED else " per 90 minutes"
        count_name, note = "Minutes", f"Players with at least {PLAYER_MINUTES} minutes."
    else:
        expr = f"(tm.{stat}_for + tm.{stat}_against)"
        rows = conn.execute(f"SELECT r.name, avg({expr}) AS v, count(*) AS n FROM matches m JOIN team_matches tm ON "
                            f"tm.fixture_id = m.fixture_id AND tm.is_home = 1 JOIN referees r ON r.referee_key = "
                            f"m.referee_key WHERE {expr} IS NOT NULL{scope_where.format(a='m')} GROUP BY "
                            f"m.referee_key HAVING n >= {REFEREE_MATCHES} ORDER BY v {desc} LIMIT ?",
                            [*scope_args, top]).fetchall()
        label, unit = REFEREE_STATS[stat], " a match"
        count_name, note = "Matches", f"Referees with at least {REFEREE_MATCHES} matches."
    shown = [(r["name"], _fmt(r["v"], stat if stat in ("possession", "rating") else None), r["n"]) for r in rows]
    scope = f"{comp_name} {season_label(season)} · {kind}s"
    if not shown:
        return _card(conn, "leaders", "Nobody has enough matches yet.", scope=scope, note=note)
    headline = (f"{'Most' if desc == 'DESC' else 'Fewest'} {label}{unit}, {comp_name} {season_label(season)}: "
                + ", ".join(f"{name} {v}" for name, v, _ in shown[:3]) + ".")
    table = {"columns": ["#", "Name", (label + unit).capitalize(), count_name],
             "rows": [[i + 1, name, v, n] for i, (name, v, n) in enumerate(shown)]}
    return _card(conn, "leaders", headline, scope=scope, table=table, note=note)


RUN = {"team_average": team_average, "line_record": line_record, "recent_form": recent_form,
       "head_to_head": head_to_head, "referee": referee, "player": player, "leaders": leaders}


def _check(kind, params):
    if kind not in TYPES:
        raise Invalid(f"Unknown question type “{kind}”.")
    p = {**blank(kind), **{k: v for k, v in (params or {}).items() if k in TYPES[kind]["params"]}}
    for name, value in p.items():
        spec = TYPES[kind]["params"][name]
        if value is None:
            if "null" not in spec["type"]:  # a name or a stat with no default: nothing to count without it
                raise Invalid(f"The {name.replace('_', ' ')} is missing.")
            continue
        if "enum" in spec and value not in spec["enum"]:
            raise Invalid(f"“{value}” is not one of the choices for {name}.")
        if name in RANGES:
            low, high = RANGES[name]
            if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
                raise Invalid(f"{name} must be a whole number from {low} to {high}.")
        if name == "line" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise Invalid("The line must be a number, for example 9.5.")
    if p.get("stat") == "possession" and p.get("side") == "match":
        raise Invalid("Possession is counted for one side, not the whole match.")
    return p


def answer(conn, kind: str, params: dict) -> dict:
    """One question type with its parameters -> one ``stats_answer.v1`` card. Never raises for bad input."""
    try:
        p = _check(kind, params)
        return RUN[kind](conn, p)
    except Invalid as exc:
        return _card(conn, "invalid", str(exc))
    except NotFound as exc:
        return _card(conn, "not_found", str(exc))
    except Which as exc:
        options = [{"label": r["name"] + (f" ({r['club']})" if r.get("club") else ""), "value": f"#{r['id']}",
                    "field": exc.field} for r in exc.rows]
        return _card(conn, "which", f"Which {exc.kind} do you mean by “{exc.query}”?", options=options)


# -- questions the types cannot answer ----------------------------------------------------------------------------

CAN_ANSWER = ("a team's averages, how often its matches went over a line, recent form, head to head, a referee's "
              "cards, a player's numbers and league leaders, from 2019/20 on")
DECLINE = {
    "asks_for_a_bet": "Spix's picks are on the matches page, each with the chance behind it. Here you can ask what "
                      "has happened, for example how often Brentford's away games went over 9.5 corners.",
    "not_football_stats": f"This tab answers questions about past football numbers: {CAN_ANSWER}.",
    "not_covered": f"Spix can't count that yet. It can count {CAN_ANSWER}.",
}


def declined(conn, reason: str) -> dict:
    reason = reason if reason in DECLINE else "not_covered"
    return _card(conn, "declined", DECLINE[reason], reason=reason)
