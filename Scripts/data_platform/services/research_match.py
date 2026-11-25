"""One match's research view for the Research Area (tracker D110, the overhaul's B2, D107).

The match, Spix's read of it and the past numbers that decide each market. Spix's figures are the read's own: the
figures the match page shows (``match_read_metrics.match_metrics``), and for each over/under market every half line
at least ``fair_prices.MIN_BOOKS`` bookmakers price on both sides, with the market's fair chance and the typical
price, as the recommendation layer lists its options. Cards lines come from the read's cards result and never carry
a suggestion: cards stay out of the picks (D61). Nothing is predicted afresh. The read keeps only whether line-ups
were confirmed, not the players, so ``lineups`` is null.

Each team's past numbers follow the research profiles' count rules (``normalize_research_row``): a match whose
statistics row is missing on either side has unknown statistics, never zeros. "This season" is the match's season
year in every competition; "the league" is the team's domestic league that season.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .. import fair_prices as fp
from ..reference_price import audit_books
from .match_read_metrics import match_metrics
from ..research_stats import normalize_research_row
from .research_profiles import (add_known, iso, most_common_position, raw_referee_key, referee_summary,
                                referee_values, starter, team_summary)
from .research_upcoming import Read, _figure, _time, kickoff_utc, match_entry, one_decimal, refs_for

SCHEMA_VERSION = "research-match.v1"
TOTALS_KEYS = {"cards": "totals_cards_over_under", "goals": "totals", "corners": "totals_corners_over_under"}
LAST_MATCHES, BOOKED, SHOOTERS, SHOOTER_FORM, STARTS_WINDOW = 10, 5, 8, 10, 5
REFEREE_MATCHES, H2H_MATCHES, H2H_SINCE = 10, 6, 2023


def _results(canonical_results) -> List[Mapping[str, Any]]:
    results = canonical_results if isinstance(canonical_results, Sequence) and not isinstance(
        canonical_results, (str, bytes)) else []
    return [r for r in results if isinstance(r, Mapping)]


def total_lines(canonical_results, market: str) -> List[dict]:
    """Every half line of an over/under market with both sides' fair chance and typical price, lines ascending.
    Each market has the goals market's over/under shape, priced on its own bookmakers' prices."""
    result = fp._results(_results(canonical_results)).get(market)
    if result is None:
        return []
    options = ((result.get("context") or {}).get("decision_audit") or {}).get("input_options")
    books, lines = audit_books(options, TOTALS_KEYS[market]), []
    for line in fp._lines(result, "goals", books):
        if line % 1 != 0.5:
            continue
        entry = fp._entry(result, "goals", books, line)
        if entry is None:
            continue
        sides = {o["side"]: {"chance": o["fair_probability"],
                             "typical_price": fp.typical_price(books, "goals", o["side"], o["line"])}
                 for o in entry["outcomes"]}
        if all(side["typical_price"] is not None for side in sides.values()):
            lines.append({"line": line, **sides})
    return lines


def read_block(card: Mapping[str, Any], canonical_results, *, status: str) -> dict:
    """The ``read`` of the match view from the card the match page shows and the read's canonical results."""
    results = _results(canonical_results)
    metrics = match_metrics(results)
    outcome = metrics.get("result_probabilities") or {}
    result = {k: outcome[k] for k in ("home", "draw", "away")} if all(k in outcome for k in ("home", "draw", "away")) \
        else None
    team_goals = metrics.get("team_goals") or {}
    home, away = one_decimal(team_goals.get("home")), one_decimal(team_goals.get("away"))
    goals = {"home": home, "away": away, "total": _figure(metrics, "goals")} if home is not None and away is not None \
        else None
    block = card.get("recommendations")
    markets = block.get("markets") if isinstance(block, Mapping) else None
    suggestions = {m.get("market"): m.get("suggestion") for m in markets or [] if isinstance(m, Mapping)}
    return {"status": status, "read_at": _time(card.get("created_at")),
            "prices_as_of": (card.get("provenance") or {}).get("evaluated_at"),
            "result": result, "goals": goals, "btts_yes": team_goals.get("both_teams_score_probability"),
            "corners_expected": _figure(metrics, "corners"), "cards_expected": _figure(metrics, "cards"),
            "lines": {market: total_lines(results, market) for market in TOTALS_KEYS},
            "suggestions": {"cards": None, "goals": suggestions.get("goals"), "corners": suggestions.get("corners")},
            "markets": markets}


def last_match(row: Mapping[str, Any]) -> dict:
    """One of a team's last matches, from its side: ``ResearchRepository.team_recent``'s row."""
    row = normalize_research_row("team", row)
    return {"event_id": str(row["api_football_id"]), "date": iso(row["kickoff_utc"]),
            "competition": {"code": row["competition_code"], "name": row["competition_name"]},
            "venue": "home" if row["team_id"] == row["home_team_id"] else "away",
            "opponent": {"id": row["opponent_team_id"], "name": row["opponent_name"],
                         "short_code": row["opponent_short_code"], "logo_url": row["opponent_logo_url"]},
            "score": {"home": row["home_goals"], "away": row["away_goals"]},
            "cards_for": add_known(row.get("yellow_cards"), row.get("red_cards")),
            "cards_against": add_known(row.get("opponent_yellow_cards"), row.get("opponent_red_cards")),
            "corners_for": row.get("corners"), "corners_against": row.get("opponent_corners"),
            "goals_for": row["result_goals_for"], "goals_against": row["result_goals_against"],
            "sot_for": row.get("shots_on"), "sot_against": row.get("opponent_shots_on")}


def likely_starters(squad_rows: Sequence[Mapping[str, Any]], limit: int = SHOOTERS) -> List[dict]:
    """Outfield players by starts in the team's last matches, then minutes, then name; at least one start."""
    by_player = defaultdict(list)
    for row in squad_rows:
        if (row.get("minutes") or 0) > 0:
            by_player[row["player_id"]].append(row)
    ranked = []
    for player_id, rows in by_player.items():
        position = most_common_position(rows)
        starts = sum(starter(row)[0] for row in rows)
        if position == "G" or not starts:
            continue
        ranked.append(((-starts, -sum(row["minutes"] for row in rows), rows[0]["name"].casefold(), player_id),
                       {"player_id": player_id, "name": rows[0]["name"],
                        "position": None if position == "unknown" else position, "starts_last_5": starts}))
    return [player for _, player in sorted(ranked, key=lambda item: item[0])[:limit]]


def likely_eleven(squad_rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    """The keeper with most starts in the team's last matches, then ten outfield players as ``likely_starters``
    ranks them (D135): most starts, never a prediction."""
    keepers = defaultdict(list)
    for row in squad_rows:
        if (row.get("minutes") or 0) > 0:
            keepers[row["player_id"]].append(row)
    ranked = []
    for player_id, rows in keepers.items():
        starts = sum(starter(row)[0] for row in rows)
        if most_common_position(rows) == "G" and starts:
            ranked.append(((-starts, -sum(row["minutes"] for row in rows), rows[0]["name"].casefold(), player_id),
                           {"player_id": player_id, "name": rows[0]["name"], "position": "G", "starts_last_5": starts}))
    keeper = [player for _, player in sorted(ranked, key=lambda item: item[0])[:1]]
    return keeper + likely_starters(squad_rows, limit=10)


def appearance(row: Mapping[str, Any]) -> dict:
    played = normalize_research_row("player", row)
    return {"date": iso(row["kickoff_utc"]),
            "opponent": row["away_name"] if row["team_id"] == row["home_team_id"] else row["home_name"],
            "minutes": row["minutes"], "sot": played["shots_on"], "started": starter(row)[0]}


def league_rank(rows: Sequence[Mapping[str, Any]], team_id: int) -> Optional[dict]:
    """Shots on target conceded a match (the profiles' ``shots_on_target_faced``) and the team's rank among the
    league's teams, 1 = the fewest."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["team_id"]].append(row)
    values = {}
    for key, team_rows in groups.items():
        metric = team_summary(team_rows)["metrics"]["shots_on_target_faced"]
        if metric["value"] is not None:
            values[key] = (metric["value"], metric["coverage"]["observed_matches"])
    if team_id not in values:
        return None
    value, matches = values[team_id]
    ordered = sorted(v for v, _ in values.values())
    return {"per_match": one_decimal(value), "matches": matches, "rank": bisect_left(ordered, value) + 1,
            "peer_count": len(values)}


def _played(row: Mapping[str, Any]) -> Dict[str, Any]:
    """A finished match's totals from both sides' statistics, under the profiles' count rules."""
    row = normalize_research_row("referee", row)
    corners = add_known(row.get("home_stats_corners"), row.get("away_stats_corners"))
    return {"cards": referee_values(row)["cards"], "corners": corners,
            "goals": add_known(row.get("home_goals"), row.get("away_goals"))}


def meeting(row: Mapping[str, Any]) -> dict:
    return {"event_id": str(row["api_football_id"]), "date": iso(row["kickoff_utc"]),
            "competition": {"code": row["competition"], "name": row["competition_name"]},
            "home": {"id": row["home_team_id"], "name": row["home_name"]},
            "away": {"id": row["away_team_id"], "name": row["away_name"]},
            "score": {"home": row["home_goals"], "away": row["away_goals"]}, **_played(row)}


def cards_per_match(rows: Sequence[Mapping[str, Any]]) -> tuple:
    """(matches with both sides' statistics, their cards a match to one decimal)."""
    summary = referee_summary(rows)
    return summary["matches"], one_decimal(summary["metrics"]["cards"]["value"])


class ResearchMatchService:
    def __init__(self, repo, *, load_read: Callable[[str], Optional[Read]], now: Callable[[], datetime],
                 outlook_hours: int):
        self.repo, self.load_read, self.now, self.outlook_hours = repo, load_read, now, outlook_hours

    def match(self, event_id: str) -> dict:
        now = self.now()
        with self.repo.session_factory() as session:
            row = self.repo.fixture(session, int(event_id))
            if row is None:
                raise LookupError("Unknown match")
            refs = refs_for(self.repo, session, [row["referee"]])
            seasons: Dict[int, list] = {}
            teams = {side: self._team(session, row, side, seasons) for side in ("home", "away")}
            referee = self._referee(session, row, refs.get(row["referee"]))
            h2h = [meeting(r) for r in self.repo.h2h(session, row["home_team_id"], row["away_team_id"],
                                                    since_year=H2H_SINCE, before=kickoff_utc(row),
                                                    limit=H2H_MATCHES)]
        read = self.load_read(str(row["api_football_id"]))
        fixture = {**match_entry(row, refs, load_read=lambda _: read, now=now, outlook_hours=self.outlook_hours),
                   "status": row["status"]}
        dates = [m["date"] for side in teams.values() for m in side["last_matches"]]
        dates += [m["date"] for m in (referee or {}).get("last_matches", [])] + [m["date"] for m in h2h]
        return {"schema_version": SCHEMA_VERSION, "fixture": fixture,
                "read": None if read is None else read_block(read[0], read[1], status=fixture["read"]["status"]),
                "teams": teams, "referee": referee, "h2h": h2h, "lineups": None,
                "data_basis": {"latest_finished_kickoff": max(dates) if dates else None, "generated_at": iso(now),
                               "source": "canonical_database", "h2h_since_season": H2H_SINCE}}

    def lineup_facts(self, event_id: str) -> dict:
        """What the line-ups need from the database before asking the provider (D135): the kickoff and each team's
        likely eleven from its last matches."""
        with self.repo.session_factory() as session:
            row = self.repo.fixture(session, int(event_id))
            if row is None:
                raise LookupError("Unknown match")
            before = kickoff_utc(row)
            likely = {side: likely_eleven(self.repo.squad_rows(session, row[f"{side}_team_id"], before=before,
                                                               matches=STARTS_WINDOW)) for side in ("home", "away")}
        return {"kickoff": before, "likely": likely}

    def lineup_block(self, facts: Mapping[str, Any], known: Mapping[str, Any]) -> dict:
        """The match page's line-ups block (D135): the provider's elevens with each player's Research Area id, null
        without a profile, and each team's likely eleven, whatever the status."""
        sides = [known[s] for s in ("home", "away") if known.get(s)]
        api_ids = {p["api_player_id"] for side in sides for p in side["start_xi"] + side["bench"]
                   if p["api_player_id"] is not None}
        likely_ids = {p["player_id"] for team in facts["likely"].values() for p in team}
        with self.repo.session_factory() as session:
            by_api = self.repo.player_ids_by_api(session, api_ids)
            profiled = self.repo.players_with_profiles(session, set(by_api.values()) | likely_ids)

        def player(p):
            found = by_api.get(p["api_player_id"])
            return {"player_id": found if found in profiled else None,
                    **{k: p[k] for k in ("name", "number", "position", "grid")}}

        def side(raw):
            return raw and {"formation": raw["formation"], "coach": raw["coach"],
                            "start_xi": [player(p) for p in raw["start_xi"]], "bench": [player(p) for p in raw["bench"]]}
        return {"status": known["status"], "checked_at": known["checked_at"],
                "home": side(known.get("home")), "away": side(known.get("away")),
                "likely": {s: [{**p, "player_id": p["player_id"] if p["player_id"] in profiled else None}
                               for p in team] for s, team in facts["likely"].items()}}

    def _team(self, session, row, side, seasons) -> dict:
        team_id, before, year = row[f"{side}_team_id"], kickoff_utc(row), row["season_year"]
        starters = likely_starters(self.repo.squad_rows(session, team_id, before=before, matches=STARTS_WINDOW))
        form = defaultdict(list)
        for item in self.repo.appearances(session, [p["player_id"] for p in starters], before=before,
                                          limit=SHOOTER_FORM):
            form[item["player_id"]].append(appearance(item))
        scope = self.repo.domestic_season(session, team_id, year=year)
        allowed = None
        if scope is not None:
            if scope["season_id"] not in seasons:
                seasons[scope["season_id"]] = self.repo.team_rows(session, scope["season_id"])
            ranked = league_rank(seasons[scope["season_id"]], team_id)
            allowed = ranked and {"competition": {"code": scope["code"], "name": scope["name"]}, **ranked}
        return {"team": {"id": team_id, **{k: row[f"{side}_{k}"] for k in ("name", "short_code", "logo_url")}},
                "last_matches": [last_match(r) for r in self.repo.team_recent(session, team_id, before=before,
                                                                              limit=LAST_MATCHES)],
                "booked": [{"player_id": r["player_id"], "name": r["name"], "yellow": int(r["yellow"]),
                            "red": int(r["red"]), "appearances": r["appearances"]}
                           for r in self.repo.booked(session, team_id, year=year, before=before, limit=BOOKED)],
                "shooters": [{**p, "last_10": form[p["player_id"]]} for p in starters],
                "allowed": allowed}

    def _referee(self, session, row, ref) -> Optional[dict]:
        """The named referee's season in every competition, the match's competition's average, his last 10."""
        if ref is None:
            return None
        before, year, key = kickoff_utc(row), row["season_year"], ref["referee_key"]
        names = self.repo.season_referee_names(session, year=year)
        aliases = {r["raw_name"] for r in self.repo.alias_rows(session, keys=[key])} | {row["referee"]}
        aliases |= {name for name in names if raw_referee_key(name) == key}
        season = [r for r in self.repo.referee_rows(session, year, sorted(aliases))
                  if kickoff_utc(r) < before]
        matches, per_match = cards_per_match(season)
        league_matches, league_per_match = cards_per_match(
            self.repo.season_matches(session, row["season_id"], before=before))
        return {**ref, "matches": matches, "cards_per_match": per_match,
                "league": {"code": row["competition_code"], "name": row["competition_name"]},
                "league_matches": league_matches, "league_cards_per_match": league_per_match,
                "last_matches": [{"date": iso(r["kickoff_utc"]),
                                  "competition": {"code": r["competition"], "name": r["competition_name"]},
                                  "home": r["home_name"], "away": r["away_name"], "cards": _played(r)["cards"]}
                                 for r in self.repo.referee_recent(session, sorted(aliases), before=before,
                                                                   limit=REFEREE_MATCHES)]}
