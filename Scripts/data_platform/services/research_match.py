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
from .research_profiles import add_known, iso, most_common_position, starter, team_summary
from .research_upcoming import Read, _figure, _time, kickoff_utc, match_entry, one_decimal, refs_for

SCHEMA_VERSION = "research-match.v1"
TOTALS_KEYS = {"cards": "totals_cards_over_under", "goals": "totals", "corners": "totals_corners_over_under"}
LAST_MATCHES, BOOKED, SHOOTERS, SHOOTER_FORM, STARTS_WINDOW = 10, 5, 8, 10, 5


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
        read = self.load_read(str(row["api_football_id"]))
        fixture = {**match_entry(row, refs, load_read=lambda _: read, now=now, outlook_hours=self.outlook_hours),
                   "status": row["status"]}
        return {"schema_version": SCHEMA_VERSION, "fixture": fixture,
                "read": None if read is None else read_block(read[0], read[1], status=fixture["read"]["status"]),
                "teams": teams, "referee": None, "h2h": [], "lineups": None,
                "data_basis": {"generated_at": iso(now)}}

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
