"""Descriptive research profiles, independent of prediction and selection logic.

Played players are observed; team statistics require two non-empty rows.
Fixture results remain available independently of the statistics. The same policy
feeds aggregation, peer builds, frequencies and charts without rewriting source.
"""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import math
import re
import time
from uuid import uuid4

try:
    from referee_names import _dedup_referee_names, _normalize_ref_name
except ModuleNotFoundError:  # python -m Scripts.data_platform from the repo root
    from Scripts.referee_names import _dedup_referee_names, _normalize_ref_name
from ..repositories.research import COMPETITION_PHASE, ResearchRepository
from ..research_stats import COUNT_NULL_POLICY, RANK_BUILD_PREFIX, normalize_research_row

PLAYER_MINUTES = 900
REFEREE_MATCHES = 15
# Teams need enough stats-observed matches to be ranked. After the comparable-round
# filter every main-stage side has at least six (a Conference League league stage);
# ten would drop every Champions/Europa League side eliminated after eight matches.
TEAM_MATCHES = 6
PLAYER_COUNTS = {
    "goals": "goals", "assists": "assists", "shots": "shots_total",
    "shots_on_target": "shots_on", "fouls_won": "fouls_drawn",
    "fouls_committed": "fouls_committed", "yellow_cards": "yellow_cards",
    "red_cards": "red_cards", "cards": "cards",
}
TEAM_METRICS = {
    "goals_for": ("result_goals_for", True), "goals_against": ("result_goals_against", False),
    "xg_for": ("expected_goals", True), "xg_against": ("opponent_expected_goals", False),
    "shots_on_target_for": ("shots_on", True), "shots_on_target_faced": ("opponent_shots_on", False),
    "corners_for": ("corners", True), "corners_against": ("opponent_corners", False),
    "possession": ("possession", True), "cards": ("cards", True),
}
PRIORITIES = {
    "player": ("goals_per_90", "shots_on_target_per_90", "assists_per_90", "shots_per_90"),
    "team": ("goals_against", "goals_for", "xg_against", "xg_for", "corners_for"),
    "referee": ("cards", "cards_per_foul", "fouls", "reds"),
}
# A headline describes what the player's role is judged on.
PLAYER_PRIORITIES = {
    "G": ("average_rating", "pass_accuracy_pct"),
    "D": ("duel_win_pct", "cards_per_90", "pass_accuracy_pct", "average_rating"),
    "M": ("assists_per_90", "shots_on_target_per_90", "goals_per_90", "pass_accuracy_pct", "duel_win_pct"),
    "F": PRIORITIES["player"],
}
POSITIONS = {"G": "goalkeepers", "D": "defenders", "M": "midfielders", "F": "forwards"}
POSITION_LABELS = {"G": "Goalkeeper", "D": "Defender", "M": "Midfielder", "F": "Forward"}
METRIC_LABELS = {
    **{k + "_per_90": k.replace("_", " ") + " per 90 minutes" for k in PLAYER_COUNTS},
    "duel_win_pct": "duel win rate", "pass_accuracy_pct": "pass accuracy",
    "shooting_accuracy_pct": "shooting accuracy", "conversion_pct": "shot conversion",
    "average_rating": "average rating", "goals_for": "goals scored per match",
    "goals_against": "goals conceded per match", "xg_for": "expected goals per match",
    "xg_against": "expected goals conceded per match", "shots_on_target_for": "shots on target per match",
    "shots_on_target_faced": "shots on target faced per match", "corners_for": "corners per match",
    "corners_against": "corners conceded per match", "possession": "possession",
    "cards": "cards per match", "yellows": "yellow cards per match", "reds": "red cards per match",
    "fouls": "fouls per match", "cards_per_foul": "cards per foul",
}


def utcnow():
    return datetime.now(timezone.utc)


def iso(value):
    if value is None:
        return None
    return value.replace(tzinfo=value.tzinfo or timezone.utc).isoformat()


def add_known(*values):
    return sum(values) if all(v is not None for v in values) else None


def recorded_sum(values):
    known = [v for v in values if v is not None]
    return sum(known) if known else None


def coverage(values):
    observed = sum(v is not None for v in values)
    return {"observed_matches": observed, "missing_matches": len(values) - observed}


def frequency(values, predicate):
    known = [v for v in values if v is not None]
    return {"count": sum(bool(predicate(v)) for v in known), "denominator": len(known),
            "unknown": len(values) - len(known)}


def percentiles(values):
    """Midrank empirical percentiles; round uses Python's half-to-even rule."""
    ordered = sorted(values)
    n = len(ordered)
    return [round(100 * (bisect_left(ordered, v) + .5 *
                        (bisect_right(ordered, v) - bisect_left(ordered, v))) / n)
            for v in values] if n else []


def starter(row):
    if row.get("substitute") is not None:
        return not row["substitute"], "substitute_flag"
    return row["minutes"] >= 60, "minutes_fallback"


def most_common_position(rows):
    counts = Counter(r["position"] for r in rows if r.get("position") in POSITIONS)
    return min(counts, key=lambda p: (-counts[p], p)) if counts else "unknown"


def player_role(starts, appearances, minutes):
    rate = starts / appearances if appearances else None
    return {"role": ("regular_starter" if rate >= .7 else "rotation" if rate >= .3 else "mostly_substitute")
            if rate is not None else None, "starts": starts, "appearances": appearances, "start_rate": rate,
            "average_minutes_per_appearance": minutes / appearances if appearances else None}


def squad_leaders(rows):
    result = {}
    for name, field in (("goals", "goals"), ("assists", "assists"), ("shots_on_target", "shots_on"), ("yellow_cards", "yellow_cards")):
        ordered = sorted(rows, key=lambda r: (-r[field], r["minutes"], r["name"].casefold(), r["player_id"]))
        result[name] = [{"player_id": r["player_id"], "name": r["name"], "value": r[field],
                         "appearances": r["appearances"]} for r in ordered[:3]]
    return result


def referee_country(aliases):
    countries = {s.split(",", 1)[1].strip() for s in aliases if "," in s and s.split(",", 1)[1].strip()}
    return next(iter(countries)) if len(countries) == 1 else None


def metric(value, unit, values, *, higher=True, basis="known_matches_after_count_null_policy"):
    return {"value": value, "unit": unit, "higher_is_more": higher,
            "coverage": coverage(values), "basis": basis}


def ratio_metric(rows, numerator, denominator):
    pairs = [r for r in rows if r.get(numerator) is not None and r.get(denominator) is not None]
    den = sum(r[denominator] for r in pairs)
    value = 100 * sum(r[numerator] for r in pairs) / den if den else None
    values = [1 if r.get(numerator) is not None and r.get(denominator) is not None else None for r in rows]
    result = metric(value, "percent", values, basis="sum_over_sum_after_count_null_policy")
    result["numerator"] = sum(r[numerator] for r in pairs)
    result["denominator"] = den
    return result


def player_summary(rows):
    rows = [normalize_research_row("player", r) for r in rows]
    rows = [dict(r, cards=add_known(r.get("yellow_cards"), r.get("red_cards")))
            for r in rows if r["_stats_observed"]]
    minutes = sum(r["minutes"] for r in rows)
    starts = [starter(r) for r in rows]
    methods = Counter(method for _, method in starts)
    totals = {name: recorded_sum([r.get(field) for r in rows]) for name, field in PLAYER_COUNTS.items()}
    metrics = {}
    for name, field in PLAYER_COUNTS.items():
        values = [r.get(field) for r in rows]
        value = totals[name] * 90 / minutes if totals[name] is not None and minutes else None
        metrics[name + "_per_90"] = metric(value, "per_90", values, basis="count_sum_after_null_policy_over_all_appearance_minutes")
    metrics["duel_win_pct"] = ratio_metric(rows, "duels_won", "duels_total")
    metrics["pass_accuracy_pct"] = ratio_metric(rows, "passes_accurate", "passes_total")
    metrics["shooting_accuracy_pct"] = ratio_metric(rows, "shots_on", "shots_total")
    metrics["conversion_pct"] = ratio_metric(rows, "goals", "shots_total")
    ratings = [r.get("rating") for r in rows]
    known = [v for v in ratings if v is not None]
    metrics["average_rating"] = metric(sum(known) / len(known) if known else None, "rating", ratings)
    checks = {"shots_on_target_1_plus": ("shots_on", 1), "shots_on_target_2_plus": ("shots_on", 2),
              "shots_3_plus": ("shots_total", 3), "scored": ("goals", 1),
              "assisted": ("assists", 1), "booked": ("cards", 1)}
    return {"matches": len(rows), "minutes": minutes, "peer_group": most_common_position(rows),
            "totals": {"appearances": len(rows), "minutes": minutes, **totals,
                       "starts": sum(v for v, _ in starts)},
            "starts_basis": {"method": next(iter(methods)) if len(methods) == 1 else "mixed" if methods else "unavailable",
                             "substitute_flag_matches": methods["substitute_flag"],
                             "minutes_fallback_matches": methods["minutes_fallback"]},
            "metrics": metrics, "frequencies": {name: frequency([r.get(field) for r in rows], lambda v, t=t: v >= t)
                                                  for name, (field, t) in checks.items()}}


def team_summary(rows):
    rows = [normalize_research_row("team", r) for r in rows]
    rows = [dict(r, cards=add_known(r.get("yellow_cards"), r.get("red_cards"))) for r in rows]
    metrics = {}
    for name, (field, higher) in TEAM_METRICS.items():
        values = [r.get(field) for r in rows]
        known = [v for v in values if v is not None]
        factor = 100 if name == "possession" else 1
        metrics[name] = metric(factor * sum(known) / len(known) if known else None,
                               "percent" if name == "possession" else "per_match", values, higher=higher)
    outcomes = []
    for r in rows:
        h, a = r["home_goals"], r["away_goals"]
        if h is None or a is None:
            outcomes.append(None)
        else:
            diff = (h - a) * (1 if r["team_id"] == r["home_team_id"] else -1)
            outcomes.append("W" if diff > 0 else "L" if diff < 0 else "D")
    goals = [add_known(r.get("result_goals_for"), r.get("result_goals_against")) for r in rows]
    btts = [int(r["result_goals_for"] > 0 and r["result_goals_against"] > 0) if total is not None else None
            for r, total in zip(rows, goals)]
    clean = [r.get("result_goals_against") for r in rows]
    corners = [add_known(r.get("corners"), r.get("opponent_corners")) for r in rows]
    cards = [add_known(r.get("cards"), r.get("opponent_yellow_cards"), r.get("opponent_red_cards")) for r in rows]
    observed = sum(r["_stats_observed"] for r in rows)
    return {"matches": len(rows), "stats_observed_matches": observed, "excluded_matches": len(rows) - observed,
            "finished_matches": len(rows), "minutes": None, "peer_group": "all",
            "totals": {"matches": len(rows), "wins": outcomes.count("W"), "draws": outcomes.count("D"),
                       "stats_observed_matches": observed, "excluded_matches": len(rows) - observed,
                       "losses": outcomes.count("L"), "unknown_results": outcomes.count(None),
                       **{name: recorded_sum([r.get(field) for r in rows])
                          for name, (field, _) in TEAM_METRICS.items() if name != "possession"},
                       "clean_sheets": sum(v == 0 for v in clean if v is not None)},
            "last_5_form": outcomes[-5:], "metrics": metrics,
            "frequencies": {"over_2_5_goals": frequency(goals, lambda v: v > 2.5),
                            "both_teams_scored": frequency(btts, bool),
                            "clean_sheet": frequency(clean, lambda v: v == 0),
                            "corners_10_plus": frequency(corners, lambda v: v >= 10),
                            "cards_4_plus": frequency(cards, lambda v: v >= 4)}}


def referee_values(row):
    row = normalize_research_row("referee", row)
    yellows = add_known(row.get("home_stats_yellow_cards"), row.get("away_stats_yellow_cards"))
    reds = add_known(row.get("home_stats_red_cards"), row.get("away_stats_red_cards"))
    return {"yellows": yellows, "reds": reds, "cards": add_known(yellows, reds),
            "fouls": add_known(row.get("home_stats_fouls_committed"), row.get("away_stats_fouls_committed"))}


def referee_summary(rows):
    rows = [normalize_research_row("referee", r) for r in rows]
    values = [referee_values(r) for r in rows]
    observed = sum(r["_stats_observed"] for r in rows)
    metrics, totals = {}, {"matches": observed, "finished_matches": len(rows), "stats_observed_matches": observed,
                          "excluded_matches": len(rows) - observed}
    for key in ("cards", "yellows", "reds", "fouls"):
        vals = [r[key] for r in values]
        totals[key] = recorded_sum(vals)
        n = sum(v is not None for v in vals)
        metrics[key] = metric(totals[key] / n if n else None, "per_match", vals)
    # No multiplication by 100 for cards per foul.
    ratio = ratio_metric(values, "cards", "fouls")
    ratio["value"] = ratio["value"] / 100 if ratio["value"] is not None else None
    ratio["unit"] = "cards_per_foul"
    metrics["cards_per_foul"] = ratio
    return {"matches": observed, "finished_matches": len(rows), "stats_observed_matches": observed,
            "excluded_matches": len(rows) - observed, "minutes": None, "peer_group": "all", "totals": totals,
            "metrics": metrics, "frequencies": {
                name: frequency([r[field] for r in values], lambda v, t=t: v >= t)
                for name, field, t in (("cards_4_plus", "cards", 4), ("cards_5_plus", "cards", 5),
                                       ("red_card_shown", "reds", 1), ("fouls_25_plus", "fouls", 25))}}


def eligibility(kind, summary):
    if kind == "player" and summary["minutes"] < PLAYER_MINUTES:
        return False, "below_minutes_threshold"
    if kind == "player" and summary["peer_group"] == "unknown":
        return False, "unknown_position"
    if kind == "referee" and summary["matches"] < REFEREE_MATCHES:
        return False, "below_matches_threshold"
    if kind == "team" and summary.get("stats_observed_matches", 0) < TEAM_MATCHES:
        return False, "below_matches_threshold"
    if not summary["matches"]:
        return False, "no_finished_matches"
    return True, None


def scope_key(scope, referee=False):
    return f"referee:{scope['year']}" if referee else f"competition:{scope['competition_id']}:{scope['year']}"


def referee_aliases(raw_names):
    normalized = sorted({n for raw in raw_names if (n := _normalize_ref_name(raw))})
    empty = {n: dict(name=n, leagues=set(), cards=0, fouls=0, yellows=0, reds=0, matches=0) for n in normalized}
    _, remap = _dedup_referee_names(empty)
    result = []
    for raw in raw_names:
        normalized_name = _normalize_ref_name(raw)
        if not normalized_name:
            continue
        name = remap[normalized_name]
        parts = name.split()
        # Signature, not longest display name, keeps URLs stable when full names arrive.
        signature = parts[0].rstrip(".")[:1] + " " + parts[-1] if len(parts) > 1 else name
        slug = re.sub(r"[^\w]+", "-", signature, flags=re.UNICODE).strip("-")
        key = slug[:110] + "-" + sha256(signature.encode()).hexdigest()[:8]
        result.append({"raw_name": raw, "referee_key": key, "name": name.title()})
    return result


def rank_rows(kind, summaries, scope, build):
    grouped = defaultdict(list)
    for key, summary in summaries.items():
        if not eligibility(kind, summary)[0]:
            continue
        for name, data in summary["metrics"].items():
            if data["value"] is not None and math.isfinite(data["value"]):
                grouped[(summary["peer_group"], name)].append((key, summary, data))
    result = []
    for (group, name), peers in grouped.items():
        values = [m["value"] for _, _, m in peers]
        ordered = sorted(values)
        # Team league average weights the available team-match observations equally.
        observed = sum(m["coverage"]["observed_matches"] for _, _, m in peers)
        avg = sum(m["value"] * m["coverage"]["observed_matches"] for _, _, m in peers) / observed if observed else None
        for (key, summary, data), pct in zip(peers, percentiles(values)):
            value = data["value"]
            rank = (len(values) - bisect_right(ordered, value) if data["higher_is_more"] else bisect_left(ordered, value)) + 1
            result.append({**build, "subject_type": kind, "subject_key": str(key), "peer_group": group,
                           "metric": name, "value": value, "percentile": pct, "rank": rank,
                           "peer_count": len(peers), "league_average": avg if kind == "team" else None,
                           "sample_minutes": summary["minutes"], "sample_matches": summary["matches"]})
    return result


def ordinal(number):
    suffix = "th" if 10 <= number % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def rank_phrase(rank, peer_count, higher_is_more):
    """Describe a league position from the nearer end of the table.

    Rank 1 is the most for "higher is more" metrics and the fewest otherwise, so
    rank 20 of 20 for cards reads "the fewest", not "the 20th most".
    """
    top, bottom = ("most", "fewest") if higher_is_more else ("fewest", "most")
    from_bottom = peer_count - rank + 1
    if rank <= from_bottom:
        word, place = top, rank
    else:
        word, place = bottom, from_bottom
    return f"the {word} of {peer_count} teams" if place == 1 else f"the {ordinal(place)} {word} of {peer_count} teams"


def metric_comparison(kind, name, data, scope, group):
    pct, value = data.get("percentile"), data["value"]
    label = METRIC_LABELS.get(name, name.replace("_", " "))
    peers = (f"{scope['name']} {POSITIONS.get(group, 'players')}" if kind == "player"
             else "teams" if kind == "team" else "referees")
    display = (None if value is None else f"{value:.0f}%" if data["unit"] == "percent"
               else f"{value:.2f}")
    result = {"metric_label": label, "display_value": display, "peer_label": peers,
              "comparison": None, "comparison_share": None, "rank_phrase": None, "comparison_text": None}
    if pct is None:
        return result
    if kind == "team":
        result.update(rank_phrase=(phrase := rank_phrase(data["rank"], data["peer_count"], data["higher_is_more"])),
                      comparison_text=phrase)
    else:
        higher = pct >= 50
        word = ("higher" if higher else "lower") if data["unit"] in ("percent", "rating") else ("more" if higher else "fewer")
        share = pct if higher else 100 - pct
        result.update(comparison="higher" if higher else "lower", comparison_share=share,
                      comparison_text=f"{word} than {share}% of {peers}")
    return result


def headline(kind, subject, scope, metrics, group):
    priorities = PLAYER_PRIORITIES.get(group, PRIORITIES["player"]) if kind == "player" else PRIORITIES[kind]
    # A zero is never the lead: "0.00 assists per 90" says nothing about a goalkeeper.
    candidates = [(name, metrics[name]) for name in priorities
                  if name in metrics and metrics[name].get("percentile") is not None and metrics[name]["value"]]
    if not candidates:
        return None
    name, data = max(candidates, key=lambda item: abs(item[1]["percentile"] - 50))
    comparison = metric_comparison(kind, name, data, scope, group)
    text = f"{comparison['display_value']} {comparison['metric_label']}, {comparison['comparison_text']}"
    partial = data["coverage"]["missing_matches"] > 0
    if partial:
        text += ". Some match observations are missing."
    return {"subject_type": kind, "metric": name, "value": data["value"],
            "percentile": data["percentile"] if kind != "team" else None,
            "rank": data["rank"], "peer_count": data["peer_count"],
            **{k: v for k, v in comparison.items() if k != "comparison_text"},
            "partial_observations": partial, "text": text}


class ResearchProfileService:
    def __init__(self, repo=None):
        self.repo = repo or ResearchRepository()

    def build(self, *, season=None, competition=None):
        if season is not None and not 1900 <= season <= 2100:
            raise ValueError("Season must be a starting year from 1900 to 2100")
        started = time.perf_counter()
        build_id, built_at = RANK_BUILD_PREFIX + uuid4().hex, utcnow()
        with self.repo.session_factory() as session:
            comp_id = self.repo.competition(session, competition)
            scopes = self.repo.scopes(session, year=season, competition_id=comp_id)
            if not scopes:
                raise ValueError("No seasons in requested build scope")
            aliases = referee_aliases(self.repo.referee_names(session))
            alias_map = {r["raw_name"]: r["referee_key"] for r in aliases}
            builds, ranks = [], []
            for scope in scopes:
                build = {"scope_key": scope_key(scope), "competition_id": scope["competition_id"],
                         "season_id": scope["season_id"], "season_year": scope["year"],
                         "build_id": build_id, "built_at": built_at}
                builds.append(build)
                for kind, rows, summary_fn in (
                    ("player", self.repo.player_rows(session, scope["season_id"]), player_summary),
                    ("team", self.repo.team_rows(session, scope["season_id"]), team_summary),
                ):
                    groups = defaultdict(list)
                    for row in rows:
                        groups[row[kind + "_id"]].append(row)
                    summaries = {key: summary_fn(rs) for key, rs in groups.items()}
                    ranks.extend(rank_rows(kind, summaries, scope, build))
            # A competition-limited build still refreshes ALL competitions for
            # referee ranks in the affected years, preserving their declared scope.
            for year in sorted({s["year"] for s in scopes}):
                scope = {"year": year}
                build = {"scope_key": scope_key(scope, True), "competition_id": None,
                         "season_id": None, "season_year": year, "build_id": build_id, "built_at": built_at}
                builds.append(build)
                groups = defaultdict(list)
                for row in self.repo.referee_rows(session, year):
                    if row["referee"] in alias_map:
                        groups[alias_map[row["referee"]]].append(row)
                ranks.extend(rank_rows("referee", {key: referee_summary(rs) for key, rs in groups.items()}, scope, build))
            self.repo.replace(session, builds, ranks, aliases)
        return {"build_id": build_id, "built_at": iso(built_at), "count_null_policy": COUNT_NULL_POLICY, "scopes": len(builds),
                "peer_rows": len(ranks), "referee_aliases": len(aliases),
                "elapsed_seconds": round(time.perf_counter() - started, 3)}

    def _ranking(self, session, kind, key, summary, scope):
        build, stored = self.repo.ranks(session, scope_key(scope, kind == "referee"), kind, key)
        eligible, reason = eligibility(kind, summary)
        metrics = {name: {**data, "rank": None, "percentile": None, "peer_count": None,
                          "league_average": None} for name, data in summary["metrics"].items()}
        matching = {}
        for row in stored:
            data = metrics.get(row["metric"])
            if (data and data["value"] is not None and row["peer_group"] == summary["peer_group"]
                and row["sample_matches"] == summary["matches"] and row["sample_minutes"] == summary["minutes"]
                and math.isclose(data["value"], row["value"], rel_tol=1e-10, abs_tol=1e-10)):
                matching[row["metric"]] = row
        if eligible and not any(d["value"] is not None for d in metrics.values()):
            reason = "no_recorded_metrics"
        elif eligible and not build:
            reason = "ranks_not_built"
        elif eligible and not build.build_id.startswith(RANK_BUILD_PREFIX):
            reason = "rank_policy_changed_rebuild_required"
        elif eligible and len(matching) != sum(d["value"] is not None for d in metrics.values()):
            reason = "ranks_stale_or_missing"
        ranked = eligible and reason is None
        if ranked:
            for name, row in matching.items():
                metrics[name].update({k: row[k] for k in ("rank", "percentile", "peer_count", "league_average")})
        return {"ranked": ranked, "reason": reason, "peer_group": summary["peer_group"],
                "sample_minutes": summary["minutes"], "sample_matches": summary["matches"],
                "minimum_minutes": PLAYER_MINUTES if kind == "player" else None,
                "minimum_matches": {"referee": REFEREE_MATCHES, "team": TEAM_MATCHES}.get(kind, 1)}, metrics, build

    def _payload(self, session, kind, subject, scope, rows, *, include_context=True):
        rows = [normalize_research_row(kind, r) for r in rows]
        summary = {"player": player_summary, "team": team_summary, "referee": referee_summary}[kind](rows)
        key = subject["referee_key"] if kind == "referee" else subject["id"]
        ranking, metrics, build = self._ranking(session, kind, key, summary, scope)
        for name, data in metrics.items():
            data.update(metric_comparison(kind, name, data, scope, summary["peer_group"]))
        subject = dict(subject)
        ids = {r[k] for r in (rows if include_context else rows[-10:]) for k in ("home_team_id", "away_team_id")}
        teams = self.repo.team_details(session, ids)
        if kind == "player":
            club = teams.get(rows[-1]["team_id"], {}) if rows else {}
            subject.update(club={k: club.get(k) for k in ("id", "name", "logo_url")},
                           position={"code": summary["peer_group"], "label": POSITION_LABELS.get(summary["peer_group"])},
                           photo_url=f"https://media.api-sports.io/football/players/{subject['api_football_id']}.png")
        elif kind == "referee":
            subject["country"] = referee_country(subject["aliases"])
        competition = ({"id": scope["competition_id"], "code": scope["code"], "name": scope["name"]}
                       if kind != "referee" else None)
        season = {"year": scope["year"], "label": scope["label"], "id": scope.get("season_id")}
        result = {"schema_version": "research-profile.v1", "subject": {"type": kind, **subject},
                  "competition": competition, "season": season, "totals": summary["totals"],
                  "metrics": metrics, "ranking": ranking, "frequencies": summary["frequencies"],
                  "headline": headline(kind, subject, scope, metrics, summary["peer_group"]),
                  "data_basis": {"source": "canonical_database", "matches": summary["matches"],
                                 "minutes": summary["minutes"], "competition": competition, "season": season,
                                 "finished_matches": summary.get("finished_matches", summary["matches"]),
                                 "stats_observed_matches": summary.get("stats_observed_matches", summary["matches"]),
                                 "excluded_matches": summary.get("excluded_matches", 0),
                                 "finished_statuses": ["FT", "AET", "PEN"],
                                 "competition_phase": COMPETITION_PHASE if kind != "referee" else "all_finished_matches",
                                 "match_scope": "as_stored_including_extra_time_when_present",
                                 "generated_at": iso(utcnow()), "build_id": build.build_id if build else None,
                                 "ranks_built_at": iso(build.built_at) if build else None,
                                 "null_policy": "played_players_observed; team_stats_require_both_nonempty_sides; count_nulls_zero; measurements_nullable; results_from_fixture_score",
                                 "count_null_policy_version": COUNT_NULL_POLICY,
                                 "stats_rows": {side: {
                                     "populated": sum(r["_stats_available"][side] for r in rows),
                                     "empty": sum(not r["_stats_available"][side] for r in rows),
                                     "count_fields_filled_zero": sum(len(r["_zero_filled_counts"][side]) for r in rows),
                                 } for side in rows[0]["_stats_available"]} if rows else {},
                                 "frequency_scope": "whole_selected_season",
                                 "cards_definition": "yellow_plus_red; booked_means_either"},
                  "previous_season": None}
        if kind == "player":
            result["starts_basis"] = summary["starts_basis"]
            result["role_summary"] = player_role(summary["totals"]["starts"], summary["matches"], summary["minutes"])
            result["shooting_funnel"] = {k: summary["totals"][k] for k in ("shots", "shots_on_target", "goals")}
            result["data_basis"]["shots_off_target_definition"] = "shots_minus_on_target_including_blocked"
        if kind == "team":
            result["last_5_form"] = summary["last_5_form"]
            result["home_away"] = {side: team_summary([r for r in rows if (r["team_id"] == r["home_team_id"]) == home])
                                   for side, home in (("home", True), ("away", False))}
            if include_context:
                result["squad_leaders"] = squad_leaders(self.repo.squad_leaders(session, key, scope["season_id"]))
        if kind == "referee":
            result["competition_breakdown"] = [{"competition": code, **referee_summary([r for r in rows if r["competition"] == code])}
                                                 for code in sorted({r["competition"] for r in rows})]
            result["data_basis"]["identity_method"] = "legacy_first_initial_last_name_grouping"
            for group in result["competition_breakdown"]:
                group["small_sample"] = group["matches"] < 5
        result["last_10"] = [self._match(kind, row, teams) for row in rows[-10:]]
        if include_context:
            # Every match in the selected scope, oldest first, for the match-by-match charts.
            result["match_log"] = [self._match(kind, row, teams, full=True) for row in rows]
        if include_context:
            result["available_scopes"] = []
            for item in self.repo.available_scopes(session, kind, key, subject.get("aliases", ())):
                available = {"competition": {"code": item["code"], "name": item["name"]},
                             "season": {"year": item["year"], "label": item["label"]},
                             "matches": item["matches"], "ranked": item["ranked"]}
                if kind == "player":
                    available["minutes"] = item["minutes"]
                if kind == "referee":
                    available.update(finished_matches=item["finished_matches"], excluded_matches=item["finished_matches"] - item["matches"])
                # The requested profile has just performed the full stale-rank check.
                if item["year"] == scope["year"] and (kind == "referee" or item["code"] == scope["code"]):
                    available["ranked"] = ranking["ranked"]
                result["available_scopes"].append(available)
        return result

    @staticmethod
    def _match(kind, row, teams, full=False):
        row = normalize_research_row(kind, row)
        home, away = row["home_team_id"], row["away_team_id"]
        base = {"fixture_id": row["fixture_id"], "date": iso(row["kickoff_utc"]), "status": row["status"],
                "home": {"id": home, **teams.get(home, {})}, "away": {"id": away, **teams.get(away, {})},
                "score": {"home": row["home_goals"], "away": row["away_goals"]},
                "stats_available": row["_stats_available"], "stats_observed": row["_stats_observed"]}
        if kind == "referee":
            values = referee_values(row)
            if full:
                values.update({f"{side}_{name}": add_known(*[row.get(f"{side}_stats_{field}") for field in fields])
                               for side in ("home", "away")
                               for name, fields in (("cards", ("yellow_cards", "red_cards")),
                                                    ("fouls", ("fouls_committed",)))})
            return {**base, "competition": row["competition"], "values": values}
        is_home = row["team_id"] == home
        opponent_id = away if is_home else home
        opponent = {"id": opponent_id, **teams.get(opponent_id, {})}
        parts = opponent.get("name", "").split()
        fallback = "".join(p[0] for p in parts)[:3] if len(parts) > 1 else "".join(parts)[:3]
        opponent["abbreviation"] = opponent.get("short_code") or fallback.upper() or None
        base.update(venue="home" if is_home else "away", opponent=opponent)
        if kind == "player":
            fields = {"minutes": "minutes", "shots": "shots_total", "shots_on_target": "shots_on",
                      "goals": "goals", "yellow_cards": "yellow_cards"}
        else:
            fields = {"goals_for": "result_goals_for", "goals_against": "result_goals_against", "xg_for": "expected_goals",
                      "xg_against": "opponent_expected_goals", "corners_for": "corners", "corners_against": "opponent_corners"}
        base["values"] = {key: row.get(field) for key, field in fields.items()}
        if kind == "player":
            shots, on = row.get("shots_total"), row.get("shots_on")
            base["values"]["shots_off_target"] = shots - on if shots is not None and on is not None and shots >= on else None
        if kind == "team":
            base["values"]["cards"] = add_known(row.get("yellow_cards"), row.get("red_cards"))
        if full and kind == "player":
            base["started"] = starter(row)[0]
            base["values"].update({key: row.get(field) for key, field in (
                ("assists", "assists"), ("passes", "passes_total"), ("passes_accurate", "passes_accurate"),
                ("tackles", "tackles"), ("interceptions", "interceptions"), ("duels_won", "duels_won"),
                ("duels_total", "duels_total"), ("fouls_committed", "fouls_committed"),
                ("fouls_drawn", "fouls_drawn"), ("red_cards", "red_cards"), ("rating", "rating"))})
            base["values"]["cards"] = add_known(row.get("yellow_cards"), row.get("red_cards"))
        if full and kind == "team":
            possession = row.get("possession")
            base["values"].update({
                "shots_for": row.get("shots_total"), "shots_against": row.get("opponent_shots_total"),
                "sot_for": row.get("shots_on"), "sot_against": row.get("opponent_shots_on"),
                "opponent_cards": add_known(row.get("opponent_yellow_cards"), row.get("opponent_red_cards")),
                "red_cards": row.get("red_cards"), "fouls": row.get("fouls_committed"),
                "offsides": row.get("offsides"),
                "possession": round(possession * 100, 1) if possession is not None else None})
        return base

    def profile(self, kind, key, *, competition=None, season=None):
        with self.repo.session_factory() as session:
            comp_id = self.repo.competition(session, competition)
            subject = self.repo.subject(session, kind, key)
            if not subject:
                raise LookupError("Unknown subject")
            if kind == "referee":
                rows = self.repo.referee_rows(session, season, subject["aliases"])
                if not rows:
                    raise LookupError("No finished matches in this season")
                year = season if season is not None else max(r["year"] for r in rows)
                rows = [r for r in rows if r["year"] == year]
                scope = {"year": year, "label": f"{year}/{str(year + 1)[-2:]}", "name": "All competitions"}
            else:
                scope = self.repo.select_scope(session, kind, key, year=season, competition_id=comp_id)
                if not scope:
                    raise LookupError("No appearances in requested competition/season")
                loader = self.repo.player_rows if kind == "player" else self.repo.team_rows
                rows = loader(session, scope["season_id"], key)
            result = self._payload(session, kind, subject, scope, rows)
            if result["ranking"]["reason"] in ("below_minutes_threshold", "below_matches_threshold"):
                previous_year = scope["year"] - 1
                if kind == "referee":
                    prior_rows = self.repo.referee_rows(session, previous_year, subject["aliases"])
                    prior_scope = {"year": previous_year, "label": f"{previous_year}/{str(previous_year + 1)[-2:]}", "name": "All competitions"}
                else:
                    prior_scope = self.repo.select_scope(session, kind, key, year=previous_year, competition_id=scope["competition_id"])
                    prior_rows = loader(session, prior_scope["season_id"], key) if prior_scope else []
                if prior_rows:
                    prior = self._payload(session, kind, subject, prior_scope, prior_rows, include_context=False)
                    result["previous_season"] = {k: prior[k] for k in ("season", "competition", "ranking", "metrics", "data_basis")}
            return result

    def search(self, kind, q, *, competition=None, limit=20):
        with self.repo.session_factory() as session:
            comp_id = self.repo.competition(session, competition)
            rows = self.repo.search(session, kind, q, comp_id, limit)
        for row in rows:
            row["last_activity_at"] = iso(row["last_activity_at"])
            if kind == "players":
                row["club"] = {"id": row.pop("club_id"), "name": row["club_name"], "logo_url": row.pop("club_logo_url")}
                code = row["position"]
                row["position"] = {"code": code, "label": POSITION_LABELS.get(code)}
            elif kind == "referees":
                row["country"] = referee_country(row.pop("_aliases"))
        return {"schema_version": "research-search.v1", "type": kind, "query": q, "results": rows,
                "count": len(rows), "data_basis": {"source": "canonical_database", "competition": competition,
                "season": None, "matches": None, "minutes": None, "build_id": None,
                "generated_at": iso(utcnow()), "scope": "subjects_with_finished_data; referee_aliases_from_latest_build"}}
