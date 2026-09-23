"""Read-only historical data diagnostics. Evidence, never automatic repairs."""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
import json
import random
import sqlite3

from .historical_recovery import numeric
from ..sync.upserts import _STAT_KEY_MAP


FIELDS = ("goals", "corners", "shots_on", "yellow_cards", "red_cards", "shots_total",
          "fouls_committed", "expected_goals", "possession", "pass_accuracy")
MARKETS = {"goals": ("goals",), "corners": ("corners",), "sot": ("shots_on",),
           "cards": ("yellow_cards", "red_cards")}
FRACTIONS = {"possession", "pass_accuracy"}
API_FIELDS = {column: label for label, column in _STAT_KEY_MAP.items() if column in FIELDS}


def state(value, field):
    n = numeric(value)
    if n is None:
        return "unknown" if value is None else "invalid"
    if n < 0 or (field in FRACTIONS and n > 1) or (field not in FRACTIONS | {"expected_goals"} and not n.is_integer()):
        return "invalid"
    return "numeric_zero" if n == 0 else "numeric_positive"


def snapshot(database, seasons):
    """Short consistent read; LEFT JOINs retain bad catalogue links for audit."""
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        checks = {"quick_check": [r[0] for r in db.execute("PRAGMA quick_check")],
                  "foreign_key_violations": [list(r) for r in db.execute("PRAGMA foreign_key_check")]}
        fixtures = [dict(r) for r in db.execute("""
            SELECT f.id AS internal_id, f.api_football_id AS fixture_id, f.competition_id,
                   f.season_id, c.code AS league, c.api_football_id AS league_id, s.year AS season,
                   s.competition_id AS season_competition_id,
                   f.home_team_id, f.away_team_id, h.api_football_id AS home_id,
                   a.api_football_id AS away_id, h.name AS home, a.name AS away,
                   f.kickoff_utc AS kickoff, f.status, f.home_goals, f.away_goals, f.last_fetched_at
            FROM fixtures f LEFT JOIN competitions c ON c.id=f.competition_id
            LEFT JOIN seasons s ON s.id=f.season_id
            LEFT JOIN teams h ON h.id=f.home_team_id LEFT JOIN teams a ON a.id=f.away_team_id
            ORDER BY c.code,s.year,f.api_football_id
        """)]
        team_rows = [dict(r) for r in db.execute("SELECT * FROM fixture_team_stats ORDER BY fixture_id,team_id")]
    by_fixture = defaultdict(list)
    for row in team_rows:
        by_fixture[row["fixture_id"]].append(row)
    checks["duplicate_provider_ids"] = [fid for fid, n in Counter(f["fixture_id"] for f in fixtures).items() if n > 1]
    checks["missing_catalogue_links"] = [f["fixture_id"] for f in fixtures
                                         if any(f[k] is None for k in ("league", "season", "home_id", "away_id"))]
    ids = {f["internal_id"] for f in fixtures}
    checks["orphan_team_rows"] = [r["id"] for r in team_rows if r["fixture_id"] not in ids]
    records = []
    for fixture in fixtures:
        if fixture["season"] not in seasons:
            continue
        rows = []
        for row in by_fixture[fixture["internal_id"]]:
            try:
                parsed = json.loads(row["stats_json"]) if row["stats_json"] else {}
                if not isinstance(parsed, dict):
                    parsed = {"_invalid_json": True}
            except (TypeError, ValueError):
                parsed = {"_invalid_json": True}
            side = "home" if row["team_id"] == fixture["home_team_id"] else "away"
            rows.append({"team_id": row["team_id"], "opponent_team_id": row["opponent_team_id"],
                         "api_team_id": fixture[side + "_id"], "is_home": row["is_home"],
                         "values": {key: row[key] for key in FIELDS}, "stats_json": parsed,
                         "updated_at": row["updated_at"]})
        records.append({**fixture, "team_rows": rows})
    return {"captured_at": datetime.now(timezone.utc).isoformat(), "seasons": seasons,
            "database_checks": checks, "fixtures": records}


def fixture_issues(fixture):
    issues = []
    if any(fixture.get(k) is None for k in ("fixture_id", "league_id", "home_id", "away_id")):
        issues.append("missing_identity")
    if fixture["competition_id"] != fixture["season_competition_id"]:
        issues.append("season_competition_mismatch")
    if fixture["home_id"] == fixture["away_id"]:
        issues.append("same_team_both_sides")
    try:
        datetime.fromisoformat(fixture["kickoff"].replace("Z", "+00:00"))
    except (ValueError, AttributeError, TypeError):
        issues.append("invalid_kickoff")
    rows = fixture["team_rows"]
    if len(rows) != 2 or {r["team_id"] for r in rows} != {fixture["home_team_id"], fixture["away_team_id"]}:
        issues.append("invalid_team_pair")
    for row in rows:
        home = row["team_id"] == fixture["home_team_id"]
        opponent = fixture["away_team_id"] if home else fixture["home_team_id"]
        if bool(row["is_home"]) != home or row["opponent_team_id"] != opponent:
            issues.append("invalid_home_opponent")
        if row["stats_json"].get("_invalid_json"):
            issues.append("invalid_stats_json")
    return sorted(set(issues))


def source_kind(fixture):
    return "legacy_recovery" if any("_history_recovery" in r["stats_json"] for r in fixture["team_rows"]) else "canonical_provider"


def market_known(fixture, market):
    if market == "goals":
        return all(state(fixture[k], "goals").startswith("numeric") for k in ("home_goals", "away_goals"))
    return len(fixture["team_rows"]) == 2 and all(
        state(row["values"][field], field).startswith("numeric")
        for row in fixture["team_rows"] for field in MARKETS[market])


def audit_snapshot(data, leagues):
    cohorts = {f"{league}:{season}": {"statuses": Counter(), "ft": 0, "sources": Counter(),
                "fields": {field: Counter() for field in FIELDS}, "known_labels": Counter(),
                "unknown_labels": Counter(), "card_eligibility_by_recorded_red": {}, "months": {}}
               for league in leagues for season in data["seasons"]}
    quarantined, anomalies = [], []
    for fixture in data["fixtures"]:
        cohort = cohorts.get(f"{fixture['league']}:{fixture['season']}")
        if cohort is None:
            continue
        cohort["statuses"][fixture["status"]] += 1
        if fixture["status"] != "FT":
            continue
        cohort["ft"] += 1
        issues = fixture_issues(fixture)
        if issues:
            quarantined.append({"fixture_id": fixture["fixture_id"], "issues": issues})
            continue
        origin = source_kind(fixture)
        cohort["sources"][origin] += 1
        month = (fixture["kickoff"] or "unknown")[:7]
        month_stats = cohort["months"].setdefault(month, Counter())
        month_stats["fixtures"] += 1
        red_known = all(state(r["values"]["red_cards"], "red_cards").startswith("numeric") for r in fixture["team_rows"])
        red_positive = any((numeric(r["values"]["red_cards"]) or 0) > 0 for r in fixture["team_rows"])
        red_category = "recorded_positive" if red_positive else "recorded_zero" if red_known else "unknown"
        red_stats = cohort["card_eligibility_by_recorded_red"].setdefault(red_category, Counter())
        red_stats["fixtures"] += 1
        for market in MARKETS:
            known = market_known(fixture, market)
            cohort["known_labels" if known else "unknown_labels"][market] += 1
            month_stats[market + "_known"] += int(known)
        red_stats["card_label_known"] += int(market_known(fixture, "cards"))
        for row in fixture["team_rows"]:
            for field, value in row["values"].items():
                classification = state(value, field)
                if classification == "unknown":
                    if field not in row["stats_json"]:
                        classification = "absent_metric"
                    elif origin == "legacy_recovery":
                        classification = "legacy_ambiguous_or_missing"
                    else:
                        classification = "provider_null"
                cohort["fields"][field][classification] += 1
                if classification == "invalid":
                    anomalies.append({"fixture_id": fixture["fixture_id"], "team_id": row["api_team_id"],
                                      "field": field, "issue": "invalid_range_or_type", "value": value})
            v = row["values"]
            if v["shots_total"] is not None and v["shots_on"] is not None and v["shots_on"] > v["shots_total"]:
                anomalies.append({"fixture_id": fixture["fixture_id"], "team_id": row["api_team_id"], "issue": "sot_exceeds_total_shots"})
            score = fixture["home_goals" if row["is_home"] else "away_goals"]
            if v["goals"] is not None and score != v["goals"]:
                anomalies.append({"fixture_id": fixture["fixture_id"], "team_id": row["api_team_id"], "issue": "score_disagreement"})
    return {"cohorts": cohorts, "database_checks": data["database_checks"], "quarantined": quarantined,
            "anomalies": anomalies, "qualification": "Diagnostics only; numerical completeness is not market eligibility."}


def tags(fixture):
    rows = fixture["team_rows"]
    result = [source_kind(fixture)]
    if any(r["values"]["red_cards"] is None for r in rows):
        result.append("unknown_red")
    if any((numeric(r["values"]["red_cards"]) or 0) > 0 for r in rows):
        result.append("positive_red")
    if any(r["values"]["red_cards"] == 0 for r in rows):
        result.append("explicit_zero_red")
    if any(r["values"]["corners"] == 0 or r["values"]["shots_on"] == 0 for r in rows):
        result.append("zero_corners_or_sot")
    if any(not any(v is not None for k, v in r["values"].items() if k != "goals") for r in rows):
        result.append("metadata_only")
    if fixture["home_goals"] == 0 or fixture["away_goals"] == 0:
        result.append("zero_goals")
    return result


def choose_sample(data, leagues, *, size=52, seed=20260922):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for fixture in data["fixtures"]:
        if fixture["league"] in leagues and fixture["status"] == "FT" and not fixture_issues(fixture):
            grouped[(fixture["league"], fixture["season"])].append(fixture)
    if size < len(grouped) * 2:
        raise ValueError("Sample too small for one random control and one targeted case per populated cohort")
    selected, used = [], set()
    def add(fixture, reason):
        used.add(fixture["fixture_id"])
        selected.append({"fixture_id": fixture["fixture_id"], "league": fixture["league"], "season": fixture["season"],
                         "league_id": fixture["league_id"], "selection": reason, "tags": tags(fixture)})
    for index, (key, fixtures) in enumerate(sorted(grouped.items())):
        fixtures = sorted(fixtures, key=lambda f: f["fixture_id"])
        add(rng.choice(fixtures), "random_control")
        desired = ("unknown_red", "positive_red", "explicit_zero_red", "metadata_only")[index % 4]
        remaining = [f for f in fixtures if f["fixture_id"] not in used]
        targeted = [f for f in remaining if desired in tags(f)]
        if remaining:
            add(rng.choice(targeted or remaining), "targeted:" + desired if targeted else "random_extra")
    candidates = sorted([f for rows in grouped.values() for f in rows if f["fixture_id"] not in used], key=lambda f: f["fixture_id"])
    for desired in ("metadata_only", "zero_corners_or_sot", "positive_red", "unknown_red"):
        subset = [f for f in candidates if f["fixture_id"] not in used and desired in tags(f)]
        if subset and len(selected) < size:
            add(rng.choice(subset), "targeted:" + desired)
    rng.shuffle(candidates)
    for fixture in candidates:
        if fixture["fixture_id"] not in used and len(selected) < size:
            add(fixture, "random_extra")
    return {"seed": seed, "requested_size": size, "fixtures": selected,
            "note": "Stratified diagnostic sample, not population prevalence or proof that null means zero."}


def compare_fixture(fixture, statistics, events):
    """Compare raw provider evidence without inventing total card semantics."""
    expected_ids = {fixture["home_id"], fixture["away_id"]}
    ids = [r.get("team", {}).get("id") for r in statistics]
    issues = []
    stored = {r["api_team_id"]: r for r in fixture["team_rows"]}
    event_summary = {tid: Counter() for tid in expected_ids}
    ambiguous_events = []
    for event in events:
        if event.get("type") != "Card":
            continue
        tid = (event.get("team") or {}).get("id")
        detail = str(event.get("detail") or "unknown")
        if tid not in expected_ids:
            ambiguous_events.append({"issue": "card_event_unknown_team", "event": event})
            continue
        event_summary[tid][detail] += 1
        elapsed = numeric((event.get("time") or {}).get("elapsed"))
        if elapsed is None or elapsed > 90 or not (event.get("player") or {}).get("id"):
            ambiguous_events.append({"issue": "card_event_period_or_actor_unclear", "event": event})
    result = {"fixture_id": fixture["fixture_id"], "league": fixture["league"], "season": fixture["season"],
              "source": source_kind(fixture), "comparisons": [], "issues": issues,
              "provider_team_ids": ids, "valid_statistics_pair": len(ids) == 2 and set(ids) == expected_ids,
              "card_events": {str(tid): dict(counts) for tid, counts in event_summary.items()},
              "event_count": len(events), "ambiguous_card_events": ambiguous_events,
              "event_caveat": "No listed card event does NOT prove zero cards. Second-yellow, bench/staff and period rules are unresolved."}
    if not result["valid_statistics_pair"]:
        result["issue"] = "empty_provider_statistics" if not statistics else "conflicting_provider_team_pair"
        return result
    comparisons = []
    for team in statistics:
        tid = team["team"]["id"]
        entries = team.get("statistics") or []
        if len({s.get("type") for s in entries}) != len(entries):
            issues.append({"team_id": tid, "issue": "duplicate_statistic_type"})
            continue
        raw = {s.get("type"): s.get("value") for s in entries}
        values = stored[tid]["values"]
        for field, label in API_FIELDS.items():
            value = raw.get(label)
            number = numeric(value.rstrip("%")) / 100 if isinstance(value, str) and value.endswith("%") and numeric(value.rstrip("%")) is not None else numeric(value)
            observed_state = "absent" if label not in raw else state(number if number is not None else value, field)
            previous = values[field]
            if observed_state in ("absent", "unknown"):
                comparison = "still_unknown" if previous is None else "known_value_now_unavailable"
            elif observed_state == "invalid":
                comparison = "invalid_provider_value"
            elif previous is None:
                comparison = "recoverable_numeric_zero" if number == 0 else "recoverable_numeric_nonzero"
            elif abs(float(previous) - number) <= 0.0001:
                comparison = "agrees"
            else:
                comparison = "numeric_disagreement"
            comparisons.append({"team_id": tid, "field": field, "stored": previous, "provider_raw": value,
                                "provider_numeric": number, "provider_state": observed_state, "comparison": comparison})
    result["comparisons"] = comparisons
    return result
