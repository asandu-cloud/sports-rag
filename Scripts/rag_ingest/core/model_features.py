"""Versioned, network-free pre-match features for candidate training/inference.

No Chroma lookups, current aggregate profiles, saved final Elo, or model writes.
Both candidate paths must receive the same captured snapshot. Production's
legacy feature schema remains separate until a candidate is validated/promoted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from statistics import mean
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "prematch-features.v1"
SNAPSHOT_VERSION = "prematch-input.v1"
MARKETS = ("goals", "corners", "cards", "sot")
RATE_FIELDS = {"goals": "goals_for_pm", "corners": "corners_pm",
               "cards": "cards_per_90_team", "sot": "sot_for_pm"}
AGAINST_FIELDS = {"goals": "goals_against_pm", "corners": "corners_against_pm",
                  "cards": "opp_cards_induced_pm", "sot": "sot_against_pm"}


def utc(value: Any) -> datetime:
    result = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Feature timestamps must have an explicit timezone")
    return result.astimezone(timezone.utc)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


@dataclass(frozen=True)
class FeaturePolicy:
    # Match the current/prior profile policy; a third season is not pooled.
    prior_strength: float = 8.0
    recent_matches: int = 5
    domestic_weight: float = 0.8
    min_european_matches: int = 3
    # New, explicit candidate Elo policy: per competition, no cross-league
    # transfer of ratings; regress towards 1500 at provider-season boundaries.
    elo_start: float = 1500.0
    elo_k: float = 32.0
    elo_season_retention: float = 0.75
    # No final whistle timestamp exists in the canonical table. This is an
    # explicit conservative completion assumption, not provider observation.
    assumed_completion_hours: float = 3.0
    rest_days_cap: float = 14.0

    def __post_init__(self):
        if self.prior_strength <= 0 or self.recent_matches < 1 or self.min_european_matches < 1:
            raise ValueError("Invalid profile policy")
        if not 0 <= self.domestic_weight <= 1 or not 0 <= self.elo_season_retention <= 1:
            raise ValueError("Invalid blend/season regression weight")
        if self.assumed_completion_hours <= 0 or self.rest_days_cap <= 0 or self.elo_k <= 0:
            raise ValueError("Invalid time/Elo policy")


def fixture_identity(row: Mapping[str, Any], competitions: Mapping[str, str]) -> dict:
    """Explicit provider IDs/seasons only; never infer identity from names."""
    result = {key: row.get(key) for key in ("fixture_id", "competition", "season", "home_team_id", "away_team_id")}
    for key in ("fixture_id", "home_team_id", "away_team_id", "season"):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f"Missing/invalid provider {key}")
    if not 2000 <= result["season"] <= 2100:
        raise ValueError("Invalid provider season")
    if result["competition"] not in competitions:
        raise ValueError("Competition is not in the supported registry")
    if result["home_team_id"] == result["away_team_id"]:
        raise ValueError("A fixture must have two different teams")
    result["kickoff"] = utc(row.get("kickoff")).isoformat()
    return result


def capture_snapshot(fixture: Mapping[str, Any], history: Iterable[Mapping[str, Any]], *,
                     as_of: Any, competitions: Mapping[str, str],
                     availability: str = "observed", policy: FeaturePolicy | None = None) -> dict:
    """Capture only eligible inputs. ``assumed_final`` is labelled backfill.

    Historical imports cannot prove what was observed at a past cutoff. The
    default requires actual observation timestamps; retrospective evaluation
    must explicitly opt into the completion/availability assumption.
    """
    if availability not in {"observed", "assumed_final"}:
        raise ValueError("availability must be observed or assumed_final")
    policy = policy or FeaturePolicy()
    target = fixture_identity(fixture, competitions)
    cutoff = utc(as_of)
    if cutoff > utc(target["kickoff"]):
        raise ValueError("Pre-match as_of cannot be after kickoff")
    referee_observed = fixture.get("referee_observed_at")
    target["referee_observed_at"] = utc(referee_observed).isoformat() if referee_observed else None
    if availability == "assumed_final" or (referee_observed and utc(referee_observed) <= cutoff):
        target["referee"] = fixture.get("referee") or None
    else:
        target["referee"] = None
    selected = []
    ids = set()
    teams = {target["home_team_id"], target["away_team_id"]}
    for raw in history:
        if raw.get("fixture_id") == target["fixture_id"]:
            continue  # Target results are never a feature source.
        row = fixture_identity(raw, competitions)
        kickoff = utc(row["kickoff"])
        if kickoff + timedelta(hours=policy.assumed_completion_hours) >= cutoff or raw.get("status") != "FT":
            continue
        if row["season"] not in {target["season"], target["season"] - 1}:
            continue
        if row["competition"] != target["competition"] and not teams.intersection({row["home_team_id"], row["away_team_id"]}):
            continue
        available = raw.get("observed_at")
        if availability == "observed" and (not available or utc(available) > cutoff):
            continue
        if row["fixture_id"] in ids:
            raise ValueError(f"Ambiguous duplicate fixture {row['fixture_id']}")
        ids.add(row["fixture_id"])
        row.update(status="FT", observed_at=utc(available).isoformat() if available else None,
                   referee=raw.get("referee") or None)
        for side in ("home", "away"):
            stats = raw.get(side) or {}
            row[side] = {key: number(stats.get(key)) for key in (*MARKETS, "shots", "fouls", "xg", "possession")}
        selected.append(row)
    selected.sort(key=lambda row: (row["kickoff"], row["fixture_id"]))
    payload = {"snapshot_version": SNAPSHOT_VERSION, "feature_version": SCHEMA_VERSION,
               "fixture": target, "as_of": cutoff.isoformat(), "availability": availability,
               "policy": asdict(policy), "competitions": dict(sorted(competitions.items())), "history": selected}
    return {**payload, "snapshot_id": digest(payload)}


def _mean(values) -> float | None:
    known = [value for value in values if value is not None]
    return mean(known) if known else None


def _side(row: Mapping[str, Any], team: int) -> str:
    return "home" if row["home_team_id"] == team else "away"


def _aggregates(rows: list[dict], team: int, venue: str, policy: FeaturePolicy) -> dict:
    result = {}
    for market in MARKETS:
        result[RATE_FIELDS[market]] = _mean(row[_side(row, team)][market] for row in rows)
        result[AGAINST_FIELDS[market]] = _mean(row["away" if _side(row, team) == "home" else "home"][market] for row in rows)
        result[f"{market}_venue_pm"] = _mean(row[venue][market] for row in rows if _side(row, team) == venue)
        result[f"{market}_last_{policy.recent_matches}"] = _mean(row[_side(row, team)][market] for row in rows[-policy.recent_matches:])
    for stat in ("shots", "fouls", "xg", "possession"):
        result[f"{stat}_pm"] = _mean(row[_side(row, team)][stat] for row in rows)
    return result


def _season_profile(history: list[dict], team: int, competition: str, season: int,
                    venue: str, policy: FeaturePolicy) -> tuple[dict, dict]:
    rows = [r for r in history if r["competition"] == competition and team in {r["home_team_id"], r["away_team_id"]}]
    current = [r for r in rows if r["season"] == season]
    prior = [r for r in rows if r["season"] == season - 1]
    current_values, prior_values = _aggregates(current, team, venue, policy), _aggregates(prior, team, venue, policy)
    n = len(current)
    prior_weight = (1.0 if n == 0 else policy.prior_strength / (n + policy.prior_strength)) if prior and n < policy.prior_strength else 0.0
    values = {}
    for key, value in current_values.items():
        previous = prior_values[key]
        values[key] = ((1 - prior_weight) * value + prior_weight * previous
                       if value is not None and previous is not None and prior_weight
                       else (previous if value is None and prior_weight else value))
    return values, {"competition": competition, "current_matches": n, "prior_matches": len(prior),
                    "prior_weight": prior_weight, "source_fixture_ids": [r["fixture_id"] for r in rows]}


def _team_profile(snapshot: dict, team: int, venue: str, policy: FeaturePolicy) -> tuple[dict, dict]:
    fixture, history, competitions = snapshot["fixture"], snapshot["history"], snapshot["competitions"]
    league, season = fixture["competition"], fixture["season"]
    values, audit = _season_profile(history, team, league, season, venue, policy)
    audit = {"mode": "competition_only", "primary": audit, "continental": None, "domestic_candidates": []}
    if competitions[league] != "continental_cup":
        return values, audit
    domestic = [r for r in history if competitions[r["competition"]] == "domestic_league"
                and team in {r["home_team_id"], r["away_team_id"]}]
    latest_season = max((r["season"] for r in domestic), default=None)
    candidates = sorted({r["competition"] for r in domestic if r["season"] == latest_season})
    audit["domestic_candidates"] = candidates
    if candidates and latest_season != season:
        # Last season's league does not prove membership after promotion or
        # relegation. Keep the continental fallback until current membership
        # is evidenced, rather than transferring an unadjusted domestic prior.
        audit["mode"] = "unknown_current_domestic"
        return values, audit
    if len(candidates) != 1:
        audit["mode"] = "ambiguous_domestic" if candidates else "continental_only"
        return values, audit
    domestic_values, domestic_audit = _season_profile(history, team, candidates[0], season, venue, policy)
    continental_audit = audit["primary"]
    audit.update(primary=domestic_audit, continental=continental_audit, mode="domestic_anchor")
    if continental_audit["current_matches"] >= policy.min_european_matches:
        audit["mode"] = "domestic_continental_blend"
        for key, value in domestic_values.items():
            european = values[key]
            if value is not None and european is not None:
                domestic_values[key] = policy.domestic_weight * value + (1 - policy.domestic_weight) * european
    return domestic_values, audit


def _elo(snapshot: dict, policy: FeaturePolicy) -> dict:
    target = snapshot["fixture"]
    ratings, last_seasons = {}, {}

    def rating(team, market, season):
        key = (team, market)
        value = ratings.get(key, policy.elo_start)
        elapsed = max(0, season - last_seasons.get(key, season))
        return policy.elo_start + (value - policy.elo_start) * policy.elo_season_retention ** elapsed

    for row in snapshot["history"]:
        if row["competition"] != target["competition"]:
            continue
        home, away, season = row["home_team_id"], row["away_team_id"], row["season"]
        for market in MARKETS:
            h, a = row["home"][market], row["away"][market]
            if h is None or a is None:
                continue
            hr, ar = rating(home, market, season), rating(away, market, season)
            observed = 1.0 if h > a else (0.5 if h == a else 0.0)
            delta = policy.elo_k * (observed - 1 / (1 + 10 ** ((ar - hr) / 400)))
            ratings[(home, market)], ratings[(away, market)] = hr + delta, ar - delta
            last_seasons[(home, market)] = last_seasons[(away, market)] = season
    return {f"{side}_elo_{market}": rating(target[f"{side}_team_id"], market, target["season"])
            for side in ("home", "away") for market in MARKETS}


def build_features(snapshot: Mapping[str, Any]) -> dict:
    """Single pure vector builder for historical and candidate live inference."""
    snapshot = dict(snapshot)
    expected = digest({key: value for key, value in snapshot.items() if key != "snapshot_id"})
    if snapshot.get("snapshot_id") != expected:
        raise ValueError("Input snapshot digest mismatch")
    if snapshot.get("snapshot_version") != SNAPSHOT_VERSION or snapshot.get("feature_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported input/feature schema")
    policy = FeaturePolicy(**snapshot["policy"])
    # Validate temporal rules again during replay (not merely at capture).
    rebuilt = capture_snapshot(snapshot["fixture"], snapshot["history"], as_of=snapshot["as_of"],
                               competitions=snapshot["competitions"], availability=snapshot["availability"], policy=policy)
    if rebuilt["history"] != snapshot["history"] or rebuilt["fixture"] != snapshot["fixture"]:
        raise ValueError("Snapshot contains ineligible historical inputs")
    target, history = snapshot["fixture"], snapshot["history"]
    values, profiles, audits = {}, {}, {}
    for side in ("home", "away"):
        team = target[f"{side}_team_id"]
        profile, audit = _team_profile(snapshot, team, side, policy)
        profiles[side], audits[side] = profile, audit
        values.update({f"{side}_{key}": value for key, value in profile.items()})
        for key in ("current_matches", "prior_matches", "prior_weight"):
            values[f"{side}_{key}"] = audit["primary"][key]
        values[f"{side}_continental_matches"] = (audit["continental"] or {}).get("current_matches", 0)
        played = [utc(r["kickoff"]) for r in history if team in {r["home_team_id"], r["away_team_id"]}]
        values[f"{side}_rest_days"] = min(policy.rest_days_cap, (utc(target["kickoff"]) - max(played)).total_seconds() / 86400) if played else None
    values.update(_elo(snapshot, policy))
    # No today's referee aggregates: same competition, past FT fixtures only.
    referee_rows = [r for r in history if target.get("referee") and r.get("referee") == target["referee"]
                    and r["competition"] == target["competition"]]
    totals = lambda stat: [r["home"][stat] + r["away"][stat] for r in referee_rows
                            if r["home"][stat] is not None and r["away"][stat] is not None]
    values["referee_cards_pm"] = _mean(totals("cards"))
    values["referee_fouls_pm"] = _mean(totals("fouls"))
    values["referee_matches"] = len(referee_rows)
    for competition in sorted(snapshot["competitions"]):
        values[f"competition_{competition}"] = float(competition == target["competition"])
    # Null stays null in the durable record. Preprocessing may impute only
    # after fitting on a training partition; real zero remains observed zero.
    numeric_names = sorted(values)
    names = numeric_names + [f"{name}__missing" for name in numeric_names]
    ordered = [values[name] for name in numeric_names] + [float(values[name] is None) for name in numeric_names]
    return {"feature_version": SCHEMA_VERSION, "snapshot_id": snapshot["snapshot_id"],
            "names": names, "values": ordered, "profiles": profiles, "profile_audit": audits,
            "provenance": {"fixture": target, "as_of": snapshot["as_of"], "availability": snapshot["availability"],
                           "policy": snapshot["policy"], "source_fixture_ids": [r["fixture_id"] for r in history]}}


def training_features(snapshot: Mapping[str, Any]) -> dict:
    return build_features(snapshot)


def inference_features(snapshot: Mapping[str, Any]) -> dict:
    return build_features(snapshot)
