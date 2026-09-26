"""Offline, development-only Phase 3 statistical comparators.

The arithmetic below reconstructs the production *core*, not a historically
published prediction. It never imports production projection wrappers or opens
profiles, databases, provider clients, audit stores or lockboxes. The caller of
``build_baseline_rows`` owns the separately audited preparation read and the
immutable sidecar write. Normal benchmark fitting consumes that sidecar only.
"""
from __future__ import annotations

import ast
from bisect import bisect_left
from collections import defaultdict
from datetime import timedelta
import hashlib
import math
from pathlib import Path
from statistics import mean

from Scripts.rag_ingest.core import model_features as core
from Scripts.data_platform.features.phase3_features import HistoryIndex

VERSION = "phase3-baselines.v1"
STATISTICAL_VERSION = "statistical-core-reconstruction.v1"
AVERAGE_VERSION = "league-venue-average.v1"
MARKETS = ("goals", "corners", "sot")
MIN_LEAGUE_FIXTURES = 5
OMITTED_CONTEXT = (
    "lineup_and_player_adjustments", "standings_and_match_stakes",
    "knockout_context", "other_live_context", "machine_learning_contribution",
)
_SOURCE_PATHS = (
    "Scripts/rag_ingest/core/projections.py",
    "Scripts/rag_ingest/core/team_resolution.py",
    "Scripts/rag_ingest/core/weights.py",
    "Scripts/rag_ingest/core/model_features.py",
    "Scripts/data_platform/features/phase3_features.py",
    "Scripts/data_platform/features/benchmarks/baselines.py",
)


def _literal_assignment(path, name):
    """Read a constant without executing the module or any of its imports."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing frozen configuration constant {name}")


def baseline_metadata(repo_root: Path) -> dict:
    """Capture source/configuration identities before sidecar preparation."""
    root = Path(repo_root)
    weights = _literal_assignment(root / _SOURCE_PATHS[2], "SCORING_WEIGHTS")
    scoring = {key: weights[key] for key in ("projection", "projection_sot", "recency", "european")}
    scoring["prior_strength"] = _literal_assignment(
        root / _SOURCE_PATHS[1], "EARLY_SEASON_PRIOR_STRENGTH")
    _validate_scoring(scoring)
    return {
        "version": VERSION, "statistical_version": STATISTICAL_VERSION,
        "average_version": AVERAGE_VERSION, "scoring": scoring,
        "scoring_sha256": core.digest(scoring),
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in _SOURCE_PATHS},
        "scope": "development_preparation_only",
        "ml_weight": 0.0, "omitted_context": list(OMITTED_CONTEXT),
        "profile_policy": "current_and_immediately_previous_provider_season",
        "domestic_identity": "unique_current_season_domestic_competition_by_provider_team_id",
        "recent_policy": "six_latest_completed_fixtures_per_competition_exponential_weights",
        "average_policy": {
            "window": "current_and_immediately_previous_provider_season",
            "minimum_league_complete_fixtures": MIN_LEAGUE_FIXTURES,
            "fallback": "all_supported_competitions_same_seasons_complete_two_team_labels",
            "pooled_minimum": 1,
            "no_support": "unavailable_not_zero",
        },
        "historical_limitation": "assumed_final is retrospective final-statistics reconstruction",
        "production_differences": [
            "exact_provider_ids_instead_of_name_or_current_Chroma_resolution",
            "strict_pre_forecast_completion_availability_instead_of_calendar_day_profile_lookup",
            "immediately_previous_provider_season_only_no_older_profile_fallback",
            "safe_frozen_history_only_missing_evidence_remains_missing",
        ],
    }


def _validate_scoring(scoring):
    for section in ("projection", "projection_sot", "recency", "european"):
        if not isinstance(scoring.get(section), dict):
            raise ValueError(f"Missing frozen scoring section {section}")
    if scoring.get("prior_strength", 0) <= 0:
        raise ValueError("Invalid frozen prior strength")
    alpha = scoring["recency"].get("alpha", .85)
    if not 0 < alpha <= 1:
        raise ValueError("Invalid recent exponential weight")
    euro = scoring["european"]
    if (not 0 <= euro["domestic_weight"] <= 1 or not 0 <= euro["euro_weight"] <= 1
            or not math.isclose(euro["domestic_weight"] + euro["euro_weight"], 1)):
        raise ValueError("Invalid European weights")


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values):
    known = [v for v in values if v is not None]
    return mean(known) if known else None


def _weighted_avg(values, alpha):
    known = [(i, x) for i, x in enumerate(values) if x is not None]
    denominator = sum(alpha ** i for i, _ in known)
    return sum(alpha ** i * x for i, x in known) / denominator if denominator else None


def _slope(values):
    known = [(i, x) for i, x in enumerate(values) if x is not None]
    if len(known) < 4:
        return None
    xs = [float(len(known) - 1 - i) for i, _ in known]
    ys = [x for _, x in known]
    x_mean, y_mean = mean(xs), mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator if denominator else 0.0


def _blend(own, opponent, own_weight, opponent_weight):
    if own is not None and opponent is not None:
        return own_weight * own + opponent_weight * opponent
    return own if own is not None else opponent


def _venue(venue, overall, weight):
    return _blend(venue, overall, weight, 1 - weight)


def statistical_projection(home_profile, away_profile, home_recent, away_recent, market, *, scoring):
    """Production core total arithmetic with explicit profiles/config and ML=0.

    Deliberately preserves the core's recent defensive fallbacks, goals xG
    rules, divergence blend and trend arithmetic, including present-nulls.
    """
    if market not in MARKETS:
        raise ValueError("Unsupported baseline market")
    pw = scoring["projection"]
    profiles = (home_profile, away_profile)
    rate = {"goals": "goals_for_pm", "corners": "corners_pm", "sot": "sot_for_pm"}[market]
    attacks = []
    for profile, venue in zip(profiles, ("home", "away")):
        attack = _number(profile.get(rate))
        if market == "goals":
            xg_h, xg_a = _number(profile.get("xg_home_pm")), _number(profile.get("xg_away_pm"))
            if attack is not None and xg_h is not None and xg_a is not None:
                xg_w = pw.get("xg_blend", .6)
                attack = xg_w * ((xg_h + xg_a) / 2) + (1 - xg_w) * attack
        attacks.append(_venue(_number(profile.get(f"{market}_{venue}_pm")), attack,
                              pw.get(f"{market}_venue_blend", 0 if market == "corners" else .5)))
    own_w, opp_w = ((scoring["projection_sot"]["own"], scoring["projection_sot"]["opp"])
                    if market == "sot" else (pw.get(f"{market}_own", .6), pw.get(f"{market}_opp", .4)))
    h_proj = _blend(attacks[0], _number(away_profile.get(f"{market}_against_pm")), own_w, opp_w)
    a_proj = _blend(attacks[1], _number(home_profile.get(f"{market}_against_pm")), own_w, opp_w)
    if market == "goals" and pw.get("home_advantage", 0) > 0:
        if h_proj is not None:
            h_proj += pw["home_advantage"]
        if a_proj is not None:
            a_proj = max(.1, a_proj - pw["home_advantage"])
    season = h_proj + a_proj if h_proj is not None and a_proj is not None else None
    if market == "goals":
        h_recent, a_recent = home_recent.get("xg_for_avg"), away_recent.get("xg_for_avg")
        recent = h_recent + a_recent if h_recent is not None and a_recent is not None else None
    else:
        h_recent, a_recent = _number(home_recent.get(f"{market}_for_avg")), _number(away_recent.get(f"{market}_for_avg"))
        recent = None
        if h_recent is not None and a_recent is not None:
            h_opp = _number(away_recent.get(f"{market}_against_avg"))
            a_opp = _number(home_recent.get(f"{market}_against_avg"))
            h_opp = h_opp if h_opp is not None else a_proj if a_proj is not None else a_recent
            a_opp = a_opp if a_opp is not None else h_proj if h_proj is not None else h_recent
            recent = (.6 * h_recent + .4 * h_opp) + (.6 * a_recent + .4 * a_opp)
    if season is not None and recent is not None:
        sw, rw = ((pw[f"blend_{market}_season"], pw[f"blend_{market}_recent"])
                  if f"blend_{market}_season" in pw and f"blend_{market}_recent" in pw
                  else (pw["blend_season"], pw["blend_recent"]))
        divergence = abs(recent - season) / max(season, 1.0)
        shift = min(.3, (divergence - .2) * 1.0) if divergence > .2 else 0
        value = (sw - shift) * season + (rw + shift) * recent
    else:
        value = season if season is not None else recent
    if value is not None:
        key = "goals_slope" if market == "goals" else f"{market}_for_slope"
        weight = scoring["recency"].get("trend_weight", 0)
        value += weight * sum(r[key] for r in (home_recent, away_recent) if r.get(key) is not None)
    return {"value": value, "season_total": season, "recent_total": recent,
            "home_projection": h_proj, "away_projection": a_proj}


def _side(row, team):
    return "home" if row["home_team_id"] == team else "away"


def _aggregate(rows, team):
    result = {"matches_played": len(rows)}
    for market in MARKETS:
        rate = {"goals": "goals_for_pm", "corners": "corners_pm", "sot": "sot_for_pm"}[market]
        result[rate] = _mean(row[_side(row, team)].get(market) for row in rows)
        result[f"{market}_against_pm"] = _mean(
            row["away" if _side(row, team) == "home" else "home"].get(market) for row in rows)
        for venue in ("home", "away"):
            result[f"{market}_{venue}_pm"] = _mean(row[venue].get(market) for row in rows if _side(row, team) == venue)
    for venue in ("home", "away"):
        result[f"xg_{venue}_pm"] = _mean(row[venue].get("xg") for row in rows if _side(row, team) == venue)
    return result


def _season_profile(history, team, competition, season, strength):
    rows = [r for r in history if r["competition"] == competition and team in (r["home_team_id"], r["away_team_id"])]
    current = [r for r in rows if r["season"] == season]
    prior = [r for r in rows if r["season"] == season - 1]
    current_values, previous = _aggregate(current, team), _aggregate(prior, team)
    weight = (1.0 if not current else strength / (len(current) + strength)) if prior and len(current) < strength else 0.0
    result = {}
    for key, value in current_values.items():
        old = previous[key]
        result[key] = ((1 - weight) * value + weight * old if value is not None and old is not None and weight
                       else old if value is None and weight else value)
    result["matches_played"] = len(current)
    return result, {"competition": competition, "current_matches": len(current), "prior_matches": len(prior),
                    "prior_weight": weight, "source_fixture_ids": [r["fixture_id"] for r in rows]}


def _recent(history, team, competition, alpha):
    rows = sorted((r for r in history if r["competition"] == competition
                   and team in (r["home_team_id"], r["away_team_id"])),
                  key=lambda r: (r["kickoff"], r["fixture_id"]), reverse=True)[:6]
    result = {"n": len(rows)}
    for market in MARKETS:
        # The goals recent component is xG, not actual goals, in production.
        stat = "xg" if market == "goals" else market
        own = [r[_side(r, team)].get(stat) for r in rows]
        key = "xg_for_avg" if market == "goals" else f"{market}_for_avg"
        result[key] = _weighted_avg(own, alpha)
        result["goals_slope" if market == "goals" else f"{market}_for_slope"] = _slope(own)
        if market != "goals":
            against = [r["away" if _side(r, team) == "home" else "home"].get(stat) for r in rows]
            result[f"{market}_against_avg"] = _weighted_avg(against, alpha)
    return result, [r["fixture_id"] for r in rows]


def reconstruct_snapshot(snapshot, *, scoring):
    """Build production-shaped, exact-ID profiles/recent inputs from a snapshot."""
    _validate_scoring(scoring)
    fixture, history, competitions = snapshot["fixture"], snapshot["history"], snapshot["competitions"]
    season, league = fixture["season"], fixture["competition"]
    profiles, recent, evidence = {}, {}, {}
    ew, alpha = scoring["european"], scoring["recency"].get("alpha", .85)
    for side in ("home", "away"):
        team = fixture[f"{side}_team_id"]
        profile, audit = _season_profile(history, team, league, season, scoring["prior_strength"])
        stats, recent_ids = _recent(history, team, league, alpha)
        item = {"team_id": team, "mode": "competition_only", "primary": audit,
                "continental": None, "recent_sources": {league: recent_ids}}
        if competitions[league] == "continental_cup":
            domestic = [r for r in history if competitions[r["competition"]] == "domestic_league"
                        and team in (r["home_team_id"], r["away_team_id"])]
            latest = max((r["season"] for r in domestic), default=None)
            candidates = sorted({r["competition"] for r in domestic if r["season"] == latest})
            item["domestic_candidates"] = candidates
            if latest != season or len(candidates) != 1:
                item["mode"] = ("unknown_current_domestic" if candidates and latest != season
                                else "ambiguous_domestic" if candidates else "continental_only")
            else:
                d_profile, d_audit = _season_profile(history, team, candidates[0], season, scoring["prior_strength"])
                item.update(mode="domestic_anchor", primary=d_audit, continental=audit)
                if audit["current_matches"] >= ew["min_euro_fixtures"]:
                    item["mode"] = "domestic_continental_blend"
                    for key, d_value in list(d_profile.items()):
                        e_value = profile.get(key)
                        if d_value is not None and e_value is not None:
                            d_profile[key] = ew["domestic_weight"] * d_value + ew["euro_weight"] * e_value
                profile = d_profile
                # Production blends recent competitions even before its
                # three-European-fixture profile gate, with missing fallbacks.
                d_recent, d_ids = _recent(history, team, candidates[0], alpha)
                item["recent_sources"][candidates[0]] = d_ids
                if d_recent["n"] and stats["n"]:
                    stats = {key: _blend(d_recent.get(key), stats.get(key), ew["domestic_weight"], ew["euro_weight"])
                             for key in set(d_recent) | set(stats)} | {"n": d_recent["n"] + stats["n"]}
                elif d_recent["n"]:
                    stats = d_recent
        profiles[side], recent[side], evidence[side] = profile, stats, item
    return {"profiles": profiles, "recent": recent, "evidence": evidence}


def _available(row, availability):
    assumed = core.utc(row["kickoff"]) + timedelta(hours=3)
    if availability == "assumed_final":
        return assumed
    observed = row.get("observed_at")
    return max(assumed, core.utc(observed)) if observed else None


class _AverageIndex:
    """Prefix sufficient statistics avoid a pooled-history scan per fixture."""
    def __init__(self, history, availability):
        buckets = defaultdict(list)
        for row in history:
            available = _available(row, availability)
            if available is None:
                continue
            for market in MARKETS:
                home, away = core.number(row["home"].get(market)), core.number(row["away"].get(market))
                if home is None or away is None:
                    continue
                for league in (row["competition"], None):
                    buckets[(league, row["season"], market)].append((available, row["fixture_id"], home, away))
        self.groups = {}
        for key, values in buckets.items():
            values.sort()
            home, away = [0.0], [0.0]
            for _, _, h, a in values:
                home.append(home[-1] + h)
                away.append(away[-1] + a)
            self.groups[key] = ([v[0] for v in values], home, away)

    def _summary(self, league, season, market, cutoff):
        count, home, away = 0, 0.0, 0.0
        for provider_season in (season - 1, season):
            dates, homes, aways = self.groups.get((league, provider_season, market), ([], [0.0], [0.0]))
            end = bisect_left(dates, cutoff)
            count += end
            home, away = home + homes[end], away + aways[end]
        return count, home, away

    def predict(self, fixture, market, cutoff):
        count, home, away = self._summary(fixture["competition"], fixture["season"], market, cutoff)
        league_count, scope = count, "competition"
        if count < MIN_LEAGUE_FIXTURES:
            count, home, away = self._summary(None, fixture["season"], market, cutoff)
            scope = "pooled" if count else "unavailable"
        return {"value": (home / count + away / count) if count else None,
                "scope": scope, "complete_fixtures": count, "league_complete_fixtures": league_count,
                "home_mean": home / count if count else None, "away_mean": away / count if count else None}


def build_baseline_rows(development_rows, history, competitions, *, confirmation_start, scoring):
    """Prepare comparator rows without accessing paths or held-out forecasts.

    ``history`` must be the evidence-filtered frozen input history. All targets
    must be initial-training/development forecasts strictly before confirmation;
    later history is discarded before statistical values are inspected. Snapshot
    hashes must replay exactly. Ineligible markets are not sidecar targets.
    """
    _validate_scoring(scoring)
    boundary = core.utc(confirmation_start)
    requested = list(development_rows)
    modes, seen = set(), set()
    for row in requested:
        fixture = row["fixture"]
        core.fixture_identity(fixture, competitions)
        cutoff = core.utc(row["as_of"])
        if (core.utc(fixture["kickoff"]) >= boundary or cutoff >= boundary
                or row.get("partition") not in {"initial_training", "development"}):
            raise ValueError("Baseline preparation refuses held-out forecasts")
        if cutoff > core.utc(fixture["kickoff"]):
            raise ValueError("Baseline forecast cutoff is after kickoff")
        if not row.get("snapshot_id") or not row.get("feature_contract_id"):
            raise ValueError("Baseline preparation requires frozen row identities")
        key = (fixture["fixture_id"], row["snapshot_id"], row["feature_contract_id"])
        if key in seen:
            raise ValueError("Duplicate baseline forecast identity")
        seen.add(key)
        modes.add(row["availability"])
    if not requested:
        return []
    if len(modes) != 1 or not modes.issubset({"assumed_final", "observed"}):
        raise ValueError("Baseline preparation cannot mix availability modes")
    mode = next(iter(modes))
    safe = [r for r in history if r.get("status") == "FT"
            and (available := _available(r, mode)) is not None and available < boundary]
    index, average = HistoryIndex(safe), _AverageIndex(safe, mode)
    output = []
    scoring_hash = core.digest(scoring)
    for row in sorted(requested, key=lambda r: (core.utc(r["as_of"]), r["fixture"]["fixture_id"], r["snapshot_id"])):
        eligible = [m for m in MARKETS if row.get("market_eligibility", {}).get(m, {}).get("eligible") is True]
        if not eligible:
            continue
        fixture, cutoff = row["fixture"], core.utc(row["as_of"])
        candidates = [r for r in index.candidates(fixture) if _available(r, mode) < cutoff]
        snapshot = core.capture_snapshot(fixture, candidates, as_of=cutoff,
                                         competitions=competitions, availability=mode)
        if snapshot["snapshot_id"] != row["snapshot_id"]:
            raise ValueError(f"Baseline snapshot replay mismatch for fixture {fixture['fixture_id']}")
        inputs = reconstruct_snapshot(snapshot, scoring=scoring)
        for market in eligible:
            league = average.predict(fixture, market, cutoff)
            stat = statistical_projection(inputs["profiles"]["home"], inputs["profiles"]["away"],
                                         inputs["recent"]["home"], inputs["recent"]["away"], market, scoring=scoring)
            output.append({
                "fixture_id": fixture["fixture_id"], "snapshot_id": row["snapshot_id"],
                "feature_contract_id": row["feature_contract_id"], "market": market,
                "league_average": league["value"], "statistical": stat["value"],
                "evidence": {"as_of": cutoff.isoformat(), "availability": mode,
                             "baseline_version": VERSION, "scoring_sha256": scoring_hash,
                             "league_average": league, "statistical": stat,
                             "profile_audit": inputs["evidence"],
                             "inputs_sha256": core.digest(inputs),
                             "history_sha256": core.digest(snapshot["history"]),
                             "omitted_context": list(OMITTED_CONTEXT), "ml_weight": 0.0,
                             "unavailable_reasons": (["league_average_no_prior_complete_labels"] if league["value"] is None else [])
                             + (["statistical_core_insufficient_inputs"] if stat["value"] is None else [])},
            })
    return output
