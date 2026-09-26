"""Research-only indexing, feature mask and actual contributing support.

The numerical reference is the existing pure pre-match builder. Indexing only
narrows its input candidates; the builder still enforces all temporal rules.
No profile policy, live loader or active artifact is changed here.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import asdict

from Scripts.rag_ingest.core import model_features as reference

VERSION = "phase3-feature-view.v1"
MARKETS = ("goals", "corners", "sot")
MIN_SUPPORT = 5


class HistoryIndex:
    def __init__(self, history):
        self.by_competition = defaultdict(list)
        self.by_team = defaultdict(list)
        seen = set()
        for row in sorted(history, key=lambda r: (reference.utc(r["kickoff"]), r["fixture_id"])):
            if row["fixture_id"] in seen:
                raise ValueError("Duplicate history fixture")
            seen.add(row["fixture_id"])
            self.by_competition[(row["competition"], row["season"])].append(row)
            for side in ("home", "away"):
                self.by_team[(row[f"{side}_team_id"], row["season"])].append(row)
        self.times = {id(rows): [reference.utc(r["kickoff"]) for r in rows]
                      for rows in (*self.by_competition.values(), *self.by_team.values())}

    def candidates(self, fixture):
        selected = {}
        cutoff = reference.utc(fixture["kickoff"])
        for season in (fixture["season"] - 1, fixture["season"]):
            groups = [self.by_competition.get((fixture["competition"], season), [])]
            groups.extend(self.by_team.get((fixture[f"{side}_team_id"], season), [])
                          for side in ("home", "away"))
            for rows in groups:
                end = bisect_left(self.times.get(id(rows), []), cutoff)
                for row in rows[:end]:
                    selected[row["fixture_id"]] = row
        return sorted(selected.values(), key=lambda r: (r["kickoff"], r["fixture_id"]))


def feature_contract(names, *, core=reference):
    selected = [name for name in names if "card" not in name.lower()]
    return {"version": VERSION, "source_feature_version": core.SCHEMA_VERSION,
            "source_names": names, "names": selected,
            "excluded_names": [name for name in names if name not in selected],
            "policy": asdict(core.FeaturePolicy()),
            "profile_window": "reference_current_plus_previous_provider_season",
            "availability": "assumed_final", "minimum_contributing_observations": MIN_SUPPORT,
            "forecast_stage": "reconstructed_immediately_before_kickoff",
            "preprocessing": "unfitted; nulls retained; train-only weighted transforms belong to Batch B"}


def _component_ids(snapshot, team, audit, market, *, core=reference):
    """Count known own-production observations with strictly positive weight.

    The existing audit lists candidate prior rows even when prior_weight=0;
    those cannot qualify a profile with too few real current observations.
    """
    if audit is None:
        return set()
    season = snapshot["fixture"]["season"]
    rows = [r for r in snapshot["history"] if r["competition"] == audit["competition"]
            and team in (r["home_team_id"], r["away_team_id"])]
    current, prior = set(), set()
    for row in rows:
        side = "home" if row["home_team_id"] == team else "away"
        if core.number(row[side].get(market)) is None:
            continue
        if row["season"] == season:
            current.add(row["fixture_id"])
        elif row["season"] == season - 1:
            prior.add(row["fixture_id"])
    return current | (prior if audit["prior_weight"] > 0 else set())


def contributing_support(snapshot, features, *, core=reference):
    result = {}
    for market in MARKETS:
        result[market] = {}
        for side in ("home", "away"):
            team = snapshot["fixture"][f"{side}_team_id"]
            audit = features["profile_audit"][side]
            ids = _component_ids(snapshot, team, audit["primary"], market, core=core)
            primary_ids = set(ids)
            # In the existing builder a continental value is used only when
            # both components are known; it is not a fallback for null domestic.
            if audit["mode"] == "domestic_continental_blend":
                continental = _component_ids(snapshot, team, audit["continental"], market, core=core)
                weight = snapshot["policy"]["domestic_weight"]
                if ids and continental:
                    ids = (ids if weight > 0 else set()) | (continental if weight < 1 else set())
            rate = features["profiles"][side][core.RATE_FIELDS[market]]
            count = len(ids) if rate is not None else 0
            result[market][side] = {"count": count, "fixture_ids": sorted(ids) if count else [],
                                    "primary_fixture_ids": sorted(primary_ids),
                                    "band": "0" if count == 0 else "1-4" if count < 5 else "5-9" if count < 10 else "10+",
                                    "rate_available": rate is not None}
    return result


def build_reference(fixture, history, competitions, *, core=reference):
    candidates = history.candidates(fixture) if isinstance(history, HistoryIndex) else history
    snapshot = core.capture_snapshot(fixture, candidates, as_of=fixture["kickoff"],
                                     competitions=competitions, availability="assumed_final")
    features = core.build_features(snapshot)
    contract = feature_contract(features["names"], core=core)
    values = dict(zip(features["names"], features["values"]))
    return snapshot, features, {
        "values": [values[name] for name in contract["names"]],
        "support": contributing_support(snapshot, features, core=core),
        "feature_contract_id": core.digest(contract),
    }


def apply_support(decisions, support):
    """Attach predeclared feature support without overriding evidence reasons."""
    result = {}
    for market, decision in decisions.items():
        reasons = list(decision["reasons"])
        if market in MARKETS:
            for side in ("home", "away"):
                item = support[market][side]
                if not item["rate_available"]:
                    reasons.append(f"missing_{side}_production_rate")
                if item["count"] < MIN_SUPPORT:
                    reasons.append(f"insufficient_{side}_history")
        reasons = list(dict.fromkeys(reasons))
        result[market] = {**decision, "eligible": not reasons, "reasons": reasons,
                          "primary_reason": reasons[0] if reasons else None}
    return result
