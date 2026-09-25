"""Explicit, offline recovery plans for the pre-automation published cohort.

Only saved identities and checksum-verified provider responses are evidence.
Missing periods, versions, prices or bookmaker policies are never invented.
"""
from datetime import datetime, timedelta, timezone
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from ..publication_identity import digest
from ..settlement import Grade, SETTLEMENT_VERSION, grade_selection, parse_fixture_result, selection_issue, utc_datetime
from ..sync.apifootball import COMPETITIONS
from ..repositories.predictions import PredictionRepository


def row_token(row):
    # Include the saved decision, grade and previous evidence; ignore operational
    # timestamps unrelated to settlement (e.g. a prospective price capture).
    return digest({key: row.get(key) for key in (
        "id", "recommendation_id", "fixture_id", "league", "kickoff", "scheduled_kickoff",
        "market", "pick", "side", "line", "bookmaker", "odds", "tracking", "outcome", "graded_at", "settlement",
    )})


class SavedProviderEvidence:
    def __init__(self, directory=None):
        self.directory = Path(directory).resolve() if directory else None
        self.rows = {}
        if self.directory:
            for item in json.loads((self.directory / "requests.json").read_text()):
                if not item.get("success") or item.get("path") not in {"/fixtures", "/fixtures/statistics"}:
                    continue
                path = (self.directory / item["file"]).resolve()
                if path.parent != self.directory:
                    raise ValueError("Evidence file must be inside the approved directory")
                raw = path.read_bytes()
                if sha256(raw).hexdigest() != item["sha256"]:
                    raise ValueError("Provider evidence checksum mismatch")
                payload = json.loads(raw)
                if payload.get("errors") or not isinstance(payload.get("response"), list):
                    raise ValueError("Invalid saved provider response")
                self.rows[(item["path"], str(item["fixture_id"]))] = (payload["response"], item["sha256"])

    def fixture(self, fid):
        pair = self.rows.get(("/fixtures", str(fid)))
        if not pair:
            return None, None
        fixtures, fixture_hash = pair
        if len(fixtures) != 1 or str(fixtures[0]["fixture"]["id"]) != str(fid):
            raise ValueError("Saved fixture identity mismatch")
        statistics, statistics_hash = self.rows.get(("/fixtures/statistics", str(fid)), (None, None))
        return parse_fixture_result(fixtures[0], statistics), {
            "directory": str(self.directory), "fixture_sha256": fixture_hash,
            "statistics_sha256": statistics_hash,
        }


class HistoricalSettlementRecovery:
    def __init__(self, repo=None, *, evidence=None):
        self.repo = repo or PredictionRepository()
        self.evidence = evidence or SavedProviderEvidence()

    def candidates(self, cutoff):
        return [p for p in self.repo.settlement_candidates(include_graded=True, published_after_id=0)
                if p["recommendation_id"] <= cutoff]

    def assessment(self, pred, now):
        issue = selection_issue(pred)
        result, source = None, None
        if issue:
            grade = Grade(pending_reason=issue)
        elif pred["market"] == "cards":
            grade = Grade(pending_reason="card_definition_unverified")
        else:
            result, source = self.evidence.fixture(pred["fixture_id"])
            spec = COMPETITIONS.get(pred["league"])
            if result is None:
                grade = Grade(pending_reason="authoritative_result_not_archived")
            elif spec is None or str(result.get("league_id")) != str(spec.api_football_id):
                grade = Grade(pending_reason="competition_identity_mismatch")
            elif not utc_datetime(result.get("kickoff")) or utc_datetime(result["kickoff"]) > now:
                grade = Grade(pending_reason="fixture_not_due")
            else:
                grade = grade_selection(pred, result)
        evidence_hash = digest({"selection": (pred.get("tracking") or {}).get("selection"),
                                "result": result, "source": source, "grade": grade.to_dict(),
                                "version": SETTLEMENT_VERSION})
        return {**grade.to_dict(), "version": SETTLEMENT_VERSION, "evidence_hash": evidence_hash,
                "result": result, "source": source, "rule": None}

    def plan(self, *, cutoff, now=None):
        if cutoff < 0:
            raise ValueError("cutoff must be nonnegative")
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("Planning time must be timezone aware")
        entries, excluded, counts = [], [], Counter()
        for pred in self.candidates(cutoff):
            kickoff = utc_datetime(pred.get("scheduled_kickoff") or pred.get("kickoff"))
            if pred["outcome"] is not None:
                counts["already_graded"] += 1
                continue
            if kickoff is None or kickoff + timedelta(hours=3) > now:
                counts["not_due_or_unknown_kickoff"] += 1
                excluded.append({"prediction_id": pred["id"], "recommendation_id": pred["recommendation_id"],
                                 "fixture_id": pred["fixture_id"], "scheduled_kickoff": pred.get("scheduled_kickoff"),
                                 "reason": "not_due" if kickoff else "unknown_kickoff"})
                continue
            assessment = self.assessment(pred, now)
            action = "grade" if assessment["outcome"] else "record_pending"
            counts[action] += 1
            entries.append({"prediction_id": pred["id"], "recommendation_id": pred["recommendation_id"],
                            "fixture_id": pred["fixture_id"], "token": row_token(pred),
                            "action": action, "assessment": assessment})
        report = {"schema": "historical-settlement-plan.v1", "created_at": now.isoformat(),
                  "cutoff": cutoff, "evidence_directory": str(self.evidence.directory) if self.evidence.directory else None,
                  "api_calls": 0, "counts": dict(counts), "entries": entries, "excluded": excluded,
                  "pending_reasons": dict(Counter(e["assessment"]["pending_reason"] for e in entries if e["action"] == "record_pending"))}
        return {**report, "plan_id": digest(report)}

    def apply(self, plan, *, guard=lambda: None):
        """Apply only reviewed, unchanged rows; retries cannot rewrite grades."""
        plan_id = plan["plan_id"]
        if plan["schema"] != "historical-settlement-plan.v1" or digest({k: v for k, v in plan.items() if k != "plan_id"}) != plan_id:
            raise ValueError("Recovery plan integrity check failed")
        if (str(self.evidence.directory) if self.evidence.directory else None) != plan["evidence_directory"]:
            raise ValueError("Recovery evidence directory differs from the plan")
        rows = {p["id"]: p for p in self.candidates(plan["cutoff"])}
        checked, counts = [], Counter()
        for item in plan["entries"]:
            pred = rows.get(item["prediction_id"])
            if pred and (pred.get("settlement") or {}).get("recovery_plan_id") == plan_id:
                if pred["outcome"] != item["assessment"]["outcome"] or pred["settlement"].get("evidence_hash") != item["assessment"]["evidence_hash"]:
                    raise ValueError("Previously recovered outcome changed; review it separately")
                counts["unchanged"] += 1
                continue
            if pred is None or row_token(pred) != item["token"]:
                raise ValueError(f"Prediction {item['prediction_id']} changed; create a fresh plan")
            assessment = self.assessment(pred, utc_datetime(plan["created_at"]))
            if assessment != item["assessment"]:
                raise ValueError("Saved evidence changed; create a fresh recovery plan")
            checked.append((pred, assessment))
        # Validate the complete plan before the first write. A DB failure is
        # resumable using the same plan; no implicit correction/regrade path.
        for pred, assessment in checked:
            guard()
            status = self.repo.record_settlement(pred, {**assessment, "recovery_plan_id": plan_id},
                                                 checked_at=datetime.now(timezone.utc))
            counts[status] += 1
            if status == "applied":
                counts["graded" if assessment["outcome"] else "pending_recorded"] += 1
        return {"plan_id": plan_id, "counts": dict(counts), "api_calls": 0}
