"""Independent, bounded prospective settlement and closing-price worker."""
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from ..publication_identity import digest
from ..repositories.match_read_cycle import MatchReadCycleRepository
from ..repositories.predictions import PredictionRepository
from ..settlement import Grade, SETTLEMENT_VERSION, selection_issue, utc_datetime
from ..settlement_policy import product_policy
from .closing_prices import ClosingPriceService, WINDOW_MINUTES
from .measurement_runtime import BoundedFootballClient, BudgetExceeded, CONTROL, MeasurementRuntime, defer, utc
from .settlement import FixtureResultLoader, SettlementService

LEASE = "prediction-measurement.v1"
PERMANENT = {"card_definition_unverified", "bookmaker_status_rule_required",
             "missing_or_unsupported_period", "missing_or_unsupported_market",
             "missing_or_conflicting_selection_identity", "conflicting_selection_identity",
             "unsupported_participant_market", "missing_bookmaker", "missing_fixture_id",
             "malformed_settlement_input", "regulation_statistics_unavailable",
             "regulation_player_statistics_unavailable"}


class MeasurementCycle:
    def __init__(self, repo=None, runtime=None, *, client=None, clock=None):
        self.repo = repo or PredictionRepository()
        self.runtime = runtime or MeasurementRuntime(self.repo._factory)
        self.leases = MatchReadCycleRepository(self.repo._factory)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.client = client

    def run_once(self, *, dry_run=False, kinds=("closing", "settlement")):
        control = self.runtime.state(CONTROL)
        if not control.get("enabled"):
            return {"status": "disabled", "errors": 0, "provider_attempts": 0}
        now, owner = self.clock(), uuid4().hex
        after = int(control["after_recommendation_id"])
        candidates = self.repo.settlement_candidates(include_graded=True, published_after_id=after)
        report = {"status": "completed", "errors": 0, "provider_attempts": 0,
                  "prospective_after_id": after, "candidates": len(candidates), "details": [],
                  "dry_run": dry_run, "held_reasons": {}, "overdue_pending": 0}
        if dry_run:
            # This command's dry run is OFFLINE, unlike the legacy resolver's.
            report["status"] = "planned"
            return report
        if not self.leases.acquire_lease(lease_key=LEASE, owner_id=owner, ttl_seconds=120, now=now):
            return {**report, "status": "overlap_skipped"}
        run_id = self.runtime.start_run(now)
        deadline = now + timedelta(seconds=50)

        def guard():
            instant = self.clock()
            if instant >= deadline or not self.leases.owns_lease(lease_key=LEASE, owner_id=owner, now=instant):
                raise BudgetExceeded("Measurement deadline/lease expired; retry next tick")

        try:
            if not candidates:
                return report
            from .. import config
            client = self.client or BoundedFootballClient(self.runtime,
                cycle_limit=min(20, int(control.get("cycle_request_limit", 20))),
                daily_limit=min(500, int(control.get("daily_request_limit", 500))),
                guard=guard, clock=self.clock, settings=config.SETTINGS)
            groups = defaultdict(list)
            for pred in candidates:
                groups[str(pred.get("fixture_id"))].append(pred)
                kickoff = utc_datetime(pred.get("scheduled_kickoff") or pred.get("kickoff"))
                if pred["outcome"] is None and kickoff and now - kickoff > timedelta(hours=24):
                    report["overdue_pending"] += 1
            # Closing work is time-critical. Settlement has its own per-cycle
            # fixture cap; retries are ordered by oldest last-check, not market.
            def sort_key(item):
                fid, preds = item
                kickoff = utc_datetime(preds[0].get("scheduled_kickoff") or preds[0].get("kickoff"))
                return kickoff or datetime.max.replace(tzinfo=timezone.utc)
            for kind in kinds:
                checked = 0
                items = sorted(groups.items(), key=sort_key) if kind == "closing" else sorted(
                    groups.items(), key=lambda item: self.runtime.state(f"measurement:settlement:{item[0]}").get("last_checked_at", ""))
                for fid, preds in items:
                    guard()
                    if checked >= 10:
                        break
                    kickoff = utc_datetime(preds[0].get("scheduled_kickoff") or preds[0].get("kickoff"))
                    if kickoff is None:
                        report["held_reasons"]["missing_fixture_kickoff"] = report["held_reasons"].get("missing_fixture_kickoff", 0) + len(preds)
                        continue
                    if kind == "settlement":
                        preds = [p for p in preds if p["outcome"] is None]
                        if now < kickoff + timedelta(hours=3):
                            continue
                    else:
                        preds = [p for p in preds if (p.get("closing_capture") or {}).get("status") != "final"]
                        if now < kickoff - timedelta(minutes=WINDOW_MINUTES):
                            continue
                    if not preds:
                        continue
                    revision = digest({"ids": [p["id"] for p in preds], "kickoff": kickoff.isoformat(),
                                       "rules": [p.get("settlement_policy") for p in preds]})
                    state = self.runtime.state(f"measurement:{kind}:{fid}")
                    if state.get("revision") == revision and (state.get("permanent") or
                            (utc_datetime(state.get("next_attempt_at")) or now) > now):
                        if state.get("permanent") and kind == "settlement":
                            reason = state["reason"]
                            report["held_reasons"][reason] = report["held_reasons"].get(reason, 0) + len(preds)
                        continue
                    eligible, blocked = [], Counter()
                    for pred in preds:
                        try:
                            issue = selection_issue(pred)
                        except Exception:
                            issue = "malformed_settlement_input"
                        if kind == "settlement" and pred["market"] == "cards" and not product_policy(pred):
                            issue = issue or "card_definition_unverified"
                        if issue:
                            blocked[issue] += 1
                            if kind == "settlement":
                                guard()
                                self.repo.record_settlement(pred, {**Grade(pending_reason=issue).to_dict(),
                                    "version": SETTLEMENT_VERSION, "evidence_hash": digest({"reason": issue}),
                                    "result": None, "rule": None}, checked_at=self.clock())
                        else:
                            eligible.append(pred)
                    for reason, count in blocked.items():
                        report["held_reasons"][reason] = report["held_reasons"].get(reason, 0) + count
                    if not eligible:
                        guard()
                        defer(self.runtime, kind, fid, now=self.clock(), reason=next(iter(blocked)),
                              revision=revision, attempts=0, permanent=True)
                        continue
                    checked += 1
                    try:
                        if kind == "closing":
                            result = ClosingPriceService(self.repo, client=client, guard=guard, clock=self.clock).process(eligible)
                            reason = result.get("pending_reason") or "captured"
                            # Opportunities near T-20, T-5 and T-1. A
                            # one-minute tick need not make an API call each time.
                            remaining = (kickoff - self.clock()).total_seconds() / 60
                            delay = max(1, remaining - 5) if remaining > 5 else max(1, remaining - 1) if remaining > 1 else 1
                            if remaining <= 0 and not result.get("finished"):
                                delay = None  # Postponed/missing kickoff: exponential retry, not minute polling.
                            permanent = bool(result.get("finished"))
                        else:
                            result = SettlementService(self.repo, result_loader=FixtureResultLoader(client)).resolve(
                                prediction_ids=[p["id"] for p in eligible], now=self.clock(), write_guard=guard)
                            reasons = result.get("pending_reasons") or {}
                            for pending_reason, count in reasons.items():
                                report["held_reasons"][pending_reason] = report["held_reasons"].get(pending_reason, 0) + count
                            reason = next(iter(reasons), "settled")
                            permanent = bool(reasons) and all(r in PERMANENT for r in reasons)
                            delay = None
                            report["errors"] += result.get("errors", 0)
                        guard()
                        defer(self.runtime, kind, fid, now=self.clock(), reason=reason, revision=revision,
                              attempts=int(state.get("attempts", 0)), permanent=permanent, minutes=delay)
                        report["details"].append({"fixture_id": fid, "kind": kind, "result": result})
                    except BudgetExceeded:
                        raise
                    except Exception as exc:
                        report["errors"] += 1
                        guard()
                        defer(self.runtime, kind, fid, now=self.clock(), reason="provider_or_persistence_failure",
                              revision=revision, attempts=int(state.get("attempts", 0)))
                        report["details"].append({"fixture_id": fid, "kind": kind, "error_type": type(exc).__name__})
            report["provider_attempts"] = getattr(client, "attempts", 0)
            if report["errors"]:
                report["status"] = "partial"
            elif report["held_reasons"] or report["overdue_pending"]:
                report["status"] = "warning"
        except Exception as exc:
            report.update(status="partial", errors=report["errors"] + 1, error=type(exc).__name__)
            report["provider_attempts"] = getattr(locals().get("client"), "attempts", 0)
        finally:
            try:
                self.runtime.finish_run(run_id, report, self.clock())
            finally:
                self.leases.release_lease(lease_key=LEASE, owner_id=owner, now=self.clock())
        return report


def run_measurement_cycle(*, dry_run=False, kinds=("closing", "settlement")):
    """Use the same single-host refresh coordination as the existing worker."""
    from .refresh_coordination import data_access, RefreshBusy
    if dry_run or not MeasurementRuntime().state(CONTROL).get("enabled"):
        return MeasurementCycle().run_once(dry_run=dry_run, kinds=kinds)
    try:
        with data_access():
            return MeasurementCycle().run_once(kinds=kinds)
    except RefreshBusy as exc:
        report = {"status": "refresh_deferred", "errors": 0, "provider_attempts": 0,
                  "reason": str(exc)}
        # Operational heartbeat only; no prediction/result writes while blocked.
        runtime = MeasurementRuntime()
        now = datetime.now(timezone.utc)
        run_id = runtime.start_run(now)
        runtime.finish_run(run_id, report, now)
        return report
