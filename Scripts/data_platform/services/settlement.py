"""On-demand canonical settlement. Scheduling/closing-price capture are separate."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from datetime import date, datetime, time, timezone
import logging

from ..publication_identity import digest
from ..repositories.predictions import PredictionRepository
from ..settlement import Grade, SETTLEMENT_VERSION, grade_selection, parse_fixture_result, provider_id, utc_datetime
from ..settlement_policy import product_policy, uses_participation_cards
from ..participation_cards import parse_participation_cards

logger = logging.getLogger(__name__)


class FixtureResultLoader:
    """Exact fixture + requested team/player stats, once each per fixture."""
    def __init__(self, client=None):
        self.client = client

    def __call__(self, fixture_id, *, need_statistics, need_players=False):
        if self.client is None:
            from .. import config
            from ..sync.apifootball import ApiFootballClient
            self.client = ApiFootballClient(settings=config.SETTINGS)
        fixtures = self.client.fixture(int(fixture_id))
        if len(fixtures) != 1 or provider_id((fixtures[0].get("fixture") or {}).get("id")) != fixture_id:
            raise ValueError("Missing or non-unique exact fixture response")
        fixture = fixtures[0]
        statistics = None
        stats_error = None
        if need_statistics and (fixture.get("fixture", {}).get("status") or {}).get("short") == "FT":
            try:
                statistics = self.client.fixture_statistics(int(fixture_id))
            except Exception:
                # A stats endpoint failure must not prevent grading goal markets.
                logger.warning("Settlement statistics unavailable for fixture %s", fixture_id, exc_info=True)
                stats_error = "statistics_fetch_failed"
        result = parse_fixture_result(fixture, statistics)
        if stats_error:
            result["statistics_error"] = stats_error
        if need_players:
            players = None
            player_error = None
            if result.get("status") == "FT":
                try:
                    players = self.client.fixture_players(int(fixture_id))
                except Exception:
                    logger.warning("Settlement players unavailable for fixture %s", fixture_id, exc_info=True)
                    player_error = "player_statistics_fetch_failed"
            try:
                result["participation_cards"] = parse_participation_cards(result, players)
            except (TypeError, ValueError, AttributeError, KeyError):
                player_error = "malformed_player_statistics"
            if player_error:
                result["participation_cards"] = {"pending_reason": player_error}
        return result


class SettlementService:
    def __init__(self, repo=None, *, result_loader=None, rules=None):
        self.repo = repo or PredictionRepository()
        self.result_loader = result_loader or FixtureResultLoader()
        # Explicit bookmaker policies can override; the default uses only a
        # product-policy snapshot already attached to the new publication.
        self.rules = rules or {}

    def _policy(self, prediction):
        tracking = prediction.get("tracking")
        selection = tracking.get("selection") if isinstance(tracking, Mapping) else None
        selection = selection if isinstance(selection, Mapping) else {}
        return self.rules.get((selection.get("bookmaker"), selection.get("market_key"), selection.get("period")),
                              product_policy(prediction))

    def resolve(self, *, on_or_before=None, league=None, dry_run=False,
                prediction_ids=None, regrade=False, now=None, write_guard=None):
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("now must be timezone aware")
        now = now.astimezone(timezone.utc)
        if on_or_before is not None and not isinstance(on_or_before, date):
            on_or_before = date.fromisoformat(str(on_or_before))
        cutoff = min(now, datetime.combine(on_or_before, time.max, timezone.utc)) if on_or_before else now
        if regrade and not prediction_ids:
            raise ValueError("Corrections require explicit prediction_ids")
        candidates = self.repo.settlement_candidates(
            league=league, prediction_ids=prediction_ids, include_graded=regrade)
        stats = {"backend": "platform", "dry_run": dry_run, "graded": 0, "would_grade": 0,
                 "hit": 0, "half_hit": 0, "miss": 0, "half_miss": 0, "push": 0, "void": 0,
                 "errors": 0, "skipped": 0, "pending": 0, "details": [], "pending_reasons": {}}
        groups = defaultdict(list)

        def record(pred, grade, result=None, policy=None):
            try:
                evidence_hash = digest({"result": result, "rule": policy,
                                        "selection": (pred.get("tracking") or {}).get("selection"),
                                        "version": SETTLEMENT_VERSION})
            except (TypeError, ValueError, AttributeError):
                # Do not let one malformed legacy payload abort other fixtures,
                # and do not serialize NaN/Infinity as valid result evidence.
                grade = Grade(pending_reason="malformed_settlement_input")
                evidence_hash, result, policy = None, None, None
            assessment = {**grade.to_dict(), "version": SETTLEMENT_VERSION,
                          "result": result, "rule": policy, "evidence_hash": evidence_hash,
                          "policy_version": (policy or {}).get("version"),
                          "basis": (policy or {}).get("basis", "exact_market_evidence")}
            detail = {"id": pred["id"], "fixture_id": pred.get("fixture_id"),
                      "fixture": f"{pred['home_team']} vs {pred['away_team']}",
                      "market": pred["market"], "pick": pred["pick"], **grade.to_dict()}
            try:
                if write_guard is not None:
                    write_guard()
                written = "dry_run" if dry_run else self.repo.record_settlement(
                    pred, assessment, checked_at=now, allow_correction=regrade)
                detail["write_status"] = written
                if written not in {"applied", "dry_run", "unchanged"}:
                    stats["errors"] += 1
                elif grade.outcome:
                    if written != "unchanged":
                        stats["would_grade" if dry_run else "graded"] += 1
                        stats[grade.outcome] += 1
                else:
                    stats["pending"] += 1
                    stats["skipped"] += 1
                    reasons = stats["pending_reasons"]
                    reasons[grade.pending_reason] = reasons.get(grade.pending_reason, 0) + 1
            except Exception:
                logger.exception("Settlement persistence failed for prediction %s", pred["id"])
                detail["write_status"] = "failed"
                stats["errors"] += 1
            stats["details"].append(detail)

        for pred in candidates:
            fid = provider_id(pred.get("fixture_id"))
            scheduled = utc_datetime(pred.get("scheduled_kickoff") or pred.get("kickoff"))
            if scheduled and scheduled > cutoff:
                stats["skipped"] += 1
                continue
            if fid is None:
                record(pred, Grade(pending_reason="missing_fixture_id"))
                continue
            groups[fid].append(pred)

        from ..sync.apifootball import COMPETITIONS
        for fid, predictions in groups.items():
            try:
                needs = {"need_statistics": any(
                    pred["market"] in {"corners", "sot"} or
                    (pred["market"] == "cards" and (not product_policy(pred)
                        or self._policy(pred) != product_policy(pred))) for pred in predictions)}
                if any(uses_participation_cards(pred) and self._policy(pred) == product_policy(pred) for pred in predictions):
                    needs["need_players"] = True
                result = self.result_loader(fid, **needs)
                kickoff = utc_datetime(result.get("kickoff"))
                if provider_id(result.get("fixture_id")) != fid:
                    raise ValueError("Fixture result ID mismatch")
            except Exception:
                logger.exception("Settlement fixture fetch failed for %s", fid)
                stats["errors"] += 1
                for pred in predictions:
                    record(pred, Grade(pending_reason="fixture_fetch_failed"))
                continue
            for pred in predictions:
                try:
                    policy = self._policy(pred)
                    spec = COMPETITIONS.get(pred["league"])
                    if spec is None or provider_id(result.get("league_id")) != str(spec.api_football_id):
                        grade = Grade(pending_reason="competition_identity_mismatch")
                    elif kickoff is None:
                        grade = Grade(pending_reason="missing_fixture_kickoff")
                    elif kickoff > cutoff:
                        grade = Grade(pending_reason="fixture_not_due")
                    else:
                        grade = grade_selection(pred, result, rules=policy)
                    record(pred, grade, result, policy)
                except Exception:
                    logger.exception("Malformed settlement input for prediction %s", pred["id"])
                    stats["errors"] += 1
                    record(pred, Grade(pending_reason="malformed_settlement_input"))
        return stats
