"""Durable budgets, activation cutoff, retries and run history for measurement.

Reuses existing operational tables. No schema migration or prediction backfill.
"""
from datetime import datetime, timedelta, timezone
import time

from sqlalchemy import func, select, update

from ..db import session_scope, retry_database_busy
from ..models import PublishedRecommendation, SyncRun, SyncWatermark
from ..sync.apifootball import ApiFootballClient, ApiFootballResponseError, API_BASE

CONTROL = "prediction_measurement:control"
RUN_KIND = "prediction_measurement"


class BudgetExceeded(RuntimeError):
    pass


def utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


class MeasurementRuntime:
    def __init__(self, session_factory=session_scope):
        self.factory = session_factory

    def state(self, scope):
        with self.factory() as session:
            row = session.scalar(select(SyncWatermark).where(SyncWatermark.scope == scope))
            return dict(row.meta or {}) if row else {}

    @retry_database_busy
    def set_state(self, scope, value):
        with self.factory() as session:
            row = session.scalar(select(SyncWatermark).where(SyncWatermark.scope == scope))
            if row is None:
                row = SyncWatermark(scope=scope)
                session.add(row)
            row.meta = dict(value)

    def activate(self, *, now=None):
        """A restart/reinstall preserves the original prospective boundary."""
        now = now or datetime.now(timezone.utc)
        with self.factory() as session:
            row = session.scalar(select(SyncWatermark).where(SyncWatermark.scope == CONTROL))
            if row is None:
                row = SyncWatermark(scope=CONTROL, meta={
                    "activated_at": now.isoformat(),
                    "after_recommendation_id": session.scalar(select(func.max(PublishedRecommendation.id))) or 0,
                    "daily_request_limit": 500, "cycle_request_limit": 20,
                })
                session.add(row)
            row.meta = {**row.meta, "enabled": True}
            return dict(row.meta)

    def deactivate(self):
        state = self.state(CONTROL)
        if state:
            self.set_state(CONTROL, {**state, "enabled": False})

    @retry_database_busy
    def reserve_request(self, *, day, limit):
        """Reserve before sending, including failures; a restart cannot reset it.

        The worker's named lease serializes initial ledger creation. The CAS
        also protects against competing processes after that first reservation.
        """
        scope = f"measurement:requests:{day}"
        with self.factory() as session:
            row = session.scalar(select(SyncWatermark).where(SyncWatermark.scope == scope))
            if row is None:
                if limit < 1:
                    raise BudgetExceeded("Daily provider request budget exhausted")
                session.add(SyncWatermark(scope=scope, meta={"attempts": 1}))
                return
            attempts = int((row.meta or {}).get("attempts", 0))
            if attempts >= limit:
                raise BudgetExceeded("Daily provider request budget exhausted")
            result = session.execute(update(SyncWatermark).where(
                SyncWatermark.id == row.id, SyncWatermark.updated_at == row.updated_at,
            ).values(meta={"attempts": attempts + 1}).execution_options(synchronize_session=False))
            if result.rowcount != 1:
                raise BudgetExceeded("Request budget changed concurrently; retry next cycle")

    def start_run(self, now):
        with self.factory() as session:
            row = SyncRun(run_kind=RUN_KIND, scope="prospective_publications", started_at=now, status="running")
            session.add(row)
            session.flush()
            return row.id

    def finish_run(self, run_id, report, now):
        with self.factory() as session:
            row = session.get(SyncRun, run_id)
            row.finished_at, row.status, row.stats = now, report["status"], dict(report)
            row.error_text = report.get("error")

    def status(self, now=None):
        now = now or datetime.now(timezone.utc)
        control = self.state(CONTROL)
        with self.factory() as session:
            rows = session.scalars(select(SyncRun).where(SyncRun.run_kind == RUN_KIND)
                                   .order_by(SyncRun.id.desc()).limit(5)).all()
            runs = [{"id": row.id, "started_at": utc(row.started_at).isoformat(),
                     "finished_at": utc(row.finished_at).isoformat() if row.finished_at else None,
                     "status": row.status, "report": row.stats, "error": row.error_text} for row in rows]
            tasks = session.scalars(select(SyncWatermark).where(
                SyncWatermark.scope.startswith("measurement:settlement:") |
                SyncWatermark.scope.startswith("measurement:closing:")
            )).all()
            pending = [{"scope": task.scope, **dict(task.meta or {})} for task in tasks
                       if (task.meta or {}).get("reason") not in {"settled", "captured"}]
        enabled = control.get("enabled", False)
        latest = runs[0] if runs else None
        stale = enabled and (latest is None or (now - utc(latest["finished_at"] or latest["started_at"])).total_seconds() > 300)
        problem = bool(latest and latest["status"] in {"failed", "partial"})
        return {"enabled": enabled, "control": control, "recent_runs": runs,
                "pending_tasks": pending,
                "requests_today": self.state(f"measurement:requests:{now.date().isoformat()}"),
                "status": "disabled" if not enabled else "error" if problem else "warn" if stale or pending or (latest and latest["status"] in {"warning", "refresh_deferred"}) else "ok",
                "stale": bool(stale)}


class BoundedFootballClient(ApiFootballClient):
    """No hidden retries/pagination/redirects; every transport attempt is counted."""
    def __init__(self, runtime, *, cycle_limit, daily_limit, guard, clock=None, **kwargs):
        super().__init__(**kwargs)
        self.runtime, self.cycle_limit, self.daily_limit = runtime, cycle_limit, daily_limit
        self.guard, self.clock, self.attempts = guard, clock or (lambda: datetime.now(timezone.utc)), 0

    def _get(self, path, params=None):
        if path not in {"/fixtures", "/fixtures/statistics", "/fixtures/players", "/odds"}:
            raise ValueError("Endpoint outside measurement scope")
        self.guard()
        if self.attempts >= self.cycle_limit:
            raise BudgetExceeded("Cycle provider request budget exhausted")
        self.runtime.reserve_request(day=self.clock().date().isoformat(), limit=self.daily_limit)
        self.attempts += 1
        response = self.session.get(API_BASE + path, params=params or {}, timeout=8, allow_redirects=False)
        if response.status_code != 200:
            raise ApiFootballResponseError(f"Measurement provider HTTP {response.status_code}")
        payload = response.json()
        if (not isinstance(payload, dict) or payload.get("errors")
                or not isinstance(payload.get("response"), list)
                or (payload.get("paging") or {}).get("total", 1) > 1):
            raise ApiFootballResponseError("Invalid, failed or paginated measurement response")
        self.guard()
        if self.request_pause_s:
            time.sleep(self.request_pause_s)
        return payload


def defer(runtime, kind, fid, *, now, reason, revision, attempts, permanent=False, minutes=None):
    delay = minutes if minutes is not None else min(360, 5 * 2 ** min(attempts, 7))
    previous = runtime.state(f"measurement:{kind}:{fid}")
    runtime.set_state(f"measurement:{kind}:{fid}", {
        "reason": reason, "revision": revision, "attempts": attempts + 1,
        "last_checked_at": now.isoformat(), "next_attempt_at": (now + timedelta(minutes=delay)).isoformat(),
        "first_checked_at": previous.get("first_checked_at", now.isoformat()),
        "permanent": permanent,
    })
