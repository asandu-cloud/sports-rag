"""Aggregated platform health report."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from ..config import SETTINGS
from ..db import session_scope
from ..models import Competition, Fixture, Team
from .checks import QualityIssue, run_data_quality_checks
from .metrics import queue_metrics, sync_job_metrics, watermark_metrics


@dataclass
class HealthSignal:
    name: str
    status: str     # 'ok' | 'warn' | 'error'
    message: str
    details: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class HealthReport:
    status: str                          # 'ok' | 'warn' | 'error'
    signals: List[HealthSignal] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    sync: Dict[str, Any] = field(default_factory=dict)
    queue: Dict[str, Any] = field(default_factory=dict)
    watermarks: Dict[str, Any] = field(default_factory=dict)
    data_quality: List[Dict[str, Any]] = field(default_factory=list)
    env: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "signals": [s.as_dict() for s in self.signals],
            "counts": self.counts,
            "sync": self.sync,
            "queue": self.queue,
            "watermarks": self.watermarks,
            "data_quality": self.data_quality,
            "env": self.env,
        }


def _worst(a: str, b: str) -> str:
    order = {"ok": 0, "warn": 1, "error": 2}
    return a if order[a] >= order[b] else b


def _severity_to_status(severity: str) -> str:
    return {"info": "ok", "warn": "warn", "error": "error"}.get(severity, "warn")


def build_health_report(*, include_data_quality: bool = True) -> HealthReport:
    report = HealthReport(status="ok")

    # Pull high-level counts first
    with session_scope() as session:
        from ..models import KBDocument, Prediction, LiveFixture
        report.counts = {
            "competitions": session.query(Competition).count(),
            "teams": session.query(Team).count(),
            "fixtures": session.query(Fixture).count(),
            "kb_documents": session.query(KBDocument).count(),
            "predictions": session.query(Prediction).count(),
            "live_fixtures": session.query(LiveFixture).count(),
        }

    report.sync = sync_job_metrics()
    report.queue = queue_metrics()
    report.watermarks = watermark_metrics()

    # Turn watermark staleness into a signal
    if report.watermarks["stale"]:
        report.signals.append(HealthSignal(
            name="stale_watermarks", status="warn",
            message=f"{len(report.watermarks['stale'])} watermark(s) have not advanced recently",
            details={"scopes": [row["scope"] for row in report.watermarks["stale"][:10]]},
        ))
        report.status = _worst(report.status, "warn")

    # Data quality
    if include_data_quality:
        issues = run_data_quality_checks()
        report.data_quality = [
            {"check": i.check, "severity": i.severity, "summary": i.summary,
             "count": i.count, "details": i.details or {}}
            for i in issues
        ]
        for issue in issues:
            status = _severity_to_status(issue.severity)
            if status == "ok":
                continue
            report.signals.append(HealthSignal(
                name=issue.check, status=status,
                message=issue.summary,
                details={"count": issue.count} | (issue.details or {}),
            ))
            report.status = _worst(report.status, status)

    # Environment snapshot (no secrets)
    report.env = {
        "database_driver": _database_driver(),
        "platform_enabled": SETTINGS.platform_enabled,
        "raw_archive_backend": SETTINGS.raw_archive_backend,
    }
    return report


def _database_driver() -> str:
    url = SETTINGS.database_url
    if "://" in url:
        return url.split("://", 1)[0]
    return url
