"""Operational telemetry must survive interruption without changing outputs."""

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))
from core import prediction_metrics as metrics
from rendering.match_read_dispatch import generate_match_reads_sync


def test_failed_stage_retains_elapsed_time_and_does_not_leak_context(monkeypatch):
    clock = iter([1.0, 3.5])
    monkeypatch.setattr(metrics, "perf_counter", lambda: next(clock))
    with metrics.collect_metrics() as report:
        metrics.select_fixture("123")
        with pytest.raises(ValueError):
            metrics.timed_call("retrieval", lambda: (_ for _ in ()).throw(ValueError("fixture failed")))
    metrics.count("must_not_leak")
    assert report.timings == {"retrieval": 2.5}
    assert report.counts == {"retrieval.calls": 1, "retrieval.failures": 1}
    assert report.snapshot()["fixtures"]["123"]["timings_seconds"]["retrieval"] == 2.5


def test_dispatch_timeout_carries_partial_profile_timings():
    from Scripts.data_platform.services.worker_runtime import DispatchTimeout

    def evaluate(*args, **kwargs):
        def lookup():
            raise DispatchTimeout("timed out")
        metrics.timed_call("profile_context", lookup)

    with pytest.raises(DispatchTimeout) as caught:
        generate_match_reads_sync(
            "EPL", date(2026, 9, 19),
            fetcher=lambda *a, **kw: ([{"id": "123", "home_team": "Home", "away_team": "Away"}], []),
            date_filter=lambda events, _: events,
            enricher=lambda events, *a: (events, []),
            evaluator=evaluate,
        )
    report = caught.value.prediction_performance
    assert report["interrupted_by"] == "DispatchTimeout"
    assert report["counts"]["profile_context.calls"] == 1
    assert report["counts"]["market_evaluation.calls"] == 1
    assert report["fixtures"]["123"]["timings_seconds"]["profile_context"] >= 0
    assert "generation" in report["timings_seconds"]
    assert metrics._active.get() is None


def test_concurrent_measurements_are_isolated():
    def run(fixture):
        with metrics.collect_metrics() as report:
            metrics.select_fixture(fixture)
            metrics.count("lookups")
        return report.snapshot()

    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(run, ["1", "2"]))
    assert set(reports[0]["fixtures"]) == {"1"}
    assert set(reports[1]["fixtures"]) == {"2"}
