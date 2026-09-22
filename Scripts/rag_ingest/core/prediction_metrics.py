"""Request-local operational measurements; never part of a prediction snapshot.

Timings are inclusive (a market contains profile/Chroma time), so they must
not be added together to calculate wall time. ContextVar keeps concurrent
callers isolated and all instrumentation is inert outside a measured run.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import wraps
import json
import logging
from time import perf_counter


_active = ContextVar("prediction_metrics", default=None)
logger = logging.getLogger(__name__)


@dataclass
class PredictionMetrics:
    timings: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    fixtures: dict = field(default_factory=dict)
    fixture_id: str | None = None

    def add(self, section, name, value):
        totals = getattr(self, section)
        totals[name] = totals.get(name, 0) + value
        if self.fixture_id is not None:
            fixture = self.fixtures.setdefault(self.fixture_id, {"timings": {}, "counts": {}})
            values = fixture[section]
            values[name] = values.get(name, 0) + value

    def snapshot(self):
        return {
            "schema_version": "prediction-performance.v1",
            "timings_are_inclusive": True,
            "timings_seconds": {k: round(v, 6) for k, v in self.timings.items()},
            "counts": dict(self.counts),
            "fixtures": {
                key: {"timings_seconds": {k: round(v, 6) for k, v in item["timings"].items()},
                      "counts": dict(item["counts"])}
                for key, item in self.fixtures.items()
            },
        }


@contextmanager
def collect_metrics():
    metrics = PredictionMetrics()
    token = _active.set(metrics)
    try:
        yield metrics
    finally:
        _active.reset(token)


def select_fixture(fixture_id):
    metrics = _active.get()
    if metrics is not None:
        metrics.fixture_id = str(fixture_id)


def count(name, value=1):
    metrics = _active.get()
    if metrics is not None:
        metrics.add("counts", name, value)


def timed_call(name, function, *args, **kwargs):
    metrics = _active.get()
    if metrics is None:
        return function(*args, **kwargs)
    start = perf_counter()
    try:
        return function(*args, **kwargs)
    except BaseException:
        metrics.add("counts", name + ".failures", 1)
        raise
    finally:
        metrics.add("timings", name, perf_counter() - start)
        metrics.add("counts", name + ".calls", 1)


def timed(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            return timed_call(name, function, *args, **kwargs)
        return wrapped
    return decorate


def measure_generation(function):
    """Return metrics on successful runs and retain partial timings on timeout."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with collect_metrics() as metrics:
            started = perf_counter()
            error = None
            try:
                run = function(*args, **kwargs)
            except BaseException as exc:
                # DispatchTimeout deliberately inherits BaseException. Preserve
                # its type and control flow; the worker records the failure.
                error = exc
                raise
            finally:
                metrics.fixture_id = None
                metrics.add("timings", "generation", perf_counter() - started)
                report = metrics.snapshot()
                if error is not None:
                    report["interrupted_by"] = type(error).__name__
                    error.prediction_performance = report
                logger.info("Match Read performance league=%s date=%s stage=%s %s",
                            args[0] if args else kwargs.get("league"),
                            args[1] if len(args) > 1 else kwargs.get("target_date"),
                            kwargs.get("stage", "pre_match"), json.dumps(report, sort_keys=True))
            return replace(run, performance=report)
    return wrapped
