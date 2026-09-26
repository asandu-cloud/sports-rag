"""Equal-fixture point and common, uncalibrated Poisson diagnostics.

The mean floor and numerical tail policy apply identically to every family.
These scores are not fitted count distributions or calibrated betting prices.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson

MEAN_FLOOR = 1e-12
TAIL_MASS_TOLERANCE = 1e-12
RPS_TAIL_TOLERANCE = 1e-10
MAX_SUPPORT = 100_000
INTERVAL_LEVELS = (0.50, 0.80, 0.95)


def _arrays(y_true, means):
    def values(items, name):
        if isinstance(items, (str, bytes)):
            raise ValueError(name + " must be a one-dimensional numeric sequence")
        try:
            raw = list(items)
            if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.integer, np.floating)) for x in raw):
                raise ValueError(name + " contains a nonnumeric value or boolean")
            result = np.asarray(raw, dtype=float)
        except (TypeError, OverflowError) as exc:
            raise ValueError(name + " is not numeric") from exc
        if result.ndim != 1 or not len(result) or not np.isfinite(result).all() or np.any(result < 0):
            raise ValueError(name + " must be nonempty, finite and nonnegative")
        return result
    targets, predictions = values(y_true, "targets"), values(means, "means")
    if targets.shape != predictions.shape or np.any(targets != np.floor(targets)):
        raise ValueError("Targets must be integer counts with one prediction per fixture")
    return targets, predictions


def numerical_metrics(y_true, means):
    targets, predictions = _arrays(y_true, means)
    errors = predictions - targets
    with np.errstate(over="raise", invalid="raise"):
        try:
            mse = float(np.mean(errors ** 2))
            denominator = float(np.sum((targets - targets.mean()) ** 2))
            result = {"n": len(targets), "mae": float(np.mean(np.abs(errors))),
                      "bias": float(errors.mean()), "rmse": math.sqrt(mse),
                      "r2": None if denominator == 0 else 1.0 - float(np.sum(errors ** 2)) / denominator,
                      "r2_unavailable_reason": "constant_targets" if denominator == 0 else None}
        except FloatingPointError as exc:
            raise ValueError("Numerical metrics overflow") from exc
    if any(value is not None and not math.isfinite(value) for key, value in result.items() if key != "r2_unavailable_reason"):
        raise ValueError("Nonfinite numerical metric")
    return result


def poisson_scores(y_true, means):
    """Poisson NLL, discrete ranked probability score and central intervals.

    RPS sums squared CDF errors on integer counts k >= 0. Beyond a cutoff K
    at or above the observed count, its remainder is at most mu*P(X>K)^2:
    sum SF(k)^2 <= SF(K)*sum SF(k) <= mu*SF(K)^2. Both mass and score
    bounds are checked; unsupported numerical ranges raise rather than truncate.
    """
    targets, predictions = _arrays(y_true, means)
    adjusted = np.maximum(predictions, MEAN_FLOOR)
    if np.any(adjusted > MAX_SUPPORT) or np.any(targets > MAX_SUPPORT):
        raise ValueError("Count/mean exceeds declared Poisson numerical support limit")
    nll = adjusted - targets * np.log(adjusted) + gammaln(targets + 1.0)
    rps, tail_bounds, tail_masses, cutoffs = [], [], [], []
    for target, mean in zip(targets, adjusted):
        quantile = poisson.isf(TAIL_MASS_TOLERANCE, mean)
        if not np.isfinite(quantile):
            raise ValueError("Poisson tail quantile is not finite")
        cutoff = max(int(target), int(math.ceil(quantile)))
        while True:
            if cutoff > MAX_SUPPORT:
                raise ValueError("Poisson tail cannot be certified within numerical support limit")
            tail_mass = float(poisson.sf(cutoff, mean))
            tail_bound = float(mean * tail_mass ** 2)
            if tail_mass <= TAIL_MASS_TOLERANCE and tail_bound <= RPS_TAIL_TOLERANCE:
                break
            cutoff += 1
        counts = np.arange(cutoff + 1, dtype=float)
        cdf = poisson.cdf(counts, mean)
        if not np.isfinite(cdf).all() or np.any(cdf < 0) or np.any(cdf > 1):
            raise ValueError("Invalid Poisson distribution")
        rps.append(float(np.sum((cdf - (counts >= target)) ** 2)))
        tail_bounds.append(tail_bound); tail_masses.append(tail_mass); cutoffs.append(cutoff)
    if not np.isfinite(nll).all() or np.any(nll < 0) or not np.isfinite(rps).all():
        raise ValueError("Invalid Poisson score")
    intervals, per_fixture_intervals = {}, {}
    for level in INTERVAL_LEVELS:
        lower = poisson.ppf((1 - level) / 2, adjusted)
        upper = poisson.ppf((1 + level) / 2, adjusted)
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(upper < lower):
            raise ValueError("Invalid Poisson interval")
        key = str(round(100 * level))
        covered = (targets >= lower) & (targets <= upper)
        intervals[key] = {"nominal_coverage": level, "coverage": float(covered.mean()),
                          "mean_width": float(np.mean(upper - lower))}
        per_fixture_intervals[key] = {"lower": lower.astype(int).tolist(), "upper": upper.astype(int).tolist(),
                                      "covered": covered.tolist()}
    return {"n": len(targets), "poisson_log_loss": float(np.mean(nll)), "rps": float(np.mean(rps)),
            "intervals": intervals, "distribution": "common_uncalibrated_poisson",
            "numerical_policy": {"mean_floor": MEAN_FLOOR, "floored_means": int(np.sum(predictions < MEAN_FLOOR)),
                "tail_mass_tolerance": TAIL_MASS_TOLERANCE, "rps_tail_tolerance": RPS_TAIL_TOLERANCE,
                "max_support": MAX_SUPPORT, "maximum_used_cutoff": max(cutoffs),
                "maximum_omitted_tail_mass": max(tail_masses), "maximum_rps_remainder_bound": max(tail_bounds),
                "interval_convention": "inclusive central equal-tail quantiles; width = upper - lower"},
            "per_fixture": {"poisson_log_loss": nll.tolist(), "rps": rps,
                            "rps_remainder_bound": tail_bounds, "intervals": per_fixture_intervals}}


def evaluate_predictions(y_true, means):
    """All predeclared diagnostics; no weighting, filtering or silent row drops."""
    # Materialize generators once; both contracts then see identical fixtures.
    targets, predictions = _arrays(y_true, means)
    return {**numerical_metrics(targets, predictions), **poisson_scores(targets, predictions)}
