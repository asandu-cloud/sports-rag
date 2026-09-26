"""Pure, development-only forecast summaries and paired calendar uncertainty.

One row per fixture/market/method is required. Multiple forecast versions must
be evaluated in separate reports, never counted as extra independent fixtures.
No prices, settlement outcomes or betting-performance estimates enter here.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from typing import Mapping

import numpy as np

from .metrics import evaluate_predictions, numerical_metrics


VERSION = "phase3-evaluation.v1"
MARKETS = ("goals", "corners", "sot")
SLICE_FIELDS = ("competition", "season", "season_stage", "source_class", "round_group",
                "support_band", "missingness_band", "forecast_stage")
GLOBAL_FIXTURE_FIELDS = ("kickoff", "competition", "season", "fold_id", "season_stage")
ANCHOR_MONDAY = datetime(1970, 1, 5, tzinfo=timezone.utc)
FAMILY_COMPARISONS = 6
MAIN_CONFIDENCE = 1.0 - 0.05 / FAMILY_COMPARISONS
MIN_PRECISION_WEEKS = 20
MIN_SLICE_FIXTURES = 200
MIN_SLICE_DATES = 20


def _utc(value):
    try:
        stamp = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Kickoff must be an explicit timezone-aware timestamp") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Kickoff must include an explicit timezone")
    return stamp.astimezone(timezone.utc)


def calendar_block(kickoff, *, block_weeks=1):
    """UTC Monday or fixed consecutive fortnight, including empty calendar gaps."""
    if type(block_weeks) is not int or block_weeks not in (1, 2):
        raise ValueError("Only the predeclared one- and two-week blocks are supported")
    day = _utc(kickoff).date()
    index = (day - ANCHOR_MONDAY.date()).days // (7 * block_weeks)
    return (ANCHOR_MONDAY + timedelta(weeks=index * block_weeks)).date().isoformat()


def _normalise(records):
    rows, seen, fixture_metadata, market_metadata = [], set(), {}, {}
    for original in records:
        if not isinstance(original, Mapping):
            raise ValueError("Every prediction row must be a mapping")
        row = dict(original)
        if type(row.get("fixture_id")) is not int or row["fixture_id"] <= 0:
            raise ValueError("Each prediction requires an exact positive fixture ID")
        if row.get("market") not in MARKETS:
            raise ValueError("Unsupported evaluation market")
        for field in ("method", "fold_id", *(f for f in SLICE_FIELDS if f != "season")):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing or invalid evaluation field: {field}")
        if type(row.get("season")) is not int or not 2000 <= row["season"] <= 2100:
            raise ValueError("An explicit provider season is required")
        row["kickoff"] = _utc(row.get("kickoff")).isoformat()
        for field in ("target", "prediction"):
            value = row.get(field)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError(f"{field} must be a finite nonnegative number")
            try:
                row[field] = float(value)
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"Invalid {field}") from exc
            if not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError(f"{field} must be a finite nonnegative number")
        if not row["target"].is_integer():
            raise ValueError("Targets must be observed integer counts")
        key = (row["market"], row["method"], row["fixture_id"])
        if key in seen:
            raise ValueError("Duplicate fixture/market/method; evaluate versions or stages separately")
        seen.add(key)
        shared = tuple(row[field] for field in GLOBAL_FIXTURE_FIELDS)
        if row["fixture_id"] in fixture_metadata and fixture_metadata[row["fixture_id"]] != shared:
            raise ValueError("Fixture metadata or fold assignment differs across prediction rows")
        fixture_metadata[row["fixture_id"]] = shared
        # Support/missingness/source evidence may differ by market, but methods
        # on one market must describe the same target and evaluation slices.
        market_key = (row["fixture_id"], row["market"])
        specific = (row["target"], *(row[field] for field in SLICE_FIELDS))
        if market_key in market_metadata and market_metadata[market_key] != specific:
            raise ValueError("Paired market target or slice metadata differs between methods")
        market_metadata[market_key] = specific
        rows.append(row)
    if not rows:
        raise ValueError("Evaluation requires nonempty prediction records")
    return sorted(rows, key=lambda row: (row["market"], row["method"], row["kickoff"], row["fixture_id"]))


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _support(rows, *, block_weeks=1):
    fixtures = sorted(row["fixture_id"] for row in rows)
    return {"unique_fixtures": len(fixtures),
            "match_dates": len({_utc(row["kickoff"]).date() for row in rows}),
            "calendar_weeks": len({calendar_block(row["kickoff"]) for row in rows}),
            "calendar_blocks": len({calendar_block(row["kickoff"], block_weeks=block_weeks) for row in rows}),
            "fixture_membership_sha256": _hash(fixtures)}


def _summary(rows, *, distribution=False):
    targets = [r["target"] for r in rows]
    predictions = [r["prediction"] for r in rows]
    if distribution:
        scores = evaluate_predictions(targets, predictions)
        scores.pop("per_fixture", None)
    else:
        scores = numerical_metrics(targets, predictions)
    support = _support(rows)
    return {**scores, "support": support,
            "slice_count_support": support["unique_fixtures"] >= MIN_SLICE_FIXTURES
                                   and support["match_dates"] >= MIN_SLICE_DATES,
            "calendar_precision_support": support["calendar_weeks"] >= MIN_PRECISION_WEEKS}


def _groups(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return dict(sorted(groups.items()))


def _equal_league(rows):
    """Descriptive macro weighting; never substitutes for equal-fixture fitting."""
    groups = _groups(rows, "competition")
    ordered, weights = [], []
    for league_rows in groups.values():
        ordered.extend(league_rows)
        weights.extend([1.0 / (len(groups) * len(league_rows))] * len(league_rows))
    mass = np.asarray(weights)
    target = np.asarray([row["target"] for row in ordered])
    predicted = np.asarray([row["prediction"] for row in ordered])
    try:
        with np.errstate(over="raise", invalid="raise"):
            errors = predicted - target
            mse = float(np.sum(mass * errors ** 2))
            target_mean = float(np.sum(mass * target))
            variance = float(np.sum(mass * (target - target_mean) ** 2))
            # Exactly constant targets have undefined R², even if floating
            # summation of weights perturbs the calculated weighted mean.
            constant = bool(np.all(target == target[0]))
            result = {"n": len(ordered), "leagues": len(groups), "mae": float(np.sum(mass * np.abs(errors))),
                      "bias": float(np.sum(mass * errors)), "rmse": math.sqrt(mse),
                      "r2": None if constant or variance == 0 else 1.0 - mse / variance,
                      "r2_unavailable_reason": "constant_targets" if constant or variance == 0 else None,
                      "weighting": "equal_league_mass_then_equal_fixture_within_league",
                      "league_fixture_counts": {league: len(items) for league, items in groups.items()},
                      "descriptive_only": True}
    except FloatingPointError as exc:
        raise ValueError("Equal-league metrics overflow") from exc
    if any(not math.isfinite(result[key]) for key in ("mae", "bias", "rmse")) or (
            result["r2"] is not None and not math.isfinite(result["r2"])):
        raise ValueError("Nonfinite equal-league metric")
    return result


def summarize(records):
    """Equal-fixture metrics by market/method, with descriptive coverage slices.

    Common uncalibrated Poisson diagnostics run once per market/method overall;
    slice and fold summaries use numerical metrics only. Different method
    coverage stays visible; use ``paired_comparison`` for strict comparisons.
    """
    rows = _normalise(records)
    result = {"version": VERSION, "weighting": "equal_fixture", "by_market": {},
              "qualification": "not_assessed", "betting_metrics": "not_computed",
              "distribution_scope": "overall_per_market_method_only",
              "slice_fields": list(SLICE_FIELDS)}
    for market, market_rows in _groups(rows, "market").items():
        methods, memberships = {}, set()
        for method, method_rows in _groups(market_rows, "method").items():
            overall = _summary(method_rows, distribution=True)
            memberships.add(overall["support"]["fixture_membership_sha256"])
            methods[method] = {
                "overall": overall,
                "by_fold": {fold: _summary(items) for fold, items in _groups(method_rows, "fold_id").items()},
                "slices": {field: {value: _summary(items) for value, items in _groups(method_rows, field).items()}
                           for field in SLICE_FIELDS},
                "equal_league": _equal_league(method_rows),
            }
        result["by_market"][market] = {"methods": methods,
                                        "identical_method_fixture_membership": len(memberships) == 1}
    return result


def _bootstrap_plan(rows, *, resamples, block_weeks, seed):
    blocks = sorted({calendar_block(row["kickoff"], block_weeks=block_weeks) for row in rows})
    lookup = {block: index for index, block in enumerate(blocks)}
    random = np.random.default_rng(seed)
    draws = random.multinomial(len(blocks), np.full(len(blocks), 1.0 / len(blocks)), size=resamples)
    configuration = {"resamples": resamples, "block_weeks": block_weeks, "seed": seed,
                     "utc_monday_anchor": ANCHOR_MONDAY.date().isoformat(),
                     "sampled_calendar_blocks": blocks, "nonempty_global_blocks": len(blocks),
                     "sampling": "common_nonempty_UTC_calendar_blocks_with_replacement",
                     "draws_sha256": hashlib.sha256(draws.astype("<i8", copy=False).tobytes()).hexdigest(),
                     "main_confidence": MAIN_CONFIDENCE, "family_comparisons": FAMILY_COMPARISONS,
                     "slice_one_sided_confidence": 0.95,
                     "quantile_method": "linear", "relative_metric_units": "fraction_not_percentage",
                     "precision_minimum_calendar_weeks": MIN_PRECISION_WEEKS,
                     "slice_minimum_fixtures": MIN_SLICE_FIXTURES,
                     "slice_minimum_match_dates": MIN_SLICE_DATES,
                     "dependence_limitation": "Repeated-team dependence may extend beyond sampled calendar blocks"}
    return lookup, draws, configuration


def _paired_summary(candidate_rows, baseline_by_fixture, *, lookup, draws, block_weeks, slice_report):
    targets = np.asarray([row["target"] for row in candidate_rows])
    candidate = np.asarray([row["prediction"] for row in candidate_rows])
    baseline = np.asarray([baseline_by_fixture[row["fixture_id"]]["prediction"] for row in candidate_rows])
    candidate_metrics = numerical_metrics(targets, candidate)
    baseline_metrics = numerical_metrics(targets, baseline)
    support = _support(candidate_rows, block_weeks=block_weeks)
    denominator = baseline_metrics["rmse"]
    improvement = None if denominator == 0 else 1.0 - candidate_metrics["rmse"] / denominator
    reasons = []
    if slice_report and support["unique_fixtures"] < MIN_SLICE_FIXTURES:
        reasons.append("fewer_than_200_slice_fixtures")
    if slice_report and support["match_dates"] < MIN_SLICE_DATES:
        reasons.append("fewer_than_20_slice_match_dates")
    if support["calendar_weeks"] < MIN_PRECISION_WEEKS:
        reasons.append("fewer_than_20_calendar_weeks")
    if denominator == 0:
        reasons.append("zero_baseline_rmse")
    result = {"candidate_metrics": candidate_metrics, "baseline_metrics": baseline_metrics,
              "support": support, "relative_rmse_improvement": improvement,
              "relative_rmse_deterioration": None if improvement is None else -improvement,
              "improvement_interval": None, "deterioration_upper_95": None,
              "interval_status": "unavailable", "unavailable_reasons": reasons,
              "valid_bootstrap_resamples": 0,
              "point_improvement_at_least_one_percent": None if improvement is None else improvement >= 0.01,
              "improvement_interval_excludes_zero": None,
              "slice_deterioration_below_five_percent": None,
              "qualification": "not_assessed"}
    if reasons:
        return result
    assignments = np.asarray([lookup[calendar_block(row["kickoff"], block_weeks=block_weeks)]
                              for row in candidate_rows], dtype=int)
    block_counts = np.bincount(assignments, minlength=len(lookup)).astype(float)
    with np.errstate(over="raise", invalid="raise"):
        try:
            candidate_sse = np.bincount(assignments, weights=(candidate - targets) ** 2, minlength=len(lookup))
            baseline_sse = np.bincount(assignments, weights=(baseline - targets) ** 2, minlength=len(lookup))
            # einsum without optimize uses bounded aggregate operations instead
            # of refitting, expanding fixtures or invoking a BLAS thread pool.
            sampled_counts = np.einsum("ij,j->i", draws, block_counts, optimize=False)
            sampled_candidate = np.einsum("ij,j->i", draws, candidate_sse, optimize=False)
            sampled_baseline = np.einsum("ij,j->i", draws, baseline_sse, optimize=False)
        except FloatingPointError as exc:
            raise ValueError("Calendar bootstrap numerical overflow") from exc
    if not all(np.isfinite(values).all() for values in (sampled_counts, sampled_candidate, sampled_baseline)):
        raise ValueError("Calendar bootstrap produced nonfinite aggregates")
    empty = int(np.sum(sampled_counts == 0))
    zero_baseline = int(np.sum((sampled_counts > 0) & (sampled_baseline == 0)))
    if empty:
        reasons.append("empty_cohort_in_bootstrap_resamples")
    if zero_baseline:
        reasons.append("zero_baseline_rmse_in_bootstrap_resamples")
    result["valid_bootstrap_resamples"] = len(draws) - empty - zero_baseline
    result["invalid_bootstrap_resamples"] = {"empty_cohort": empty, "zero_baseline_rmse": zero_baseline}
    if reasons:
        # Dropping invalid draws would silently change the declared interval.
        return result
    # Counts cancel in the ratio of paired RMSEs, while block SSEs retain the
    # equal-fixture weighting even when sampled weeks have different sizes.
    bootstrap_improvement = 1.0 - np.sqrt(sampled_candidate / sampled_baseline)
    if not np.isfinite(bootstrap_improvement).all():
        raise ValueError("Nonfinite relative RMSE bootstrap result")
    if slice_report:
        upper = float(np.quantile(-bootstrap_improvement, 0.95, method="linear"))
        result["deterioration_upper_95"] = upper
        result["slice_deterioration_below_five_percent"] = upper < 0.05
    else:
        alpha = (1.0 - MAIN_CONFIDENCE) / 2.0
        lower, upper = np.quantile(bootstrap_improvement, [alpha, 1.0 - alpha], method="linear")
        result["improvement_interval"] = {"confidence": MAIN_CONFIDENCE, "lower": float(lower), "upper": float(upper)}
        result["improvement_interval_excludes_zero"] = bool(lower > 0)
    result["interval_status"] = "available"
    return result


def paired_comparison(records, candidate, baseline, resamples=2000, block_weeks=1, seed=42):
    """Strict paired RMSE intervals with one shared block draw plan per call.

    Main intervals use family-wise 95% / six-comparison Bonferroni coverage.
    Slice deterioration uses one-sided 95%, with explicit unsupported slices.
    Numerical gates are evidence fields, not model/promotion approval.
    """
    if not isinstance(candidate, str) or not isinstance(baseline, str) or not candidate or not baseline or candidate == baseline:
        raise ValueError("Choose two distinct named prediction methods")
    if type(resamples) is not int or not 2 <= resamples <= 20_000:
        raise ValueError("Bootstrap resamples must be an integer from 2 through 20000")
    if type(seed) is not int or seed < 0:
        raise ValueError("Bootstrap seed must be a nonnegative integer")
    if type(block_weeks) is not int or block_weeks not in (1, 2):
        raise ValueError("Only the predeclared one- and two-week blocks are supported")
    all_rows = _normalise(records)
    rows = [row for row in all_rows if row["method"] in (candidate, baseline)]
    if {row["method"] for row in rows} != {candidate, baseline}:
        raise ValueError("Both named methods must have predictions")
    candidates, baselines = {}, {}
    for market, market_rows in _groups(rows, "market").items():
        candidates[market] = [row for row in market_rows if row["method"] == candidate]
        baselines[market] = {row["fixture_id"]: row for row in market_rows if row["method"] == baseline}
        if {row["fixture_id"] for row in candidates[market]} != set(baselines[market]):
            raise ValueError(f"Paired fixture membership mismatch for {market}; no silent intersection is allowed")
    lookup, draws, configuration = _bootstrap_plan(rows, resamples=resamples, block_weeks=block_weeks, seed=seed)
    result = {"version": VERSION, "candidate": candidate, "baseline": baseline,
              "configuration": configuration, "by_market": {}, "weighting": "equal_fixture",
              "qualification": "not_assessed", "betting_metrics": "not_computed"}
    for market in sorted(candidates):
        selected = candidates[market]

        def compare(items, *, slice_report):
            return _paired_summary(items, baselines[market], lookup=lookup, draws=draws,
                                   block_weeks=block_weeks, slice_report=slice_report)

        result["by_market"][market] = {
            "overall": compare(selected, slice_report=False),
            "slices": {field: {value: compare(items, slice_report=True)
                               for value, items in _groups(selected, field).items()}
                       for field in SLICE_FIELDS},
        }
    return result
