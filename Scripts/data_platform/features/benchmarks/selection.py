"""Frozen recipe search and selection-honest chronological OOF blending.

Only supplied earlier development records enter selection. This module does
not load data, fit estimators, inspect held-out targets or select by profit.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
import re

import numpy as np

from Scripts.rag_ingest.core.model_features import digest
from .data import family_gate

VERSION = "phase3-forward-selection.v1"
FAMILIES = ("ridge", "poisson", "lightgbm", "xgboost", "catboost")
LOOKBACKS = (730, 1460, None)
HALF_LIVES = (None, 180, 365, 730)
PARAMETERS = {
    "ridge": ({"alpha": 1.0}, {"alpha": 10.0}, {"alpha": 50.0}, {"alpha": 100.0}),
    "poisson": ({"alpha": .1}, {"alpha": 1.0}, {"alpha": 10.0}),
    "lightgbm": ({"num_leaves": 7, "min_child_samples": 50}, {"num_leaves": 15, "min_child_samples": 100}),
    "xgboost": ({"max_depth": 3}, {"max_depth": 4}),
    "catboost": ({"depth": 3}, {"depth": 4}),
}
SEED_PARAMETERS = {"ridge": {"alpha": 50.0}, "poisson": {"alpha": 1.0},
                   "lightgbm": {"num_leaves": 7, "min_child_samples": 50},
                   "xgboost": {"max_depth": 3}, "catboost": {"depth": 3}}
TREE_FIXED = {"iterations": 300, "learning_rate": .03, "early_stopping": False,
              "xgboost": {"tree_method": "hist", "objective": "count:poisson", "reg_lambda": 10},
              "catboost": {"loss_function": "Poisson", "l2_leaf_reg": 10, "bootstrap_type": "No"},
              "lightgbm": {"objective": "poisson"}}
_HEX = re.compile(r"[0-9a-f]{64}")
_FOLD = re.compile(r"development-([0-9]{2})")


def recipes():
    """Return fresh dictionaries for the 156 approved reference recipes."""
    result = []
    for family in FAMILIES:
        for params in PARAMETERS[family]:
            config = "-".join(f"{key}{value:g}" for key, value in sorted(params.items()))
            for lookback in LOOKBACKS:
                for half_life in HALF_LIVES:
                    ident = f"{family}-{config}-window{lookback or 'all'}-half{half_life or 'none'}"
                    result.append({"recipe_id": ident, "family": family, "params": dict(params),
                                   "lookback_days": lookback, "half_life_days": half_life})
    return result


def tie_key(recipe):
    """Exact-score ties: simpler family, shorter window, stable recipe ID."""
    family = recipe.get("family")
    if family not in FAMILIES or recipe.get("lookback_days") not in LOOKBACKS:
        raise ValueError("Recipe is outside the frozen family/window contract")
    return (FAMILIES.index(family), recipe["lookback_days"] or math.inf, recipe["recipe_id"])


def _utc(value):
    if not isinstance(value, (str, datetime)):
        raise ValueError("An explicit timezone-aware cutoff is required")
    value = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Cutoff must include timezone")
    return value.astimezone(timezone.utc)


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _fold_number(value):
    match = _FOLD.fullmatch(value) if isinstance(value, str) else None
    if not match or not int(match[1]):
        raise ValueError("Invalid development fold ID")
    return int(match[1])


def _fold_ids(values):
    result = list(values)
    numbers = [_fold_number(item) for item in result]
    if numbers != sorted(set(numbers)):
        raise ValueError("Selection folds must be unique and chronological")
    return result


def _hash(value, name):
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise ValueError("Invalid " + name + " hash")
    return value


def _finish_choice(result):
    result["choice_id"] = digest(result)
    return result


def choose_recipe(records, prior_fold_ids, *, fit_cutoff, family=None, min_folds=2):
    """Rank complete recipes on the exact same earlier fixtures/targets.

    Each complete score record has recipe_id, fold_id, status, rmse, n,
    validation_membership_sha256, targets_sha256, support, fit_cutoff,
    training_max_label_available_at and max_label_available_at. ``fit_cutoff``
    in a record is the model's source-fold cutoff; this function's argument is
    the receiving fit/selection cutoff. A caller must filter late labels before
    aggregating records. Failed/unsupported/missing recipes remain explicit.
    """
    prior = _fold_ids(prior_fold_ids)
    cutoff = _utc(fit_cutoff)
    if family is not None and family not in FAMILIES:
        raise ValueError("Unknown selection family")
    if type(min_folds) is not int or min_folds < 1:
        raise ValueError("Minimum prior folds must be a positive integer")
    catalog = {r["recipe_id"]: r for r in recipes() if family is None or r["family"] == family}
    all_ids = {r["recipe_id"] for r in recipes()}
    selected, cohorts, source_cutoffs = {}, {}, {}
    provenance = []
    for record in records:
        # Crucially, no scores/provenance from non-allowed folds are inspected.
        fold = record.get("fold_id")
        if fold not in prior:
            continue
        recipe_id = record.get("recipe_id")
        if recipe_id not in all_ids:
            raise ValueError("Score references a recipe outside the frozen grid")
        if recipe_id not in catalog:
            continue
        key = (recipe_id, fold)
        if key in selected:
            raise ValueError("Duplicate recipe/fold score")
        status = record.get("status")
        if status not in {"complete", "failed", "insufficient_support", "incomplete"}:
            raise ValueError("Unknown recipe score status")
        item = {"recipe_id": recipe_id, "fold_id": fold, "status": status}
        if status == "complete":
            n, rmse = record.get("n"), record.get("rmse")
            if type(n) is not int or n <= 0 or not _finite(rmse) or rmse < 0:
                raise ValueError("Invalid development score/count")
            source = _utc(record.get("fit_cutoff"))
            training_max = _utc(record.get("training_max_label_available_at"))
            scoring_max = _utc(record.get("max_label_available_at"))
            if not training_max < source < scoring_max < cutoff:
                raise ValueError("Training/scoring labels are not strictly available before their receiving cutoff")
            support = record.get("support")
            if not isinstance(support, dict) or support.get("validation_unique_fixtures") != n:
                raise ValueError("Selection score count disagrees with support")
            gate = family_gate(support, catalog[recipe_id]["family"])
            cohort = (n, _hash(record.get("validation_membership_sha256"), "validation membership"),
                      _hash(record.get("targets_sha256"), "target"), scoring_max.isoformat())
            if fold in cohorts and cohort != cohorts[fold]:
                raise ValueError("Recipes do not use identical validation cohorts/targets")
            if fold in source_cutoffs and source != source_cutoffs[fold]:
                raise ValueError("Recipes disagree on their source-fold fitting cutoff")
            cohorts[fold], source_cutoffs[fold] = cohort, source
            item.update(rmse=float(rmse), n=n, gate=gate, fit_cutoff=source.isoformat(),
                        training_max_label_available_at=training_max.isoformat(), max_label_available_at=scoring_max.isoformat(),
                        validation_membership_sha256=cohort[1], targets_sha256=cohort[2])
        selected[key] = item
        provenance.append(item)
    known = [source_cutoffs[f] for f in prior if f in source_cutoffs]
    if any(a >= b for a, b in zip(known, known[1:])):
        raise ValueError("Source-fold cutoffs are not chronological")
    provenance.sort(key=lambda r: (_fold_number(r["fold_id"]), r["recipe_id"]))
    result = {"version": VERSION, "status": "insufficient_development_evidence", "recipe": None, "recipe_id": None,
              "pooled_rmse": None, "n": 0, "selection_fold_ids": prior, "selection_cutoff": cutoff.isoformat(),
              "selection_records_sha256": digest(provenance), "selection_max_label_available_at":
                  max((r["max_label_available_at"] for r in provenance if r["status"] == "complete"), default=None),
              "family": family, "minimum_prior_folds": min_folds, "ranking": [], "excluded": [],
              "aggregation": "equal-fixture pooled squared error, then square root"}
    for recipe_id, recipe in catalog.items():
        reasons = []
        if len(prior) < min_folds:
            reasons.append("insufficient_prior_validation_folds")
        available = []
        for fold in prior:
            record = selected.get((recipe_id, fold))
            if record is None:
                reasons.append(f"{fold}:missing_score")
            elif record["status"] != "complete":
                reasons.append(f"{fold}:{record['status']}")
            elif not record["gate"]["qualified"]:
                reasons.extend(f"{fold}:{why}" for why in record["gate"]["reasons"])
            else:
                available.append(record)
        if reasons:
            result["excluded"].append({"recipe_id": recipe_id, "reasons": reasons})
            continue
        count = sum(r["n"] for r in available)
        try:
            squared = math.fsum(r["rmse"] ** 2 * r["n"] for r in available)
        except OverflowError as exc:
            raise ValueError("Selection score overflow") from exc
        if count <= 0 or not math.isfinite(squared):
            raise ValueError("Invalid aggregate selection score")
        result["ranking"].append({"recipe_id": recipe_id, "rmse": math.sqrt(squared / count), "n": count})
    result["ranking"].sort(key=lambda item: (item["rmse"], tie_key(catalog[item["recipe_id"]])))
    result["excluded"].sort(key=lambda item: tie_key(catalog[item["recipe_id"]]))
    if result["ranking"]:
        best = result["ranking"][0]
        result.update(status="selected", recipe=catalog[best["recipe_id"]], recipe_id=best["recipe_id"],
                      pooled_rmse=best["rmse"], n=best["n"])
    return _finish_choice(result)


def chronological_choices(records_by_prediction_fold, folds, *, family=None):
    """Select a genuine adaptive OOF path, with explicit first/second warmups.

    ``folds`` are ordered dicts containing fold_id and fit_cutoff. The mapping
    supplies score records filtered for each receiving fold's cutoff. A score
    from the receiving fold itself is ignored by choose_recipe. Family paths
    use their fixed Batch B default; the primary cross-family path starts with
    Ridge alpha 50, all earlier examples, 365-day decay.
    """
    if family is not None and family not in FAMILIES:
        raise ValueError("Unknown selection family")
    folds = list(folds)
    ids = _fold_ids([f["fold_id"] for f in folds])
    if [_fold_number(f) for f in ids] != list(range(1, len(ids) + 1)):
        raise ValueError("An OOF path must start at the first development fold without gaps")
    cutoffs = [_utc(f["fit_cutoff"]) for f in folds]
    if any(a >= b for a, b in zip(cutoffs, cutoffs[1:])):
        raise ValueError("OOF prediction cutoffs must be chronological")
    result = {}
    for index, (fold_id, cutoff) in enumerate(zip(ids, cutoffs)):
        if not index:
            seed_family = family or "ridge"
            seed = next(r for r in recipes() if r["family"] == seed_family and r["params"] == SEED_PARAMETERS[seed_family]
                        and r["lookback_days"] is None and r["half_life_days"] == 365)
            choice = {"version": VERSION, "status": "selected", "recipe": seed, "recipe_id": seed["recipe_id"],
                      "selection_fold_ids": [], "selection_cutoff": cutoff.isoformat(),
                      "selection_max_label_available_at": None, "selection_records_sha256": digest([]),
                      "minimum_prior_folds": 0, "family": family, "pooled_rmse": None, "n": 0,
                      "ranking": [], "excluded": []}
        else:
            choice = choose_recipe(records_by_prediction_fold.get(fold_id, []), ids[:index],
                                   fit_cutoff=cutoff, family=family, min_folds=1 if index == 1 else 2)
            choice.pop("choice_id")
        end = folds[index].get("validation_end")
        end = _utc(end) if end is not None else cutoffs[index + 1] if index + 1 < len(cutoffs) else None
        if end is not None and (end <= cutoff or (index + 1 < len(cutoffs) and end > cutoffs[index + 1])):
            raise ValueError("OOF evaluation interval overlaps the next fitting cutoff")
        choice.update(prediction_fold_id=fold_id, prediction_end=end.isoformat() if end is not None else None,
                      prior_fold_ids=ids[:index],
                      stage="fixed_seed" if index == 0 else "one_fold_warmup" if index == 1 else "nested_forward",
                      outer_evaluation_eligible=index >= 2)
        result[fold_id] = _finish_choice(choice)
    return result


def fit_blend(targets, statistical, ml):
    """Equal-fixture convex least squares; exact ties choose the lower ML weight."""
    arrays = []
    for values in (targets, statistical, ml):
        try:
            values = list(values)
        except TypeError as exc:
            raise ValueError("Blend values must be numeric sequences") from exc
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)) for v in values):
            raise ValueError("Blend values must be numeric, not booleans/nulls")
        try:
            array = np.asarray(values, dtype=float)
        except (ValueError, OverflowError) as exc:
            raise ValueError("Invalid blend values") from exc
        if array.ndim != 1 or not len(array) or not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError("Blend values must be finite nonnegative one-dimensional arrays")
        arrays.append(array)
    y, baseline, model = arrays
    if not (y.shape == baseline.shape == model.shape) or np.any(y != np.floor(y)):
        raise ValueError("Blend targets must be integer counts on identical paired fixtures")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            difference, residual = model - baseline, y - baseline
            denominator = float(np.dot(difference, difference))
            numerator = float(np.dot(difference, residual))
            unconstrained = numerator / denominator if denominator else None
            analytic = min(1.0, max(0.0, unconstrained)) if unconstrained is not None else 0.0
            candidates = sorted({0.0, analytic, 1.0})
            objectives = [(float(np.dot(y - (baseline + weight * difference), y - (baseline + weight * difference))), weight)
                          for weight in candidates]
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError("Blend optimization overflow") from exc
    if any(not math.isfinite(score) for score, _ in objectives) or not math.isfinite(denominator + numerator):
        raise ValueError("Nonfinite blend objective")
    objective, weight = min(objectives)
    return {"version": VERSION, "weight_ml": weight, "weight_statistical": 1.0 - weight, "n": len(y),
            "sse": objective, "rmse": math.sqrt(objective / len(y)), "unconstrained_weight_ml": unconstrained,
            "reason": "identical_predictions" if not denominator else "boundary_statistical" if weight == 0
                      else "boundary_ml" if weight == 1 else "interior", "tie_rule": "lower_ml_weight",
            "pairs_sha256": digest({"targets": y.tolist(), "statistical": baseline.tolist(), "ml": model.tolist()})}


def fit_oof_blend(rows, choices, *, fit_cutoff, min_folds=2):
    """Fit only earlier OOF pairs from the recorded adaptive selection path.

    Every row carries fixture_id, fold_id, recipe_id, kickoff, label_available_at,
    training_max_label_available_at, target, statistical and ml. Choices must be
    the signed output from chronological_choices, not a retrospectively chosen
    fixed recipe's OOF predictions. A checksum detects accidental substitutions;
    this is a research integrity contract, not authentication of untrusted code.
    """
    if type(min_folds) is not int or min_folds < 1:
        raise ValueError("Minimum blend folds must be positive")
    cutoff = _utc(fit_cutoff)
    rows = list(rows)
    seen, used_choices = set(), {}
    for row in rows:
        fid, fold = row.get("fixture_id"), row.get("fold_id")
        if type(fid) is not int or fid <= 0 or fid in seen:
            raise ValueError("Duplicate or invalid OOF fixture")
        seen.add(fid)
        choice = choices.get(fold)
        if not isinstance(choice, dict) or choice.get("status") != "selected":
            raise ValueError("OOF row has no valid chronological selection")
        if choice.get("choice_id") != digest({k: v for k, v in choice.items() if k != "choice_id"}):
            raise ValueError("OOF choice provenance checksum mismatch")
        number = _fold_number(fold)
        prior = [f"development-{i:02d}" for i in range(1, number)]
        if (choice.get("prediction_fold_id") != fold or choice.get("selection_fold_ids") != prior
                or choice.get("prior_fold_ids") != prior or row.get("recipe_id") != choice.get("recipe_id")):
            raise ValueError("Retrospective or nonchronological OOF recipe selection")
        canonical = next((r for r in recipes() if r["recipe_id"] == choice["recipe_id"]), None)
        if choice.get("recipe") != canonical or canonical is None:
            raise ValueError("OOF recipe is outside the frozen grid")
        if choice.get("family") is not None and canonical["family"] != choice["family"]:
            raise ValueError("OOF family path changed family")
        source_cutoff = _utc(choice.get("selection_cutoff"))
        kickoff, label_time = _utc(row.get("kickoff")), _utc(row.get("label_available_at"))
        if not (_utc(row.get("training_max_label_available_at")) < source_cutoff <= kickoff < label_time < cutoff):
            raise ValueError("OOF training/scoring labels are not available at their cutoff")
        if choice.get("prediction_end") is not None and kickoff >= _utc(choice["prediction_end"]):
            raise ValueError("OOF fixture lies outside its source validation fold")
        if number == 1:
            if choice.get("stage") != "fixed_seed" or choice.get("selection_max_label_available_at") is not None:
                raise ValueError("First OOF fold must use the predeclared seed")
            family = choice.get("family") or "ridge"
            recipe = choice.get("recipe", {})
            if (family not in SEED_PARAMETERS or recipe.get("family") != family or recipe.get("params") != SEED_PARAMETERS[family]
                    or recipe.get("lookback_days") is not None or recipe.get("half_life_days") != 365):
                raise ValueError("First OOF fold used a retrospectively chosen seed")
        else:
            required_stage = "one_fold_warmup" if number == 2 else "nested_forward"
            if (choice.get("stage") != required_stage or choice.get("minimum_prior_folds") != (1 if number == 2 else 2)
                    or _utc(choice.get("selection_max_label_available_at")) >= source_cutoff):
                raise ValueError("OOF hyperparameter selection observed its own or future outcomes")
            if not choice.get("ranking") or choice["ranking"][0]["recipe_id"] != choice["recipe_id"]:
                raise ValueError("OOF recipe differs from the preceding-fold winner")
        used_choices[fold] = choice["choice_id"]
    if len(used_choices) < min_folds:
        raise ValueError("Insufficient prior chronological OOF folds for blend")
    rows.sort(key=lambda r: (_utc(r["kickoff"]), r["fixture_id"]))
    result = fit_blend([r["target"] for r in rows], [r["statistical"] for r in rows], [r["ml"] for r in rows])
    result.update(fit_cutoff=cutoff.isoformat(), fold_ids=sorted(used_choices, key=_fold_number),
                  choices_sha256=digest(used_choices), fixture_ids_sha256=digest([r["fixture_id"] for r in rows]),
                  max_label_available_at=max(_utc(r["label_available_at"]) for r in rows).isoformat(),
                  oof_policy="adaptive_preceding_fold_selection_with_fixed_seed_and_one_fold_warmup")
    return result
