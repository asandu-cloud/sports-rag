"""Fixed-architecture, earlier-only selection for the exploratory C extension."""
from __future__ import annotations

import math
import re

from Scripts.rag_ingest.core.model_features import digest, utc
from .metrics import numerical_metrics
from .selection import fit_blend

KINDS = ("residual_ridge", "offset_xgboost", "team_poisson", "team_catboost")
VERSION = "phase3-extension-selection.v1"


def _validate(row):
    if type(row.get("fixture_id")) is not int or row["fixture_id"] <= 0:
        raise ValueError("Invalid fixture identity")
    if row.get("kind") not in KINDS or not re.fullmatch(r"development-0[1-4]", row.get("fold_id", "")):
        raise ValueError("Invalid architecture or fold")
    if not re.fullmatch(r"[0-9a-f]{64}", row.get("snapshot_id", "")):
        raise ValueError("Invalid snapshot identity")
    if not (utc(row["training_max_label_available_at"]) < utc(row["fit_cutoff"])
            <= utc(row["kickoff"]) < utc(row["label_available_at"])):
        raise ValueError("Prediction violates its training or label cutoff")
    for key in ("target", "prediction", "statistical"):
        value = row.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid target or prediction")
    if row["target"] != int(row["target"]):
        raise ValueError("Target must be a recorded count")


def choose_architecture(records, *, prior_fold_ids, cutoff):
    """Complete equal cohorts only; omit late labels before computing any score."""
    prior = list(prior_fold_ids)
    if prior != [f"development-{i:02d}" for i in range(1, len(prior) + 1)] or len(prior) > 4:
        raise ValueError("Expected consecutive earlier folds")
    if not prior:
        if records:
            raise ValueError("First-block fixed seed cannot inspect predictions")
        result = {"kind": KINDS[0], "policy": "fixed_seed", "prior_fold_ids": [],
                  "selection_cutoff": cutoff, "scores": [], "max_label_available_at": None}
    else:
        cutoff_time = utc(cutoff)
        selected = {kind: {} for kind in KINDS}
        seen = set()
        late = set()
        for row in records:
            _validate(row)
            key = (row["kind"], row["fixture_id"])
            if key in seen:
                raise ValueError("Duplicate fixture/version across selection folds")
            seen.add(key)
            if row["fold_id"] not in prior or utc(row["fit_cutoff"]) >= cutoff_time:
                raise ValueError("Future or undeclared fold entered architecture choice")
            if utc(row["label_available_at"]) >= cutoff_time:
                late.add(row["fixture_id"])
                continue
            selected[row["kind"]][row["fixture_id"]] = row
        reference = selected[KINDS[0]]
        if not reference or {r["fold_id"] for r in reference.values()} != set(prior):
            raise ValueError("Insufficient earlier fold evidence")
        identity_fields = ("snapshot_id", "target", "kickoff", "label_available_at", "fold_id", "statistical")
        scores = []
        for kind in KINDS:
            mapping = selected[kind]
            if set(mapping) != set(reference):
                raise ValueError("Architecture selection cohorts differ")
            for fid, row in mapping.items():
                if any(row[key] != reference[fid][key] for key in identity_fields):
                    raise ValueError("Architecture selection targets or snapshots differ")
            ordered = [mapping[fid] for fid in sorted(mapping)]
            scores.append({"kind": kind, "n": len(ordered),
                           "metrics": numerical_metrics([r["target"] for r in ordered],
                                                        [r["prediction"] for r in ordered]),
                           "prediction_sha256": digest(ordered)})
        choice = min(scores, key=lambda r: (r["metrics"]["rmse"], KINDS.index(r["kind"])))
        result = {"kind": choice["kind"], "policy": "one_fold_warmup" if len(prior) == 1 else "earlier_folds",
                  "prior_fold_ids": prior, "selection_cutoff": cutoff, "scores": scores,
                  "late_label_fixture_ids": sorted(late),
                  "membership_sha256": digest(sorted(reference)),
                  "max_label_available_at": max(r["label_available_at"] for r in reference.values())}
    result["version"] = VERSION
    result["choice_id"] = digest(result)
    return result


def blend_from_oof(records, choices, *, cutoff):
    """Use only the adaptive architecture that was chosen before each forecast."""
    rows, seen = [], set()
    for row in records:
        _validate(row)
        if row["fixture_id"] in seen:
            raise ValueError("Duplicate adaptive OOF fixture")
        seen.add(row["fixture_id"])
        choice = choices.get(row["fold_id"])
        if (not choice or choice.get("choice_id") != digest({k: v for k, v in choice.items() if k != "choice_id"})
                or choice["kind"] != row["kind"] or row.get("choice_id") != choice["choice_id"]
                or utc(choice["selection_cutoff"]) != utc(row["fit_cutoff"])):
            raise ValueError("OOF row does not match its earlier architecture choice")
        number = int(row["fold_id"].split("-")[1])
        if choice["prior_fold_ids"] != [f"development-{i:02d}" for i in range(1, number)]:
            raise ValueError("OOF choice has a future/nonconsecutive selection fold")
        if choice.get("max_label_available_at") is not None and utc(choice["max_label_available_at"]) >= utc(row["fit_cutoff"]):
            raise ValueError("Architecture choice used a future result")
        if utc(row["fit_cutoff"]) >= utc(cutoff):
            raise ValueError("Blend received current/future forecasts")
        if utc(row["label_available_at"]) < utc(cutoff):
            rows.append(row)
    rows.sort(key=lambda row: (utc(row["kickoff"]), row["fixture_id"]))
    evidence = {"fit_cutoff": cutoff, "n": len(rows), "oof_fixture_ids": [r["fixture_id"] for r in rows],
                "oof_fold_ids": sorted({r["fold_id"] for r in rows}), "oof_sha256": digest(rows),
                "max_label_available_at": max((r["label_available_at"] for r in rows), default=None)}
    if len(evidence["oof_fold_ids"]) < 2:
        return {**evidence, "status": "warmup", "weight_ml": 0.0}
    return {**fit_blend([r["target"] for r in rows], [r["statistical"] for r in rows],
                       [r["prediction"] for r in rows]), **evidence, "status": "fitted"}
