"""Fingerprint the deployed numerical code and available model artifacts.

This is deployment provenance, not proof that a particular ML model contributed
to a prediction, nor a replacement for historical input/profile snapshots.
Cached per process: restart workers after changing numerical code/artifacts.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path


@lru_cache(maxsize=1)
def prediction_system_manifest():
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / "Scripts/rag_ingest/core").glob("*.py"))
    paths += [root / "Scripts/rag_ingest" / name for name in
              ("weights.py", "prob_models.py", "ml_edge.py", "odds_provider.py")]
    paths += [root / "Index/ml_models" / name for name in
              ("model_r2.json", "goals_ridge.joblib", "corners_ridge.joblib",
               "cards_ridge.joblib", "sot_ridge.joblib")]
    files = {}
    for path in paths:
        try:
            with path.open("rb") as handle:
                files[str(path.relative_to(root))] = hashlib.file_digest(handle, "sha256").hexdigest()
        except FileNotFoundError:
            files[str(path.relative_to(root))] = None
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"version": "prediction-system.v1:" + digest, "files_sha256": files,
            "scope": "deployed_code_and_available_artifacts; not model-use or input-vintage evidence"}
