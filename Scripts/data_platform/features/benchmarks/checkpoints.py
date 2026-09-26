"""Immutable task checkpoints and exclusive, specification-bound continuation."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
import time

import numpy as np

from Scripts.rag_ingest.core.model_features import digest
from .artifacts import read_json, sha, verify_complete

VERSION = "phase3-forward-checkpoints.v1"
_ATTEMPT = re.compile(r"attempt-[0-9]{4}")
_HASH = re.compile(r"[0-9a-f]{64}")


def _directory(path):
    path = Path(path).absolute()
    if path.is_symlink() or path.resolve() != path or not path.is_dir():
        raise ValueError("Unsafe or missing checkpoint directory")
    return path


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path, value):
    """Durably publish complete JSON without ever replacing an existing name.

    The temporary file lives under the destination's .runtime directory, on
    the same filesystem. A hard link publishes its fsynced inode atomically;
    link's EEXIST behavior supplies the no-overwrite guarantee. A killed writer
    can leave only an uncommitted .runtime file or a complete final document.
    """
    path = Path(path).absolute()
    parent = _directory(path.parent)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    runtime = parent / ".runtime"
    if runtime.is_symlink():
        raise ValueError("Unsafe atomic-write runtime directory")
    runtime.mkdir(exist_ok=True)
    _directory(runtime)
    descriptor, temporary_name = tempfile.mkstemp(prefix="json-", suffix=".pending", dir=runtime)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic commit, no replace even against a race.
        _fsync_directory(parent)
    finally:
        # Interrupted processes may leave this file; completion never hashes it.
        temporary.unlink(missing_ok=True)
        _fsync_directory(runtime)


def complete_atomic(directory):
    """Checksum immutable artifacts, then commit COMPLETE atomically once."""
    directory = _directory(directory)
    destination = directory / "COMPLETE.json"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    hashes = {}
    for item in sorted(directory.rglob("*")):
        relative = item.relative_to(directory)
        if ".runtime" in relative.parts or item == destination:
            continue
        if item.is_symlink() or not item.resolve().is_relative_to(directory):
            raise ValueError("Unsafe completion artifact path")
        if item.is_file():
            with item.open("rb") as handle:
                os.fsync(handle.fileno())
            hashes[str(relative)] = sha(item)
    atomic_write_json(destination, hashes)
    return hashes


def _verify_attempt(directory, attempt_name, specification_id, *, contract=None):
    directory = _directory(directory)
    if not isinstance(attempt_name, str) or not _ATTEMPT.fullmatch(attempt_name):
        raise ValueError("Invalid checkpoint attempt pointer")
    attempt = _directory(directory / attempt_name)
    if (attempt / "COMPLETE.json").is_symlink():
        raise ValueError("Unsafe checkpoint completion manifest")
    artifacts = verify_complete(attempt, required={"STARTED.json", "result.json", "predictions.npz"})
    start = read_json(attempt / "STARTED.json")
    if (not isinstance(start, dict) or set(start) != {"version", "specification_id", "task"}
            or start["version"] != VERSION or start["specification_id"] != specification_id
            or (contract is not None and start["task"] != contract)):
        raise ValueError("Checkpoint task contract mismatch")
    if digest(start) != directory.name:
        raise ValueError("Checkpoint task key does not match its saved specification/contract")
    return attempt, artifacts


def _verified_pointer(directory, specification_id, *, contract=None):
    directory = _directory(directory)
    success = directory / "SUCCESS.json"
    if success.is_symlink():
        raise ValueError("Unsafe checkpoint success pointer")
    pointer = read_json(success)
    if (not isinstance(pointer, dict) or set(pointer) != {"attempt", "completion_sha256"}
            or not isinstance(pointer.get("completion_sha256"), str)
            or not _HASH.fullmatch(pointer["completion_sha256"])):
        raise ValueError("Invalid checkpoint attempt pointer")
    attempt, artifacts = _verify_attempt(directory, pointer["attempt"], specification_id, contract=contract)
    if digest(artifacts) != pointer["completion_sha256"]:
        raise ValueError("Checkpoint completion identity mismatch")
    return attempt, pointer


def _load_result(attempt):
    result = read_json(attempt / "result.json")
    if not isinstance(result, dict):
        raise ValueError("Invalid checkpoint result")
    with np.load(attempt / "predictions.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != {"predictions", "raw_predictions"}:
            raise ValueError("Invalid checkpoint prediction archive")
        predictions = arrays["predictions"].copy()
        raw = arrays["raw_predictions"].copy()
    if (predictions.ndim != 1 or raw.shape != predictions.shape
            or not np.isfinite(predictions).all() or np.any(predictions < 0)
            or not np.isfinite(raw).all()):
        raise ValueError("Invalid checkpoint predictions")
    return {**result, "predictions": predictions, "raw_predictions": raw}


def verify_committed_tasks(path, specification_id):
    """Read-only verification of all published task pointers for RESULT resume.

    Incomplete task directories are reported separately, without repairing or
    deleting them. Each published pointer binds a fully checksummed attempt,
    its exact specification/task-key identity and valid saved prediction arrays.
    """
    path = _directory(path)
    verified, incomplete = [], []
    for directory in sorted(path.iterdir()):
        if directory.name == ".runtime":
            continue
        if not _HASH.fullmatch(directory.name) or not directory.is_dir() or directory.is_symlink():
            raise ValueError("Unsafe or unrecognized task checkpoint directory")
        success = directory / "SUCCESS.json"
        if success.exists() or success.is_symlink():
            attempt, pointer = _verified_pointer(directory, specification_id)
            _load_result(attempt)
            verified.append({"checkpoint_id": directory.name, **pointer})
        else:
            incomplete.append(directory.name)
    return {"version": VERSION, "specification_id": specification_id,
            "committed_tasks": len(verified), "tasks": verified, "incomplete_tasks": incomplete}


class RunBudgetReached(RuntimeError):
    """A resumable stop before starting another fit."""


@contextmanager
def experiment_lock(path):
    path = Path(path)
    lock = path / ".run.lock"
    if lock.is_symlink():
        raise ValueError("Unsafe experiment lock")
    with lock.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process owns this experiment") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class Checkpoints:
    def __init__(self, path, specification_id, *, max_seconds=28800, max_fits=2500):
        self.path = Path(path).absolute()
        _directory(self.path.parent)
        if self.path.is_symlink():
            raise ValueError("Unsafe checkpoint root")
        self.specification_id = specification_id
        self.started = time.monotonic()
        self.max_seconds, self.max_fits = max_seconds, max_fits
        self.created = 0
        self.reused = 0
        self.cache = {}
        self.path.mkdir(exist_ok=True)
        _directory(self.path)
        # Count all actual fit attempts, including interrupted attempts. The
        # experiment budget cannot be reset by restarting the process.
        self.previous_fits = len(list(self.path.glob("*/attempt-*/STARTED.json")))

    def budget(self):
        if time.monotonic() - self.started >= self.max_seconds:
            raise RunBudgetReached("Active execution budget reached; resume with the same specification")
        if self.previous_fits + self.created >= self.max_fits:
            raise RunBudgetReached("Experiment fit budget reached; no automatic search expansion")

    def get(self, contract, compute):
        key = digest({"version": VERSION, "specification_id": self.specification_id, "task": contract})
        if key in self.cache:
            self.reused += 1
            return self.cache[key]
        directory = self.path / key
        if directory.is_symlink():
            raise ValueError("Unsafe task checkpoint")
        success = directory / "SUCCESS.json"
        if not success.exists() and not success.is_symlink() and directory.exists():
            # A crash after COMPLETE but before SUCCESS must not spend a second
            # fit. Choose the earliest valid completed attempt deterministically.
            for orphan in sorted(directory.glob("attempt-*")):
                if (orphan / "COMPLETE.json").exists() or (orphan / "COMPLETE.json").is_symlink():
                    attempt, artifacts = _verify_attempt(directory, orphan.name, self.specification_id, contract=contract)
                    _load_result(attempt)
                    atomic_write_json(success, {"attempt": attempt.name, "completion_sha256": digest(artifacts)})
                    break
        if success.exists() or success.is_symlink():
            attempt, _ = _verified_pointer(directory, self.specification_id, contract=contract)
            result = _load_result(attempt)
            self.reused += 1
        else:
            self.budget()
            directory.mkdir(exist_ok=True)
            attempts = list(directory.glob("attempt-*"))
            if any(not _ATTEMPT.fullmatch(p.name) or p.is_symlink() or not p.is_dir() for p in attempts):
                raise ValueError("Unsafe checkpoint attempt directory")
            number = max((int(p.name.split("-")[1]) for p in attempts), default=0) + 1
            attempt = directory / f"attempt-{number:04d}"
            attempt.mkdir()
            atomic_write_json(attempt / "STARTED.json", {"version": VERSION,
                       "specification_id": self.specification_id, "task": contract})
            self.created += 1
            began = time.monotonic()
            try:
                result = compute(attempt)
                result = {**result, "fit_seconds": time.monotonic() - began}
                predictions = np.asarray(result.pop("predictions"), dtype=float)
                raw = np.asarray(result.pop("raw_predictions", predictions), dtype=float)
                if (predictions.ndim != 1 or raw.shape != predictions.shape
                        or not np.isfinite(predictions).all() or np.any(predictions < 0)
                        or not np.isfinite(raw).all()):
                    raise ValueError("Invalid checkpoint predictions")
                np.savez_compressed(attempt / "predictions.npz",
                                    predictions=predictions, raw_predictions=raw)
                atomic_write_json(attempt / "result.json", result)
                hashes = complete_atomic(attempt)
                atomic_write_json(success, {"attempt": attempt.name, "completion_sha256": digest(hashes)})
                result = {**result, "predictions": predictions, "raw_predictions": raw}
            except Exception as exc:
                # A completed attempt is immutable even if pointer publication
                # fails. The next invocation will verify and adopt it.
                if not (attempt / "COMPLETE.json").exists():
                    atomic_write_json(attempt / "FAILED.json", {"type": type(exc).__name__, "message": str(exc)})
                raise
        result["checkpoint_id"] = key
        self.cache[key] = result
        return result

    def statistics(self):
        return {"version": VERSION, "new_fits": self.created, "existing_attempts": self.previous_fits,
                "task_reuse": self.reused, "unique_tasks_loaded": len(self.cache),
                "invocation_seconds": time.monotonic() - self.started,
                "max_seconds_per_invocation": self.max_seconds, "max_fits_per_experiment": self.max_fits}
