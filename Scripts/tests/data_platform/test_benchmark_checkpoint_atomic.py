from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from Scripts.data_platform.features.benchmarks import checkpoints
from Scripts.data_platform.features.benchmarks.artifacts import verify_complete
from Scripts.rag_ingest.core.model_features import digest


class Interrupted(BaseException):
    """Simulated process interruption, bypassing ordinary failure recording."""


def computed(attempt):
    return {"status": "complete", "predictions": np.array([1., 2.]), "raw_predictions": np.array([.5, 2.])}


def files(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file() and ".runtime" not in p.parts}


def test_atomic_json_publishes_only_complete_fsynced_payload(tmp_path, monkeypatch):
    destination = tmp_path / "SUCCESS.json"
    original_link, original_fsync = checkpoints.os.link, checkpoints.os.fsync
    synced = []
    monkeypatch.setattr(checkpoints.os, "fsync", lambda fd: (synced.append(fd), original_fsync(fd))[1])
    def inspect_link(source, target):
        assert synced, "file bytes must be fsynced before the commit link"
        assert not destination.exists()
        assert Path(source).parent == tmp_path / ".runtime"
        assert json.loads(Path(source).read_text()) == {"message": "complete", "values": list(range(1000))}
        original_link(source, target)
    monkeypatch.setattr(checkpoints.os, "link", inspect_link)
    checkpoints.atomic_write_json(destination, {"message": "complete", "values": list(range(1000))})
    assert json.loads(destination.read_text())["message"] == "complete"
    assert len(synced) >= 3  # inode, destination directory, temporary directory cleanup
    assert list((tmp_path / ".runtime").iterdir()) == []


def test_atomic_no_overwrite_including_concurrent_writers(tmp_path):
    destination = tmp_path / "RESULT.json"
    def publish(index):
        try:
            checkpoints.atomic_write_json(destination, {"writer": index})
            return index
        except FileExistsError:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(publish, range(8)))
    winners = [r for r in results if r is not None]
    assert len(winners) == 1
    original = destination.read_bytes()
    assert json.loads(original) == {"writer": winners[0]}
    with pytest.raises(FileExistsError):
        checkpoints.atomic_write_json(destination, {"replacement": True})
    assert destination.read_bytes() == original


def test_interruption_before_commit_never_leaves_partial_final_name(tmp_path, monkeypatch):
    original_link = checkpoints.os.link
    def stop(source, target):
        assert json.loads(Path(source).read_text()) == {"complete": True}
        raise Interrupted()
    monkeypatch.setattr(checkpoints.os, "link", stop)
    with pytest.raises(Interrupted):
        checkpoints.atomic_write_json(tmp_path / "RESULT.json", {"complete": True})
    assert not (tmp_path / "RESULT.json").exists()
    monkeypatch.setattr(checkpoints.os, "link", original_link)
    checkpoints.atomic_write_json(tmp_path / "RESULT.json", {"complete": True})
    assert json.loads((tmp_path / "RESULT.json").read_text()) == {"complete": True}


def test_invalid_json_and_symlink_paths_never_publish(tmp_path):
    with pytest.raises(ValueError):
        checkpoints.atomic_write_json(tmp_path / "invalid.json", {"bad": float("nan")})
    assert list(tmp_path.iterdir()) == []
    external = tmp_path / "external"
    external.mkdir()
    (tmp_path / ".runtime").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="runtime"):
        checkpoints.atomic_write_json(tmp_path / "RESULT.json", {})
    assert not (tmp_path / "RESULT.json").exists()


def test_completion_hashes_nested_artifacts_but_excludes_runtime_and_never_overwrites(tmp_path):
    (tmp_path / "value.json").write_text('{"complete":true}')
    nested = tmp_path / "candidate"
    nested.mkdir()
    (nested / "model.bin").write_bytes(b"fitted")
    checkpoints.complete_atomic(nested)
    scratch = tmp_path / ".runtime"
    scratch.mkdir()
    (scratch / "partial.pending").write_bytes(b"uncommitted")
    expected = checkpoints.complete_atomic(tmp_path)
    assert set(expected) == {"value.json", "candidate/model.bin", "candidate/COMPLETE.json"}
    assert verify_complete(tmp_path) == expected
    before = (tmp_path / "COMPLETE.json").read_bytes()
    with pytest.raises(FileExistsError):
        checkpoints.complete_atomic(tmp_path)
    assert (tmp_path / "COMPLETE.json").read_bytes() == before


@pytest.mark.parametrize("exception", [Interrupted, RuntimeError])
def test_orphan_complete_attempt_is_recovered_before_budget_without_refitting(tmp_path, monkeypatch, exception):
    original_link = checkpoints.os.link
    def stop_pointer(source, target):
        if Path(target).name == "SUCCESS.json":
            raise exception("interrupted pointer publication")
        return original_link(source, target)
    monkeypatch.setattr(checkpoints.os, "link", stop_pointer)
    store = checkpoints.Checkpoints(tmp_path / "tasks", "frozen", max_fits=1)
    with pytest.raises(exception):
        store.get({"task": 1}, computed)
    attempt = next((tmp_path / "tasks").glob("*/attempt-*"))
    before = files(attempt)
    assert "COMPLETE.json" in before and "FAILED.json" not in before
    assert not (attempt.parent / "SUCCESS.json").exists()
    monkeypatch.setattr(checkpoints.os, "link", original_link)
    resumed = checkpoints.Checkpoints(tmp_path / "tasks", "frozen", max_fits=1)
    restored = resumed.get({"task": 1}, lambda _: pytest.fail("completed orphan must not refit"))
    assert restored["predictions"].tolist() == [1, 2]
    assert resumed.created == 0 and resumed.reused == 1 and resumed.previous_fits == 1
    assert files(attempt) == before
    assert len(list((tmp_path / "tasks").glob("*/attempt-*"))) == 1
    verified = checkpoints.verify_committed_tasks(tmp_path / "tasks", "frozen")
    assert verified["committed_tasks"] == 1 and verified["incomplete_tasks"] == []


def test_interrupted_completion_preserves_partial_attempt_and_creates_separate_retry(tmp_path, monkeypatch):
    original_link = checkpoints.os.link
    def stop_completion(source, target):
        if Path(target).name == "COMPLETE.json":
            raise Interrupted()
        return original_link(source, target)
    monkeypatch.setattr(checkpoints.os, "link", stop_completion)
    store = checkpoints.Checkpoints(tmp_path / "tasks", "frozen", max_fits=2)
    with pytest.raises(Interrupted):
        store.get({"task": 1}, computed)
    first = next((tmp_path / "tasks").glob("*/attempt-*"))
    before = files(first)
    assert "COMPLETE.json" not in before and "result.json" in before
    monkeypatch.setattr(checkpoints.os, "link", original_link)
    resumed = checkpoints.Checkpoints(tmp_path / "tasks", "frozen", max_fits=2)
    resumed.get({"task": 1}, computed)
    assert files(first) == before
    assert resumed.created == 1 and resumed.previous_fits == 1
    assert len(list(first.parent.glob("attempt-*"))) == 2


def orphan(tmp_path, monkeypatch):
    original = checkpoints.os.link
    def stop(source, target):
        if Path(target).name == "SUCCESS.json":
            raise Interrupted()
        return original(source, target)
    monkeypatch.setattr(checkpoints.os, "link", stop)
    with pytest.raises(Interrupted):
        checkpoints.Checkpoints(tmp_path / "tasks", "spec").get({"task": 1}, computed)
    monkeypatch.setattr(checkpoints.os, "link", original)
    return next((tmp_path / "tasks").glob("*/attempt-*"))


def test_corrupted_orphan_completion_fails_closed_without_refit(tmp_path, monkeypatch):
    attempt = orphan(tmp_path, monkeypatch)
    (attempt / "result.json").write_text('{"tampered":true}')
    with pytest.raises(ValueError, match="checksum"):
        checkpoints.Checkpoints(tmp_path / "tasks", "spec").get({"task": 1}, lambda _: pytest.fail("cannot hide corruption by refitting"))
    assert not (attempt.parent / "SUCCESS.json").exists()


def test_resigned_orphan_with_wrong_task_contract_is_not_adopted(tmp_path, monkeypatch):
    attempt = orphan(tmp_path, monkeypatch)
    start = json.loads((attempt / "STARTED.json").read_text())
    start["task"] = {"task": 999}
    (attempt / "STARTED.json").write_text(json.dumps(start))
    complete = json.loads((attempt / "COMPLETE.json").read_text())
    complete["STARTED.json"] = hashlib.sha256((attempt / "STARTED.json").read_bytes()).hexdigest()
    (attempt / "COMPLETE.json").write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="contract"):
        checkpoints.Checkpoints(tmp_path / "tasks", "spec").get({"task": 1}, lambda _: pytest.fail("wrong contract cannot refit"))


def test_verifier_binds_all_pointers_specification_keys_and_counts_partial_tasks(tmp_path):
    tasks = tmp_path / "tasks"
    store = checkpoints.Checkpoints(tasks, "spec")
    first = store.get({"task": 1}, computed)
    store.get({"task": 2}, computed)
    incomplete = tasks / digest({"version": checkpoints.VERSION, "specification_id": "spec", "task": {"task": 3}})
    incomplete.mkdir()
    result = checkpoints.verify_committed_tasks(tasks, "spec")
    assert result["committed_tasks"] == 2 and result["incomplete_tasks"] == [incomplete.name]
    with pytest.raises(ValueError, match="contract"):
        checkpoints.verify_committed_tasks(tasks, "other-spec")
    directory = tasks / first["checkpoint_id"]
    directory.rename(tasks / ("f" * 64))
    with pytest.raises(ValueError, match="key"):
        checkpoints.verify_committed_tasks(tasks, "spec")


@pytest.mark.parametrize("change", ["pointer_digest", "pointer_path", "attempt_symlink", "pointer_symlink", "completion_corrupt", "bad_predictions"])
def test_verifier_rejects_committed_integrity_failures(tmp_path, change):
    tasks = tmp_path / "tasks"
    result = checkpoints.Checkpoints(tasks, "spec").get({"task": 1}, computed)
    directory = tasks / result["checkpoint_id"]
    success = directory / "SUCCESS.json"
    pointer = json.loads(success.read_text())
    attempt = directory / pointer["attempt"]
    if change == "pointer_digest":
        pointer["completion_sha256"] = "0" * 64
        success.write_text(json.dumps(pointer))
    elif change == "pointer_path":
        pointer["attempt"] = "../outside"
        success.write_text(json.dumps(pointer))
    elif change == "attempt_symlink":
        moved = tmp_path / "moved-attempt"
        attempt.rename(moved)
        attempt.symlink_to(moved, target_is_directory=True)
    elif change == "pointer_symlink":
        moved = directory / "saved-success.json"
        success.rename(moved)
        success.symlink_to(moved)
    elif change == "completion_corrupt":
        (attempt / "COMPLETE.json").write_text("{}")
    elif change == "bad_predictions":
        np.savez_compressed(attempt / "predictions.npz", predictions=np.array([-1.]), raw_predictions=np.array([-1.]))
        manifest = json.loads((attempt / "COMPLETE.json").read_text())
        manifest["predictions.npz"] = hashlib.sha256((attempt / "predictions.npz").read_bytes()).hexdigest()
        (attempt / "COMPLETE.json").write_text(json.dumps(manifest))
        pointer["completion_sha256"] = digest(manifest)
        success.write_text(json.dumps(pointer))
    with pytest.raises(ValueError):
        checkpoints.verify_committed_tasks(tasks, "spec")
