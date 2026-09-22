"""Local cross-process gate between refresh writes and fixture generation.

Use a separate advisory lock, not a database transaction, while running model
or provider work. This gate is for the current single-host deployment; it is
not a distributed lock for multiple cloud machines.
"""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import time
from uuid import uuid4

from sqlalchemy.engine import make_url


class RefreshBusy(RuntimeError):
    pass


def _paths():
    from .. import config
    url = make_url(config.SETTINGS.database_url)
    if url.get_backend_name() != "sqlite" or not url.database or url.database == ":memory:":
        raise RefreshBusy("Local refresh coordination requires a file-backed SQLite database.")
    database = Path(url.database).resolve()
    return Path(str(database) + ".refresh.lock"), Path(str(database) + ".refresh-incomplete")


def refresh_generation():
    lock_path, _ = _paths()
    try:
        return Path(str(lock_path) + ".generation").read_text()
    except FileNotFoundError:
        return "initial"


@contextmanager
def _locked_file(path, mode, deadline):
    with path.open("a+") as handle:
        while True:
            try:
                fcntl.flock(handle, mode | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RefreshBusy("Data refresh or fixture generation is active; retry on the next tick.")
                time.sleep(0.1)
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def data_access(*, writer=False, full_refresh=False, wait_seconds=0):
    lock_path, incomplete = _paths()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # refresh-season passes its descriptor explicitly to its child commands.
    # Only a descriptor for this exact lock file is accepted as inherited.
    inherited = os.environ.get("BETTING_REFRESH_LOCK_FD")
    inherited_valid = False
    if inherited is not None:
        try:
            descriptor = int(inherited)
            inherited_stat = os.fstat(descriptor)
            disk_stat = lock_path.stat()
            if (inherited_stat.st_dev, inherited_stat.st_ino) == (disk_stat.st_dev, disk_stat.st_ino):
                inherited_valid = True
        except (OSError, ValueError):
            pass
    if inherited_valid:
        yield
        return
    deadline = time.monotonic() + wait_seconds
    mode = fcntl.LOCK_EX if writer else fcntl.LOCK_SH
    # A waiting writer closes the entrance to new readers. Otherwise a busy
    # worker could repeatedly reacquire its shared gate between fixtures and
    # starve the automatic upstream refresh indefinitely.
    intent = Path(str(lock_path) + ".intent")
    with _locked_file(intent, mode, deadline) as entrance, _locked_file(lock_path, mode, deadline) as handle:
        if not writer:
            fcntl.flock(entrance, fcntl.LOCK_UN)
        previous = os.environ.get("BETTING_REFRESH_LOCK_FD")
        try:
            if not writer and incomplete.exists():
                raise RefreshBusy("A full season refresh did not complete; rerun refresh-season before generating cards.")
            if writer:
                os.environ["BETTING_REFRESH_LOCK_FD"] = str(handle.fileno())
                Path(str(lock_path) + ".generation").write_text(uuid4().hex)
            if full_refresh:
                incomplete.write_text("Full refresh in progress or interrupted; rerun refresh-season.\n")
            yield
            if full_refresh:
                incomplete.unlink(missing_ok=True)
        finally:
            if writer:
                if previous is None:
                    os.environ.pop("BETTING_REFRESH_LOCK_FD", None)
                else:
                    os.environ["BETTING_REFRESH_LOCK_FD"] = previous
