"""Refresh subprocess deadlines, including cleanup of a stage's descendants."""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time


def positive_seconds(name, default):
    value = float(os.environ.get(name, default))
    if not 0 < value < float("inf"):
        raise ValueError(f"{name} must be a positive finite number of seconds")
    return value


def run_bounded(command, *, cwd, timeout, check=False, pass_fds=()):
    """Isolate the outer stage; nested stages share that process group.

    A timeout, SIGTERM or Ctrl-C reaps the child and kills surviving stage
    descendants *before* the caller releases its refresh gate. No unrelated
    process groups or lock files are touched.
    """
    nested = os.environ.get("BETTING_REFRESH_STAGE_GROUP") == str(os.getpgrp())
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    # The child learns its actual group from getpgrp(); descendants can then
    # inherit that exact identity (not an unvalidated boolean bypass).
    bootstrap = "import os,sys; os.environ['BETTING_REFRESH_STAGE_GROUP']=str(os.getpgrp()); os.execvpe(sys.argv[1],sys.argv[1:],os.environ)"
    import sys
    argv = list(command) if nested else [sys.executable, "-c", bootstrap, *command]
    previous = None
    def interrupted(signum, frame):
        raise InterruptedError("Refresh stage interrupted by SIGTERM")
    proc = subprocess.Popen(argv, cwd=cwd, env=env, pass_fds=pass_fds, start_new_session=not nested)
    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        return_code = proc.wait(timeout=timeout)
        result = subprocess.CompletedProcess(command, return_code)
        if check:
            result.check_returncode()
        return result
    finally:
        try:
            if nested:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                proc.wait()
            else:
                # Cleanup even after an unsuccessful child exits on its own:
                # it may have left a grandchild holding an inherited lock FD.
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    deadline = time.monotonic() + 2
                    while time.monotonic() < deadline:
                        proc.poll()
                        try:
                            os.killpg(proc.pid, 0)
                        except ProcessLookupError:
                            break
                        except PermissionError:
                            # Darwin can briefly report EPERM for a group
                            # containing only reparenting/zombie descendants.
                            # Keep waiting; never treat EPERM as cleanup done.
                            pass
                        time.sleep(0.05)
                    else:
                        os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        finally:
            if previous is not None:
                signal.signal(signal.SIGTERM, previous)
