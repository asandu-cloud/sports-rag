"""Bound synchronous worker calls and prevent idle sleep during local cycles."""

from contextlib import contextmanager
import os
import signal
import subprocess
import sys
import threading


class DispatchTimeout(BaseException):
    """Escape provider fallbacks which intentionally catch ordinary Exceptions."""


@contextmanager
def dispatch_deadline(seconds: float):
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        raise RuntimeError("Bounded Match Read dispatch requires a POSIX main-thread worker")
    previous_handler = signal.getsignal(signal.SIGALRM)
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise RuntimeError("Cannot replace an existing worker alarm")

    def expire(_signum, _frame):
        raise DispatchTimeout(f"Fixture dispatch exceeded {seconds:.0f} seconds")

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


@contextmanager
def prevent_idle_sleep():
    """An assertion lasts only while this command runs; no power settings change."""
    process = None
    if sys.platform == "darwin" and os.path.isfile("/usr/bin/caffeinate"):
        try:
            process = subprocess.Popen(
                ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except OSError:
            pass
    try:
        yield
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
