import signal

import pytest

from data_platform.services.worker_runtime import DispatchTimeout, dispatch_deadline


def test_deadline_escapes_provider_fallback_and_restores_signal():
    handler = signal.getsignal(signal.SIGALRM)
    with pytest.raises(DispatchTimeout):
        with dispatch_deadline(2):
            try:
                signal.raise_signal(signal.SIGALRM)
            except Exception:
                pytest.fail("Provider fallback swallowed the worker deadline")
    assert signal.getsignal(signal.SIGALRM) == handler
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


def test_success_cancels_deadline():
    with dispatch_deadline(2):
        pass
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


def test_deadline_interrupts_a_blocked_call():
    import threading

    with pytest.raises(DispatchTimeout):
        with dispatch_deadline(0.02):
            threading.Event().wait(timeout=1)
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


def test_idle_sleep_assertion_is_scoped_to_cycle(monkeypatch):
    from unittest.mock import Mock
    from data_platform.services import worker_runtime

    process = Mock()
    process.poll.return_value = None
    popen = Mock(return_value=process)
    monkeypatch.setattr(worker_runtime.sys, "platform", "darwin")
    monkeypatch.setattr(worker_runtime.os.path, "isfile", lambda _: True)
    monkeypatch.setattr(worker_runtime.subprocess, "Popen", popen)
    with pytest.raises(ValueError):
        with worker_runtime.prevent_idle_sleep():
            raise ValueError("cycle failed")
    assert popen.call_args.args[0][:3] == ["/usr/bin/caffeinate", "-i", "-w"]
    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(timeout=5)
