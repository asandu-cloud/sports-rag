"""Rework A0: the current engine must reproduce the recorded golden bundle exactly.

The check runs in a fresh process, as the Match Read worker does: other tests
patch and warm engine module state, which would otherwise leak into it.

When a change is intended, read the summary, then re-record:
    python -m Scripts.ops.golden_snapshots record --date <fixture date> [--date ...]
"""
import os
import subprocess
import sys

import pytest

from Scripts.ops import golden_snapshots as golden

pytestmark = pytest.mark.skipif(not golden.BUNDLE.exists(), reason="no golden bundle recorded on this machine")


def test_engine_reproduces_the_golden_bundle():
    env = dict(os.environ, ANONYMIZED_TELEMETRY="False")
    run = subprocess.run([sys.executable, "-m", "Scripts.ops.golden_snapshots", "check"], cwd=golden.ROOT,
                         env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, "engine output changed:\n" + run.stdout[-6000:] + run.stderr[-2000:]
