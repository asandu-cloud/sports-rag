import json
import os
from unittest.mock import Mock

from data_platform.services.refresh_alerts import blockage_status, observe_gate


def test_prolonged_marker_alert_is_deduplicated_and_recovers(tmp_path, monkeypatch):
    monkeypatch.setenv("REFRESH_ALERT_AFTER_SECONDS", "3600")
    marker = tmp_path / "db.refresh-incomplete"
    marker.write_text("interrupted")
    os.utime(marker, (10000, 10000))
    sender = Mock(return_value=True)
    observe_gate(marker, blocked=True, now=14000, sender=sender)
    observe_gate(marker, blocked=True, now=14500, sender=sender)
    assert sender.call_count == 1
    assert blockage_status(marker, now=14500)["prolonged"]
    marker.unlink()
    observe_gate(marker, blocked=False, now=15000, sender=sender)
    assert sender.call_count == 2
    assert "recovered" in sender.call_args.args[0]
    assert not blockage_status(marker, now=15000)["blocked"]


def test_failed_webhook_leaves_local_alert_and_retries_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("REFRESH_ALERT_AFTER_SECONDS", "10")
    marker = tmp_path / "marker"
    sender = Mock(return_value=False)
    observe_gate(marker, blocked=True, now=10000, sender=sender)
    observe_gate(marker, blocked=True, now=10020, sender=sender)
    observe_gate(marker, blocked=True, now=10030, sender=sender)
    assert sender.call_count == 1
    state = json.loads((tmp_path / "marker.alert.json").read_text())
    assert state["message"] and not state["notification_delivered"]
    observe_gate(marker, blocked=True, now=10321, sender=sender)
    assert sender.call_count == 2


def test_pending_recovery_is_retried_without_marking_gate_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv("REFRESH_ALERT_AFTER_SECONDS", "10")
    marker = tmp_path / "marker"
    sender = Mock(side_effect=[True, False, True])
    observe_gate(marker, blocked=True, now=10000, sender=sender)
    observe_gate(marker, blocked=True, now=10020, sender=sender)
    observe_gate(marker, blocked=False, now=10100, sender=sender)
    assert not blockage_status(marker, now=10100)["blocked"]
    observe_gate(marker, blocked=False, now=10500, sender=sender)
    assert sender.call_count == 3


def test_retry_does_not_reset_incomplete_marker_age(settings, engine):
    from data_platform.services.refresh_coordination import _paths, data_access
    marker = _paths()[1]
    marker.write_text("old")
    os.utime(marker, (10000, 10000))
    try:
        with data_access(writer=True, full_refresh=True):
            assert marker.stat().st_mtime == 10000
            raise RuntimeError("failed")
    except RuntimeError:
        pass
    assert marker.stat().st_mtime == 10000


def test_kb_cli_fails_on_processing_errors_or_unembedded_docs(monkeypatch):
    from types import SimpleNamespace
    from data_platform import cli, kb
    for result in ({"error_count": 1, "pending_embed": 0}, {"error_count": 0, "pending_embed": 1}):
        monkeypatch.setattr(kb, "refresh_kb", lambda **kw: result)
        assert cli.cmd_kb_refresh(SimpleNamespace(dry_run=False, batch_size=50)) == 1
