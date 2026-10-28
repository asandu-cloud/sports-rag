import json
import logging
import os
from unittest.mock import Mock

import pytest
import requests

from data_platform.services import refresh_alerts
from data_platform.services.refresh_alerts import blockage_status, observe_gate

DESTINATIONS = ("REFRESH_ALERT_WEBHOOK_URL", "REFRESH_ALERT_TELEGRAM_TOKEN", "REFRESH_ALERT_TELEGRAM_CHAT_ID")


class Answer:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, {"ok": True} if body is None else body

    def json(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def telegram(monkeypatch, *, token="123:abc", chat="4567"):
    for name in DESTINATIONS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("REFRESH_ALERT_TELEGRAM_TOKEN", token)
    monkeypatch.setenv("REFRESH_ALERT_TELEGRAM_CHAT_ID", chat)


def test_alerts_go_to_the_telegram_chat_when_configured(monkeypatch):
    telegram(monkeypatch)
    posts = []
    monkeypatch.setattr(requests, "post", lambda url, json, timeout: posts.append((url, json)) or Answer())
    assert refresh_alerts._send("Spix refresh blocked for 61 minutes.") is True
    assert posts == [("https://api.telegram.org/bot123:abc/sendMessage",
                      {"chat_id": "4567", "text": "Spix refresh blocked for 61 minutes.",
                       "disable_web_page_preview": True})]


@pytest.mark.parametrize("answer", [Answer(200, {"ok": False}), Answer(401, {"ok": False, "description": "Unauthorized"}),
                                    requests.ConnectionError("https://api.telegram.org/bot123:abc/sendMessage")])
def test_a_failed_telegram_delivery_is_false_and_never_logs_the_token(monkeypatch, caplog, answer):
    telegram(monkeypatch)

    def post(url, json, timeout):
        if isinstance(answer, Exception):
            raise answer
        return answer
    monkeypatch.setattr(requests, "post", post)
    with caplog.at_level(logging.INFO):
        assert refresh_alerts._send("Spix refresh blocked for 61 minutes.") is False
    assert "123:abc" not in caplog.text


def test_without_a_destination_nothing_is_sent(monkeypatch):
    for name in DESTINATIONS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(requests, "post", lambda *a, **k: 1 / 0)
    assert refresh_alerts._send("x") is False and not refresh_alerts.configured()
    telegram(monkeypatch, chat="")  # a bot token alone has nowhere to send
    assert not refresh_alerts.configured()


def test_the_blockage_status_counts_telegram_as_a_destination(tmp_path, monkeypatch):
    telegram(monkeypatch)
    assert blockage_status(tmp_path / "marker")["webhook_configured"]


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
