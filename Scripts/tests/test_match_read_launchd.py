from __future__ import annotations

import plistlib
from pathlib import Path

from ops.match_read_launchd import (
    INTERVAL_SECONDS,
    WORKER_LABEL,
    build_paths,
    normalise_mode,
    render_plist,
)


def test_rendered_match_read_plist_is_valid_and_self_contained(tmp_path):
    root = tmp_path / "Betting & RAG"
    paths = build_paths(
        root=root,
        python_executable=root / ".venv" / "bin" / "python",
        plist_path=tmp_path / "agent.plist",
    )

    rendered = render_plist(paths=paths, mode="website")
    payload = plistlib.loads(rendered.encode("utf-8"))

    assert "__" not in rendered
    assert payload["Label"] == WORKER_LABEL
    assert payload["WorkingDirectory"] == str(root.resolve())
    assert payload["StartInterval"] == INTERVAL_SECONDS == 600
    assert payload["RunAtLoad"] is True
    assert payload["ProgramArguments"] == [
        str(root / ".venv" / "bin" / "python"),
        "-m",
        "Scripts.data_platform",
        "match-read-cycle",
        "--once",
        "--mode",
        "website",
    ]
    assert payload["StandardOutPath"].endswith("Index/logs/match-read-worker.stdout.log")
    assert payload["StandardErrorPath"].endswith("Index/logs/match-read-worker.stderr.log")
    assert "EnvironmentVariables" not in payload
    assert "API_FOOTBALL_KEY" not in rendered
    assert "OPENAI_API_KEY" not in rendered


def test_mode_validation_fails_closed_to_explicit_choices():
    assert normalise_mode(" SHADOW ") == "shadow"
    assert normalise_mode("website") == "website"
    try:
        normalise_mode("discord")
    except ValueError as exc:
        assert "shadow, website" in str(exc)
    else:  # pragma: no cover - makes the intended failure explicit
        raise AssertionError("invalid mode should be rejected")
