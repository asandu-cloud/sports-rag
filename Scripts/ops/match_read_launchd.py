#!/usr/bin/env python3
"""Install and operate the local macOS Match Read launchd schedule.

This is intentionally only an operational wrapper around the existing
one-shot worker.  It neither imports Match Read code nor sources ``.env``.
The generated plist contains only paths, the selected safe mode, and logging
configuration; the application continues to load its normal project-level
configuration when it starts in the configured working directory.

Examples
--------
    ./.venv/bin/python Scripts/ops/match_read_launchd.py install
    ./.venv/bin/python Scripts/ops/match_read_launchd.py install --mode website
    ./.venv/bin/python Scripts/ops/match_read_launchd.py status
    ./.venv/bin/python Scripts/ops/match_read_launchd.py uninstall
"""

from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape as xml_escape


WORKER_LABEL = "com.bettingrag.match-read-worker"
VALID_MODES = ("shadow", "website")
INTERVAL_SECONDS = 600
_TEMPLATE_PATH = Path(__file__).with_name("match_read_worker.plist.template")


@dataclass(frozen=True)
class LaunchdPaths:
    """Resolved local paths used by the generated launchd service."""

    project_root: Path
    python_executable: Path
    log_directory: Path
    plist_path: Path

    @property
    def stdout_log(self) -> Path:
        return self.log_directory / "match-read-worker.stdout.log"

    @property
    def stderr_log(self) -> Path:
        return self.log_directory / "match-read-worker.stderr.log"


def project_root() -> Path:
    """Return this checkout's root without relying on the caller's CWD."""
    return Path(__file__).resolve().parents[2]


def normalise_mode(mode: str) -> str:
    value = str(mode or "").strip().lower()
    if value not in VALID_MODES:
        choices = ", ".join(VALID_MODES)
        raise ValueError(f"--mode must be one of: {choices}")
    return value


def build_paths(
    *,
    root: Path | str | None = None,
    python_executable: Path | str | None = None,
    plist_path: Path | str | None = None,
) -> LaunchdPaths:
    """Build defaults without creating files or touching launchd."""
    resolved_root = Path(root or project_root()).expanduser().resolve()
    # Do *not* resolve this path: a virtual environment's ``bin/python`` is
    # normally a symlink.  Launching its resolved target would bypass the venv
    # and could silently run a system Python without this project's packages.
    configured_python = Path(
        python_executable or (resolved_root / ".venv" / "bin" / "python")
    ).expanduser()
    resolved_python = configured_python.absolute()
    resolved_plist = Path(
        plist_path or (Path.home() / "Library" / "LaunchAgents" / f"{WORKER_LABEL}.plist")
    ).expanduser().resolve()
    return LaunchdPaths(
        project_root=resolved_root,
        python_executable=resolved_python,
        log_directory=resolved_root / "Index" / "logs",
        plist_path=resolved_plist,
    )


def render_plist(*, paths: LaunchdPaths, mode: str) -> str:
    """Render a valid plist without writing it or reading environment secrets."""
    mode = normalise_mode(mode)
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    replacements = {
        "__LABEL__": WORKER_LABEL,
        "__PYTHON_EXECUTABLE__": str(paths.python_executable),
        "__PROJECT_ROOT__": str(paths.project_root),
        "__MODE__": mode,
        "__STDOUT_LOG__": str(paths.stdout_log),
        "__STDERR_LOG__": str(paths.stderr_log),
    }
    rendered = template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, xml_escape(value))

    unresolved = [token for token in replacements if token in rendered]
    if unresolved:
        raise RuntimeError(f"Launchd template has unresolved placeholders: {unresolved}")

    # This is a portable validation step and catches malformed templates before
    # an install can leave an invalid service file in LaunchAgents.
    parsed = plistlib.loads(rendered.encode("utf-8"))
    if parsed.get("Label") != WORKER_LABEL:
        raise RuntimeError("Rendered launchd plist has an unexpected label")
    if parsed.get("StartInterval") != INTERVAL_SECONDS:
        raise RuntimeError("Rendered launchd plist has an unexpected interval")
    return rendered


def _assert_installable(paths: LaunchdPaths) -> None:
    if sys.platform != "darwin":
        raise RuntimeError("The Match Read launchd helper can only install on macOS.")
    if not paths.project_root.is_dir():
        raise RuntimeError(f"Project root does not exist: {paths.project_root}")
    module_entry = paths.project_root / "Scripts" / "data_platform" / "__main__.py"
    if not module_entry.is_file():
        raise RuntimeError(
            f"{paths.project_root} does not look like a Betting RAG checkout "
            "(Scripts/data_platform/__main__.py is missing)."
        )
    if not paths.python_executable.is_file() or not os.access(paths.python_executable, os.X_OK):
        raise RuntimeError(
            "Expected the project virtual-environment interpreter at "
            f"{paths.python_executable}. Create .venv first or pass --python-executable."
        )


def _launchd_domain() -> str:
    return f"gui/{os.getuid()}"


def _service_target() -> str:
    return f"{_launchd_domain()}/{WORKER_LABEL}"


def _run_launchctl(arguments: Iterable[str], *, check: bool) -> subprocess.CompletedProcess[str]:
    command = ["launchctl", *arguments]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if check and result.returncode:
        detail = (result.stderr or result.stdout or "unknown launchctl error").strip()
        raise RuntimeError(f"{' '.join(command)} failed: {detail}")
    return result


def _write_atomically(path: Path, contents: str) -> None:
    """Write a plist atomically, never exposing a partially rendered service."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(contents)
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def install(*, paths: LaunchdPaths, mode: str, dry_run: bool = False) -> dict[str, object]:
    """Install/reload the service, then start a first cycle immediately."""
    mode = normalise_mode(mode)
    rendered = render_plist(paths=paths, mode=mode)
    if dry_run:
        return {
            "action": "install",
            "dry_run": True,
            "mode": mode,
            "plist_path": str(paths.plist_path),
            "working_directory": str(paths.project_root),
            "python_executable": str(paths.python_executable),
            "log_directory": str(paths.log_directory),
        }

    _assert_installable(paths)
    paths.log_directory.mkdir(parents=True, exist_ok=True)

    # bootout is intentionally best-effort: a first install has nothing to
    # unload, while a mode change needs the old service removed before reload.
    # Try the service target first as it also catches an older plist that was
    # moved after it had already been loaded.
    _run_launchctl(["bootout", _service_target()], check=False)
    _run_launchctl(["bootout", _launchd_domain(), str(paths.plist_path)], check=False)
    _write_atomically(paths.plist_path, rendered)
    _run_launchctl(["bootstrap", _launchd_domain(), str(paths.plist_path)], check=True)
    # ``RunAtLoad`` starts the first cycle. Do not also kickstart it here: that
    # could create an avoidable overlapping first invocation.
    return {
        "action": "install",
        "mode": mode,
        "label": WORKER_LABEL,
        "interval_seconds": INTERVAL_SECONDS,
        "plist_path": str(paths.plist_path),
        "working_directory": str(paths.project_root),
        "python_executable": str(paths.python_executable),
        "log_directory": str(paths.log_directory),
        "stdout_log": str(paths.stdout_log),
        "stderr_log": str(paths.stderr_log),
        "service_target": _service_target(),
    }


def _installed_mode(plist_path: Path) -> str | None:
    if not plist_path.is_file():
        return None
    try:
        payload = plistlib.loads(plist_path.read_bytes())
    except (OSError, plistlib.InvalidFileException):
        return "invalid plist"
    arguments = payload.get("ProgramArguments") or []
    try:
        return str(arguments[arguments.index("--mode") + 1])
    except (ValueError, IndexError):
        return "unknown"


def status(*, paths: LaunchdPaths) -> tuple[dict[str, object], int]:
    """Return whether the exact expected plist and service are present."""
    report: dict[str, object] = {
        "action": "status",
        "label": WORKER_LABEL,
        "service_target": _service_target(),
        "plist_path": str(paths.plist_path),
        "plist_installed": paths.plist_path.is_file(),
        "mode": _installed_mode(paths.plist_path),
        "working_directory": str(paths.project_root),
        "log_directory": str(paths.log_directory),
        "stdout_log": str(paths.stdout_log),
        "stderr_log": str(paths.stderr_log),
    }
    if sys.platform != "darwin":
        report["loaded"] = False
        report["note"] = "launchd status is only available on macOS."
        return report, 1

    result = _run_launchctl(["print", _service_target()], check=False)
    report["loaded"] = result.returncode == 0
    if result.returncode:
        report["launchctl_message"] = (result.stderr or result.stdout).strip()
    return report, 0 if result.returncode == 0 else 1


def uninstall(*, paths: LaunchdPaths, dry_run: bool = False) -> dict[str, object]:
    """Unload this exact service and remove only its plist; logs remain."""
    if dry_run:
        return {
            "action": "uninstall",
            "dry_run": True,
            "plist_path": str(paths.plist_path),
            "logs_retained": str(paths.log_directory),
        }

    if sys.platform != "darwin":
        raise RuntimeError("The Match Read launchd helper can only uninstall on macOS.")
    _run_launchctl(["bootout", _launchd_domain(), str(paths.plist_path)], check=False)
    removed = False
    if paths.plist_path.is_file():
        paths.plist_path.unlink()
        removed = True
    return {
        "action": "uninstall",
        "label": WORKER_LABEL,
        "plist_path": str(paths.plist_path),
        "plist_removed": removed,
        "logs_retained": str(paths.log_directory),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install/status/uninstall the local Match Read launchd service."
    )
    parser.add_argument("action", choices=("install", "status", "uninstall", "render"))
    parser.add_argument(
        "--mode", choices=VALID_MODES, default="shadow",
        help="Worker release mode when installing/rendering (default: shadow).",
    )
    parser.add_argument(
        "--project-root", type=Path, default=None,
        help="Checkout root (default: infer from this helper).",
    )
    parser.add_argument(
        "--python-executable", type=Path, default=None,
        help="Interpreter to place in ProgramArguments (default: <root>/.venv/bin/python).",
    )
    parser.add_argument(
        "--plist-path", type=Path, default=None,
        help="LaunchAgent plist path (default: ~/Library/LaunchAgents/<label>.plist).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="For install/uninstall, show the intended change without touching launchd or files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = build_paths(
        root=args.project_root,
        python_executable=args.python_executable,
        plist_path=args.plist_path,
    )
    try:
        if args.action == "render":
            print(render_plist(paths=paths, mode=args.mode), end="")
            return 0
        if args.action == "install":
            report = install(paths=paths, mode=args.mode, dry_run=args.dry_run)
            print(_pretty_report(report))
            return 0
        if args.action == "status":
            report, exit_code = status(paths=paths)
            print(_pretty_report(report))
            return exit_code
        report = uninstall(paths=paths, dry_run=args.dry_run)
        print(_pretty_report(report))
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"match-read launchd helper: {exc}", file=sys.stderr)
        return 2


def _pretty_report(report: dict[str, object]) -> str:
    """Avoid a JSON dependency while keeping shell output easy to scan."""
    return "\n".join(f"{key}: {value}" for key, value in report.items())


if __name__ == "__main__":
    raise SystemExit(main())
