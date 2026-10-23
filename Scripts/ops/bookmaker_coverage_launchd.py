#!/usr/bin/env python3
"""Run the bookmaker coverage watch four times a day (tracker open item, after the 16-18 Oct drop).

``Scripts.ops.bookmaker_coverage`` reads the odds archive only; an alert goes to the private Discord webhook in
``REFRESH_ALERT_WEBHOOK_URL`` when set, and always to its log and ``Index/bookmaker_coverage/latest.json``. Times
missed while the Mac sleeps run on wake (launchd calendar intervals).
"""
import argparse
import json
import os
from pathlib import Path
import plistlib
import sys

try:
    from .match_read_launchd import build_paths, _assert_installable, _run_launchctl, _write_atomically
except ImportError:
    from match_read_launchd import build_paths, _assert_installable, _run_launchctl, _write_atomically

LABEL = "com.bettingrag.bookmaker-coverage"
SCHEDULE = [{"Hour": hour, "Minute": 45} for hour in (9, 13, 17, 21)]


def render(paths):
    return plistlib.dumps({
        "Label": LABEL,
        "ProgramArguments": [str(paths.python_executable), "-B", "-m", "Scripts.ops.bookmaker_coverage"],
        "WorkingDirectory": str(paths.project_root), "StartCalendarInterval": SCHEDULE,
        "ProcessType": "Background",
        "StandardOutPath": str(paths.log_directory / "bookmaker-coverage.stdout.log"),
        "StandardErrorPath": str(paths.log_directory / "bookmaker-coverage.stderr.log"),
    }, sort_keys=False).decode()


def operate(action, paths, *, dry_run=False):
    target, domain = f"gui/{os.getuid()}/{LABEL}", f"gui/{os.getuid()}"
    if action == "render" or dry_run:
        return {"action": action, "dry_run": dry_run, "configuration": plistlib.loads(render(paths).encode())}
    if action == "status":
        response = _run_launchctl(["print", target], check=False)
        return {"installed": paths.plist_path.is_file(), "loaded": response.returncode == 0, "label": LABEL}
    _assert_installable(paths)
    if paths.plist_path.exists() and plistlib.loads(paths.plist_path.read_bytes()).get("Label") != LABEL:
        raise RuntimeError("Refusing to replace an unrelated launch agent")
    if action == "install":
        paths.log_directory.mkdir(parents=True, exist_ok=True)
        _write_atomically(paths.plist_path, render(paths))
        _run_launchctl(["bootout", target], check=False)
        _run_launchctl(["bootstrap", domain, str(paths.plist_path)], check=True)
        return {"installed": True, "label": LABEL, "schedule": SCHEDULE}
    _run_launchctl(["bootout", target], check=False)
    return {"stopped": True, "files_retained": True, "label": LABEL}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "status", "render", "stop"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args(argv)
    paths = build_paths(root=args.project_root, plist_path=Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist")
    try:
        report = operate(args.action, paths, dry_run=args.dry_run)
        print(json.dumps(report, indent=2))
        return int(args.action == "status" and not report.get("loaded", False))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"bookmaker-coverage scheduler: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
