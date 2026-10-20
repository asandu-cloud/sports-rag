#!/usr/bin/env python3
"""Capture OddsPapi true closes every day at 10:30 local time (rework A1 step 4, D18).

After the 07:15 data refresh; yesterday's matches are over by then. A refresh still running defers the
run to the next day (the capture only writes ``extras["true_close"]``, so a skipped day is caught up).
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

LABEL = "com.bettingrag.true-close"
SCHEDULE = {"Hour": 10, "Minute": 30}  # daily, after the 07:15 refresh


def render(paths):
    return plistlib.dumps({
        "Label": LABEL, "ProgramArguments": [str(paths.python_executable), "-B", "-m", "Scripts.ops.true_close"],
        "WorkingDirectory": str(paths.project_root), "StartCalendarInterval": SCHEDULE,
        "ProcessType": "Background",
        "StandardOutPath": str(paths.log_directory / "true-close.stdout.log"),
        "StandardErrorPath": str(paths.log_directory / "true-close.stderr.log"),
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
        print(f"true-close scheduler: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
