#!/usr/bin/env python3
"""Operate the local prospective measurement job without touching other schedules."""
import argparse
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys

try:
    from .match_read_launchd import build_paths, _assert_installable, _run_launchctl, _write_atomically
except ImportError:
    from match_read_launchd import build_paths, _assert_installable, _run_launchctl, _write_atomically

LABEL = "com.bettingrag.prediction-measurement"


def render(paths):
    return plistlib.dumps({
        "Label": LABEL, "ProgramArguments": [str(paths.python_executable), "-m", "Scripts.data_platform", "measurement-cycle", "--once"],
        "WorkingDirectory": str(paths.project_root), "StartInterval": 60,
        "RunAtLoad": True, "ProcessType": "Background",
        "StandardOutPath": str(paths.log_directory / "prediction-measurement.stdout.log"),
        "StandardErrorPath": str(paths.log_directory / "prediction-measurement.stderr.log"),
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
        # Establish the durable prospective boundary before the first tick.
        result = subprocess.run([str(paths.python_executable), "-m", "Scripts.data_platform", "measurement-enable"],
                                cwd=paths.project_root, check=False, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError("Could not enable measurement; no launch agent installed")
        paths.log_directory.mkdir(parents=True, exist_ok=True)
        _write_atomically(paths.plist_path, render(paths))
        _run_launchctl(["bootout", target], check=False)
        try:
            _run_launchctl(["bootstrap", domain, str(paths.plist_path)], check=True)
        except Exception:
            # Fail closed: a failed install must not leave refresh hooks enabled.
            subprocess.run([str(paths.python_executable), "-m", "Scripts.data_platform", "measurement-disable"],
                           cwd=paths.project_root, check=False, capture_output=True, text=True)
            raise
        return {"installed": True, "label": LABEL, "interval_seconds": 60, "activation": json.loads(result.stdout)}
    _run_launchctl(["bootout", target], check=False)
    # Keep the plist/logs/evidence recoverable. Pausing also blocks refresh hooks.
    result = subprocess.run([str(paths.python_executable), "-m", "Scripts.data_platform", "measurement-disable"],
                            cwd=paths.project_root, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Launch agent stopped, but measurement-disable failed; check status")
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
        print(f"measurement scheduler: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
