"""Operate only the independent, local pre-match odds archive schedule."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import plistlib

from Scripts.ops.match_read_launchd import build_paths, _assert_installable, _run_launchctl, _write_atomically
from Scripts.ops.odds_archive import INTERVAL

LABEL = "com.bettingrag.odds-archive"


def render(paths):
    return plistlib.dumps({
        "Label": LABEL,
        "ProgramArguments": [str(paths.python_executable), "-B", "-m", "Scripts.ops.odds_archive", "--execute"],
        "WorkingDirectory": str(paths.project_root), "StartInterval": INTERVAL,
        "RunAtLoad": True, "ProcessType": "Background",
        "StandardOutPath": str(paths.log_directory / "odds-archive.stdout.log"),
        "StandardErrorPath": str(paths.log_directory / "odds-archive.stderr.log"),
    }, sort_keys=False).decode()


def operate(action, paths, *, dry_run=False):
    target, domain = f"gui/{os.getuid()}/{LABEL}", f"gui/{os.getuid()}"
    if action == "render" or dry_run:
        return {"action": action, "configuration": plistlib.loads(render(paths).encode())}
    if action == "status":
        result = _run_launchctl(["print", target], check=False)
        details = {}
        for line in result.stdout.splitlines():
            key, sep, value = line.strip().partition(" = ")
            if sep and key in ("state", "pid", "runs", "last exit code"):
                details[key] = value
        return {"installed": paths.plist_path.is_file(), "loaded": result.returncode == 0,
                "label": LABEL, "process": details}
    _assert_installable(paths)
    if paths.plist_path.exists() and plistlib.loads(paths.plist_path.read_bytes()).get("Label") != LABEL:
        raise RuntimeError("Refusing to replace an unrelated launch agent")
    if action == "stop":
        result = _run_launchctl(["bootout", target], check=False)
        remaining = _run_launchctl(["print", target], check=False)
        if remaining.returncode == 0:
            raise RuntimeError("Archive job still loaded")
        # Removing just this schedule prevents automatic restart at next login.
        # The database, raw evidence and logs remain intact.
        paths.plist_path.unlink(missing_ok=True)
        return {"stopped": True, "archive_and_logs_retained": True, "label": LABEL}
    if action != "install":
        raise ValueError("Unknown action")
    if paths.plist_path.exists():
        if paths.plist_path.read_text() != render(paths):
            raise RuntimeError("Existing archive configuration differs; stop and review before replacing")
        if _run_launchctl(["print", target], check=False).returncode == 0:
            return {"installed": True, "loaded": True, "unchanged": True, "label": LABEL}
    paths.log_directory.mkdir(parents=True, exist_ok=True)
    _write_atomically(paths.plist_path, render(paths))
    _run_launchctl(["bootstrap", domain, str(paths.plist_path)], check=True)
    return {"installed": True, "loaded": True, "label": LABEL, "interval_seconds": INTERVAL}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("render", "install", "status", "stop"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    paths = build_paths(plist_path=Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist")
    try:
        result = operate(args.action, paths, dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
        return int(args.action == "status" and not result.get("loaded"))
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
