#!/usr/bin/env python3
"""Install the two local upstream schedules without changing card publication.

Schedule check: hourly, provider data only when the six-hour coverage expires.
Joined refresh: daily at 07:15 local time, using the existing additive workflow.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import plistlib
import sys

try:
    from .match_read_launchd import build_paths, _assert_installable, _run_launchctl, _write_atomically
except ImportError:
    from match_read_launchd import build_paths, _assert_installable, _run_launchctl, _write_atomically


@dataclass(frozen=True)
class Job:
    name: str
    label: str
    arguments: tuple[str, ...]
    interval_seconds: int | None = None


JOBS = {
    "schedule": Job("fixture-schedule", "com.bettingrag.fixture-schedule",
                    ("sync-schedule", "--days-ahead", "35", "--if-stale-hours", "6"), 3600),
    "data": Job("daily-data-refresh", "com.bettingrag.daily-data-refresh",
                ("refresh-season", "--include-europe")),
}


def paths_for(job, root=None):
    return build_paths(root=root, plist_path=Path.home() / "Library/LaunchAgents" / f"{job.label}.plist")


def render(job, paths):
    payload = {
        "Label": job.label,
        "ProgramArguments": [str(paths.python_executable), "-m", "Scripts.data_platform", *job.arguments],
        "WorkingDirectory": str(paths.project_root),
        "RunAtLoad": job.interval_seconds is not None,
        "ProcessType": "Background",
        "StandardOutPath": str(paths.log_directory / f"{job.name}.stdout.log"),
        "StandardErrorPath": str(paths.log_directory / f"{job.name}.stderr.log"),
    }
    if job.interval_seconds is not None:
        payload["StartInterval"] = job.interval_seconds
    else:
        payload["StartCalendarInterval"] = {"Hour": 7, "Minute": 15}
    return plistlib.dumps(payload, sort_keys=False).decode()


def operate(action, job, paths, *, dry_run=False):
    domain = f"gui/{os.getuid()}"
    target = f"{domain}/{job.label}"
    report = {"action": action, "job": job.name, "label": job.label,
              "plist": str(paths.plist_path), "dry_run": dry_run}
    if action == "render":
        return plistlib.loads(render(job, paths).encode())
    if action == "status":
        result = _run_launchctl(["print", target], check=False)
        report.update(installed=paths.plist_path.is_file(), loaded=result.returncode == 0)
        return report
    if dry_run:
        return {**report, "configuration": plistlib.loads(render(job, paths).encode())}
    _assert_installable(paths)
    # Touch only these upstream labels, never the Match Read worker.
    if paths.plist_path.exists():
        payload = plistlib.loads(paths.plist_path.read_bytes())
        if payload.get("Label") != job.label:
            raise RuntimeError(f"Refusing to replace an unrelated agent at {paths.plist_path}")
    _run_launchctl(["bootout", target], check=False)
    if action == "uninstall":
        paths.plist_path.unlink(missing_ok=True)
        return {**report, "logs_retained": True}
    paths.log_directory.mkdir(parents=True, exist_ok=True)
    _write_atomically(paths.plist_path, render(job, paths))
    _run_launchctl(["bootstrap", domain, str(paths.plist_path)], check=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "status", "render", "uninstall"))
    parser.add_argument("--job", choices=("all", *JOBS), default="all")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    jobs = JOBS.values() if args.job == "all" else (JOBS[args.job],)
    try:
        reports = [operate(args.action, job, paths_for(job, args.project_root), dry_run=args.dry_run) for job in jobs]
        print(json.dumps(reports, indent=2))
        return int(args.action == "status" and any(not row["loaded"] for row in reports))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"upstream schedule: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
