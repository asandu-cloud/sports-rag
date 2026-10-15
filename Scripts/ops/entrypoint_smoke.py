"""Smoke-check every installed launchd job and the website in seconds (rework step A0).

Reads ``~/Library/LaunchAgents/com.bettingrag.*.plist``, so the list always
matches what actually runs. Each job is exercised without side effects: its own
interpreter and entry point with ``--help`` (module, subcommand and argument
parser must import and build), and for the uvicorn website an import of the
app object.

    python -m Scripts.ops.entrypoint_smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path
import plistlib
import subprocess
import sys
import time

from Scripts.ops.odds_archive import ROOT

LAUNCH_AGENTS = Path.home() / "Library/LaunchAgents"
PREFIX = "com.bettingrag."


def smoke_command(arguments):
    """The side-effect-free version of a job's ProgramArguments."""
    python, rest = arguments[0], list(arguments[1:])
    flags = []
    while rest and rest[0].startswith("-") and rest[0] != "-m":
        flags.append(rest.pop(0))
    if rest[:2] == ["-m", "uvicorn"]:
        target = next(a for a in rest[2:] if ":" in a and not a.startswith("-"))
        module, attribute = target.split(":", 1)
        code = f"import importlib; getattr(importlib.import_module({module!r}), {attribute!r})"
        return [python, *flags, "-c", code]
    entry = rest[:2] if rest[:1] == ["-m"] else rest[:1]
    after = rest[len(entry):]
    subcommand = after[:1] if after and not after[0].startswith("-") else []
    return [python, *flags, *entry, *subcommand, "--help"]


def jobs(launch_agents=LAUNCH_AGENTS):
    found = []
    for path in sorted(Path(launch_agents).glob(f"{PREFIX}*.plist")):
        config = plistlib.loads(path.read_bytes())
        found.append(dict(label=config["Label"], command=smoke_command(config["ProgramArguments"])))
    return found


# Subcommands import their code lazily, so --help alone would not catch a broken
# import inside a job. Import every module the jobs and the website draw on.
PRODUCTION_PACKAGES = ("Scripts.data_platform", "Scripts.rag_ingest.core", "Scripts.rag_ingest.rendering",
                       "Scripts.web_app")
IMPORT_ALL = """
import importlib, pkgutil, sys
sys.path.insert(0, "Scripts/rag_ingest")
failed = []
for package in {packages!r}:
    for info in pkgutil.walk_packages(importlib.import_module(package).__path__, package + "."):
        if ".alembic" in info.name or info.name.endswith("__main__"):
            continue
        try:
            importlib.import_module(info.name)
        except BaseException as exc:
            failed.append(f"{{info.name}}: {{type(exc).__name__}}: {{exc}}")
print("\\n".join(failed))
sys.exit(1 if failed else 0)
"""


def production_imports(python=sys.executable):
    return dict(label="imports: production modules",
                command=[python, "-c", IMPORT_ALL.format(packages=PRODUCTION_PACKAGES)])


def run(entries, *, cwd=ROOT, timeout=120):
    results = []
    for entry in entries:
        started = time.monotonic()
        try:
            done = subprocess.run(entry["command"], cwd=cwd, capture_output=True, text=True, timeout=timeout)
            ok, output = done.returncode == 0, (done.stdout + done.stderr)[-2000:]
        except (OSError, subprocess.TimeoutExpired) as exc:
            ok, output = False, f"{type(exc).__name__}: {exc}"
        results.append(dict(entry, ok=ok, seconds=round(time.monotonic() - started, 1), output=output))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--launch-agents", type=Path, default=LAUNCH_AGENTS)
    args = parser.parse_args(argv)
    entries = jobs(args.launch_agents)
    if not entries:
        print(f"no {PREFIX}* jobs in {args.launch_agents}")
        return 1
    results = run(entries + [production_imports(entries[0]["command"][0])])
    for r in results:
        print(f"{'ok  ' if r['ok'] else 'FAIL'} {r['seconds']:>5}s  {r['label']}")
        if not r["ok"]:
            print("      " + r["output"].strip().replace("\n", "\n      "))
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
