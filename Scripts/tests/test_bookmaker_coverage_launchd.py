import plistlib

from ops.match_read_launchd import build_paths
from ops.bookmaker_coverage_launchd import LABEL, SCHEDULE, operate, render


def test_the_coverage_watch_runs_four_times_a_day_and_only_reads(tmp_path):
    paths = build_paths(root=tmp_path / "Project & Files", plist_path=tmp_path / "job.plist")
    job = plistlib.loads(render(paths).encode())
    assert job["Label"] == LABEL == "com.bettingrag.bookmaker-coverage"
    assert job["ProgramArguments"] == [str(paths.python_executable), "-B", "-m", "Scripts.ops.bookmaker_coverage"]
    assert job["StartCalendarInterval"] == SCHEDULE and [s["Hour"] for s in SCHEDULE] == [9, 13, 17, 21]
    assert job["StandardOutPath"].endswith("bookmaker-coverage.stdout.log")
    assert "EnvironmentVariables" not in job


def test_a_dry_run_installs_nothing(tmp_path):
    paths = build_paths(root=tmp_path, plist_path=tmp_path / "agent.plist")
    assert operate("install", paths, dry_run=True)["dry_run"] is True
    assert not paths.plist_path.exists()
