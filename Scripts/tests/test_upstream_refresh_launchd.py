import plistlib

from ops.match_read_launchd import build_paths
from ops.upstream_refresh_launchd import JOBS, render, operate


def test_upstream_jobs_are_separate_and_do_not_publish_or_retrain(tmp_path):
    paths = build_paths(root=tmp_path / "Project & Files", plist_path=tmp_path / "job.plist")
    schedule = plistlib.loads(render(JOBS["schedule"], paths).encode())
    data = plistlib.loads(render(JOBS["data"], paths).encode())
    assert schedule["StartInterval"] == 3600
    assert schedule["RunAtLoad"] is True
    assert schedule["ProgramArguments"][-5:] == ["sync-schedule", "--days-ahead", "35", "--if-stale-hours", "6"]
    assert data["StartCalendarInterval"] == {"Hour": 7, "Minute": 15}
    assert data["RunAtLoad"] is False
    assert data["ProgramArguments"][-2:] == ["refresh-season", "--include-europe"]
    assert "--season" not in data["ProgramArguments"]
    for payload in (schedule, data):
        assert "EnvironmentVariables" not in payload
        assert "match-read-cycle" not in payload["ProgramArguments"]
        assert "--retrain-ml" not in payload["ProgramArguments"]
        assert payload["ProgramArguments"][0] == str(paths.python_executable)


def test_upstream_dry_run_does_not_install_files(tmp_path):
    paths = build_paths(root=tmp_path, plist_path=tmp_path / "agent.plist")
    result = operate("install", JOBS["schedule"], paths, dry_run=True)
    assert result["dry_run"] is True
    assert not paths.plist_path.exists()
