import plistlib

import pytest

from Scripts.ops import entrypoint_smoke as smoke

PY = "/repo/.venv/bin/python"


@pytest.mark.parametrize("arguments, expected", [
    ([PY, "-m", "Scripts.data_platform", "refresh-season", "--include-europe"],
     [PY, "-m", "Scripts.data_platform", "refresh-season", "--help"]),
    ([PY, "-B", "-m", "Scripts.ops.odds_archive", "--execute", "--near-kickoff-hours", "3"],
     [PY, "-B", "-m", "Scripts.ops.odds_archive", "--help"]),
    ([PY, "-B", "/repo/Research/phase6/controller.py", "capture"],
     [PY, "-B", "/repo/Research/phase6/controller.py", "capture", "--help"]),
])
def test_jobs_are_smoked_with_help_on_the_same_entry_point(arguments, expected):
    assert smoke.smoke_command(arguments) == expected


def test_the_website_is_smoked_by_importing_its_app():
    command = smoke.smoke_command([PY, "-B", "-m", "uvicorn", "Scripts.web_app.api:app", "--port", "8000"])
    assert command[:3] == [PY, "-B", "-c"]
    assert "Scripts.web_app.api" in command[3] and "'app'" in command[3]


def test_jobs_come_from_the_installed_launch_agents(tmp_path):
    for label, args in [("com.bettingrag.website", [PY, "-m", "uvicorn", "Scripts.web_app.api:app"]),
                        ("com.other.thing", [PY, "-m", "elsewhere"])]:
        (tmp_path / f"{label}.plist").write_bytes(plistlib.dumps(dict(Label=label, ProgramArguments=args)))
    (found,) = smoke.jobs(tmp_path)
    assert found["label"] == "com.bettingrag.website"


def test_a_failing_entry_point_is_reported(tmp_path):
    good = dict(label="good", command=[smoke.sys.executable, "-c", "print('ok')"])
    bad = dict(label="bad", command=[smoke.sys.executable, "-c", "import does_not_exist_anywhere"])
    results = {r["label"]: r for r in smoke.run([good, bad], cwd=tmp_path)}
    assert results["good"]["ok"] and not results["bad"]["ok"]
    assert "does_not_exist_anywhere" in results["bad"]["output"]


def test_smoke_also_imports_every_production_module():
    entry = smoke.production_imports()
    assert entry["label"] == "imports: production modules"
    code = entry["command"][-1]
    for package in ("Scripts.data_platform", "Scripts.rag_ingest.core", "Scripts.rag_ingest.rendering", "Scripts.web_app"):
        assert package in code
