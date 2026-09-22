from datetime import datetime, timezone
import json
import hashlib
import socket
from pathlib import Path
import subprocess
import sys

import joblib
import pytest

from Scripts.data_platform.features.model_dataset import capture_fixture_snapshot
from Scripts.ops.prediction_candidate import fit_candidate, predict_candidate_bundle
from Scripts.ops.prediction_features import export_dataset
from Scripts.tests.data_platform.test_model_dataset import _seed, _path


def test_candidate_fit_only_uses_training_rows_and_never_writes_active_models(settings, engine, session_factory, tmp_path, monkeypatch):
    _seed(session_factory, count=9)
    root = tmp_path / "project"
    active = root / "Index/ml_models/model_r2.json"
    active.parent.mkdir(parents=True)
    active.write_text('{"goals": 0.0181}')
    dataset = export_dataset(root=root, database=_path(settings), name="data", seasons=[2026], availability="assumed_final")
    before_database = hashlib.sha256(_path(settings).read_bytes()).hexdigest()
    def no_network(*args, **kwargs):
        raise AssertionError("Candidate fitting must not call a provider or publish")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    candidate = fit_candidate(root=root, dataset=dataset, name="candidate", market="goals",
                              train_through=datetime(2026, 8, 19, tzinfo=timezone.utc))
    metadata = json.loads((candidate / "manifest.json").read_text())
    assert metadata["training_fixture_ids"] == [101, 102, 103, 104, 105, 106]
    assert metadata["exclusions"]["outside_training_cutoff"] == 2
    assert metadata["promotion_allowed"] is False
    model = joblib.load(candidate / "candidate/model.joblib")
    index = metadata["feature_contract"]["names"].index("home_current_matches")
    assert model.named_steps["imputer"].statistics_[index] == 3.5
    snapshot = capture_fixture_snapshot(_path(settings), 108, as_of=datetime(2026, 8, 21, tzinfo=timezone.utc))
    prediction = predict_candidate_bundle(candidate, snapshot)
    assert prediction["value"] == pytest.approx(0)
    assert prediction["published"] is False
    assert active.read_text() == '{"goals": 0.0181}'
    assert list(active.parent.iterdir()) == [active]
    assert hashlib.sha256(_path(settings).read_bytes()).hexdigest() == before_database
    with pytest.raises(ValueError, match="simple new name"):
        fit_candidate(root=root, dataset=dataset, name="../ml_models", market="goals",
                      train_through=datetime(2026, 8, 19, tzinfo=timezone.utc))
    with (candidate / "candidate/model.joblib").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        predict_candidate_bundle(candidate, snapshot)


def test_insufficient_candidate_data_does_not_create_artifacts(settings, engine, session_factory, tmp_path):
    _seed(session_factory)
    root = tmp_path / "project"
    dataset = export_dataset(root=root, database=_path(settings), name="tiny", seasons=[2026], availability="assumed_final")
    with pytest.raises(ValueError, match="Fewer than five"):
        fit_candidate(root=root, dataset=dataset, name="candidate", market="goals",
                      train_through=datetime(2026, 9, 1, tzinfo=timezone.utc))
    assert not (root / "Index/prediction_experiments/candidate").exists()


@pytest.mark.parametrize("args", [["--train"], ["--train-cumulative"], ["--cv"], ["--predict", "Home", "Away", "EPL"]])
def test_legacy_training_and_raw_row_diagnostics_are_retired(args):
    script = Path(__file__).resolve().parents[2] / "rag_ingest/ml_edge.py"
    result = subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)
    assert result.returncode == 2
    assert "retired" in result.stderr
