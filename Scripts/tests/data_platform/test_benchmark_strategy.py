"""The frozen system forecast must apply its saved statistical/ML strategy."""
from copy import deepcopy

import numpy as np
import pytest

from Scripts.data_platform.features.benchmarks import artifacts
from Scripts.tests.data_platform.test_benchmark_runner import candidate


@pytest.mark.parametrize("weight,method", [(0.0, "statistical"), (.25, "selected_blend"), (1.0, "selected_ml")])
def test_frozen_strategy_applies_saved_weight_and_preserves_estimator_api(candidate, tmp_path, weight, method):
    _, model, original_rows, metadata = candidate
    rows = [{**row, "fixture": {"fixture_id": index + 1}} for index, row in enumerate(original_rows)]
    path = tmp_path / method
    artifacts.save_candidate(path, model, metadata={**metadata, "qualification": "UNCONFIRMED_BATCH_C_DEVELOPMENT",
                             "baseline_contract_id": "b" * 64, "strategy": method, "weight_ml": weight})
    baselines = [{"fixture_id": row["fixture"]["fixture_id"], "snapshot_id": row["snapshot_id"],
                  "feature_contract_id": "a" * 64, "market": "goals", "statistical": 9.0} for row in rows]
    raw = artifacts.predict_candidate(path, rows, feature_contract_id="a" * 64)
    system = artifacts.predict_strategy(path, rows, feature_contract_id="a" * 64,
                                       statistical_rows=baselines, baseline_contract_id="b" * 64)
    assert np.array_equal(system, (1 - weight) * 9.0 + weight * raw)
    with pytest.raises(ValueError, match="contract"):
        artifacts.predict_strategy(path, rows, feature_contract_id="a" * 64,
                                   statistical_rows=baselines, baseline_contract_id="c" * 64)
    mismatch = deepcopy(baselines)
    mismatch[0]["market"] = "corners"
    with pytest.raises(ValueError, match="snapshot/market"):
        artifacts.predict_strategy(path, rows, feature_contract_id="a" * 64,
                                   statistical_rows=mismatch, baseline_contract_id="b" * 64)
    with pytest.raises(ValueError, match="identical"):
        artifacts.predict_strategy(path, rows, feature_contract_id="a" * 64,
                                   statistical_rows=baselines[:-1], baseline_contract_id="b" * 64)

