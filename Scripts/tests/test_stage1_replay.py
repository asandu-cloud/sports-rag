"""Stage 1 review: the corners selector replayed with stored floors and with the fixed dispersion."""
from Scripts.ops import stage1_replay


def test_replay_runs_the_selector_with_both_variances(monkeypatch):
    seen = []
    line_selection, alpha = stage1_replay._engine()

    def fake(options, mean, variance, **kwargs):
        seen.append(variance)
        return {"bet_recommendation": {"side": "Under", "point": 10.5, "odds": 1.5, "bookmaker": "A", "_model_prob": 0.7}}

    monkeypatch.setattr(line_selection, "select_best_total_recommendation", fake)
    options = [{"market_key": "totals_corners_over_under", "side": s, "point": 10.5, "odds": o, "bookmaker": "A",
                "period": "regulation_time"} for s, o in (("over", 2.6), ("under", 1.5))]
    result = {"projection": {"value": 10.0, "variance": 15.0},
              "context": {"decision_audit": {"input_options": options}}}
    picks = stage1_replay.replay(result)
    assert seen == [15.0, 10.0 + alpha * 100.0]
    assert picks["fixed"]["side"] == "under" and picks["floors"]["point"] == 10.5
    assert set(picks["books"]) == {"A"}


def test_a_result_without_prices_is_skipped():
    assert stage1_replay.replay({"projection": {"value": 10.0, "variance": 15.0}, "context": {}}) is None
