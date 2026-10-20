import pytest
import requests
from requests.adapters import BaseAdapter
from requests.models import Response

from Scripts.ops import golden_snapshots as golden


class FakeCollection:
    """Stands in for a Chroma collection; counts real reads."""
    name = "football_top5"
    reads = 0

    def get(self, **kwargs):
        FakeCollection.reads += 1
        return {"ids": ["a"], "metadatas": [{"team": kwargs["where"]["team"], "goals_for_pm": 1.5}]}

    def query(self, **kwargs):
        FakeCollection.reads += 1
        return {"ids": [["b"]], "distances": [[0.25]]}


class CannedAdapter(BaseAdapter):
    calls = 0

    def send(self, request, **kwargs):
        CannedAdapter.calls += 1
        response = Response()
        response.status_code = 200
        response._content = b'{"response": [{"referee": "M. Oliver"}]}'
        response.headers["Content-Type"] = "application/json"
        response.url = request.url
        response.request = request
        return response

    def close(self):
        pass


def session():
    s = requests.Session()
    s.mount("https://", CannedAdapter())
    return s


def evaluator(scale=1.0):
    """A fake engine: reads a team profile and a referee, returns two 'market results'."""
    def evaluate(event, league, **kwargs):
        meta = FakeCollection().get(where={"team": event["home_team"]})["metadatas"][0]
        referee = session().get("https://v3.example/fixtures", params={"id": event["id"]},
                                headers={"x-apisports-key": "SECRET"}).json()["response"][0]["referee"]
        p = round(meta["goals_for_pm"] / 3 * scale, 6)
        return [{"market": {"key": "totals"}, "decision": {"status": "recommended", "model_probability": p},
                 "provenance": {"generated_at": kwargs["generated_at"], "system_version": "code:1"},
                 "context": {"referee": referee}}]
    return evaluate


TARGETS = dict(chroma_targets=[(FakeCollection, "get"), (FakeCollection, "query")])
EVENTS = [dict(league="EPL", fixture_date="2026-10-17", event={"id": 1, "home_team": "Everton"}),
          dict(league="EPL", fixture_date="2026-10-17", event={"id": 2, "home_team": "Fulham"})]


@pytest.fixture(autouse=True)
def counters():
    FakeCollection.reads = CannedAdapter.calls = 0


def test_replay_serves_recorded_reads_without_touching_the_sources():
    pool = {}
    with golden.Capture("record", pool, **TARGETS):
        live = FakeCollection().get(where={"team": "Everton"})
        body = session().get("https://v3.example/fixtures", params={"id": 1}).json()
    assert FakeCollection.reads == 1 and CannedAdapter.calls == 1

    with golden.Capture("replay", pool, **TARGETS):
        assert FakeCollection().get(where={"team": "Everton"}) == live
        replayed = session().get("https://v3.example/fixtures", params={"id": 1})
        assert replayed.status_code == 200 and replayed.json() == body
    assert FakeCollection.reads == 1 and CannedAdapter.calls == 1


def test_replay_fails_loudly_on_a_read_that_was_never_recorded():
    with golden.Capture("replay", {}, **TARGETS):
        with pytest.raises(golden.GoldenMiss, match="chroma"):
            FakeCollection().get(where={"team": "Arsenal"})
        with pytest.raises(golden.GoldenMiss, match="http"):
            session().get("https://v3.example/fixtures", params={"id": 9})


def test_recorded_http_never_keeps_request_headers():
    pool = {}
    with golden.Capture("record", pool, **TARGETS):
        session().get("https://v3.example/fixtures", params={"id": 1}, headers={"x-apisports-key": "SECRET"})
    assert "SECRET" not in repr(pool)


def test_compare_ignores_volatile_fields_and_reports_real_changes():
    old = [{"decision": {"status": "recommended", "model_probability": 0.6},
            "provenance": {"generated_at": "a", "system_version": "code:1", "model_version": "code:1"},
            "context": {"prediction_system": {"version": "code:1"}}}]
    same = [{"decision": {"status": "recommended", "model_probability": 0.6 + 1e-13},
             "provenance": {"generated_at": "b", "system_version": "code:2", "model_version": "code:2"},
             "context": {"prediction_system": {"version": "code:2"}}}]
    changed = [{"decision": {"status": "no_bet", "model_probability": 0.55},
                "provenance": {"generated_at": "a", "system_version": "code:1", "model_version": "code:1"},
                "context": {"prediction_system": {"version": "code:1"}}}]
    assert golden.compare(old, same) == []
    diffs = golden.compare(old, changed)
    assert ("[0].decision.status", "recommended", "no_bet") in diffs
    assert ("[0].decision.model_probability", 0.6, 0.55) in diffs


def test_bundle_round_trip_detects_an_engine_change(tmp_path):
    path = tmp_path / "golden.json.gz"
    bundle = golden.record(EVENTS, evaluator(), path, generated_at="2026-10-15T00:00:00Z", **TARGETS)
    assert len(bundle["fixtures"]) == 2 and path.exists()
    reads = FakeCollection.reads, CannedAdapter.calls

    assert golden.check(path, evaluator(), **TARGETS) == []
    assert (FakeCollection.reads, CannedAdapter.calls) == reads  # replay never reached the sources

    report = golden.check(path, evaluator(scale=1.1), **TARGETS)
    assert [r["fixture"] for r in report] == ["EPL 2026-10-17 1", "EPL 2026-10-17 2"]
    assert report[0]["changes"][0][0] == "[0].decision.model_probability"


def test_an_engine_error_on_replay_is_reported_not_raised(tmp_path):
    path = tmp_path / "golden.json.gz"
    golden.record(EVENTS[:1], evaluator(), path, generated_at="2026-10-15T00:00:00Z", **TARGETS)

    def broken(event, league, **kwargs):
        raise KeyError("goals_for_pm")

    (report,) = golden.check(path, broken, **TARGETS)
    assert "KeyError" in report["error"]


def test_a_missed_read_is_reported_even_when_the_engine_swallows_it(tmp_path):
    path = tmp_path / "golden.json.gz"
    golden.record(EVENTS[:1], evaluator(), path, generated_at="2026-10-15T00:00:00Z", **TARGETS)

    def tolerant(event, league, **kwargs):
        try:
            FakeCollection().get(where={"team": "Never recorded"})
        except Exception:
            pass  # the engine's own fallback hides the miss
        return evaluator()(event, league, **kwargs)

    (report,) = golden.check(path, tolerant, **TARGETS)
    assert report["missing_reads"] and "Never recorded" in report["missing_reads"][0]


def test_summary_lists_pick_changes_before_audit_detail():
    report = [dict(fixture="EPL 2026-10-17 1", changes=[
        ("[0].context.decision_audit.candidates[0].option._ev", -0.1, -0.2),
        ("[0].projection.value", 2.6, 2.7),
        ("[0].decision.status", "recommended", "no_bet")])]
    lines = golden.summarize(report).splitlines()
    assert [line.strip().split(":")[0] for line in lines[1:]] == [
        "[0].decision.status", "[0].projection.value", "[0].context.decision_audit.candidates[0].option._ev"]


def test_document_store_reads_are_recorded_like_chroma_reads():
    from Scripts.data_platform.vectors.document_store import DocumentStore
    assert (DocumentStore, "get") in golden._chroma_targets()


def test_live_documents_mode_replays_http_but_reads_documents_from_the_source(tmp_path):
    path = tmp_path / "golden.json.gz"
    golden.record(EVENTS, evaluator(), path, generated_at="2026-10-15T00:00:00Z", **TARGETS)
    reads = FakeCollection.reads
    assert golden.check(path, evaluator(), chroma_targets=[]) == []
    assert FakeCollection.reads == reads + 2  # documents read live; HTTP still served from the bundle


def test_the_audits_code_fingerprint_is_volatile():
    old = [{"context": {"decision_audit": {"prediction_version": "code:1", "selector_ran": True}}}]
    new = [{"context": {"decision_audit": {"prediction_version": "code:2", "selector_ran": True}}}]
    assert golden.compare(old, new) == []


def test_preview_against_a_baseline_replays_both_settings(tmp_path, monkeypatch):
    import os

    def switched(event, league, **kwargs):  # a stage switch that withdraws the pick
        result = evaluator()(event, league, **kwargs)[0]
        result["market"] = {"key": "totals", "group": "goals"}
        if os.environ.get("STAGE_X") == "on":
            result["decision"] = {"status": "no_bet", "model_probability": 0.1}
        return [result]

    path = tmp_path / "golden.json.gz"
    monkeypatch.setenv("STAGE_X", "off")
    golden.record(EVENTS[:1], switched, path, generated_at="2026-10-15T00:00:00Z", **TARGETS)
    monkeypatch.setenv("STAGE_X", "on")
    assert [r["change"] for r in golden.preview(path, switched)] == ["removed"]
    assert [r["change"] for r in golden.preview(path, switched, baseline={"STAGE_X": "on"})] == [None]  # same settings
    assert [r["change"] for r in golden.preview(path, switched, baseline={"STAGE_X": "off"})] == ["removed"]
    assert os.environ["STAGE_X"] == "on"  # restored after each baseline replay
