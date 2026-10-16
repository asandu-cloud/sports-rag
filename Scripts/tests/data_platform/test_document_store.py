"""SQLite document store answering the engine's Chroma metadata queries (rework A2a)."""
import pytest

from Scripts.data_platform.vectors.document_store import DocumentStore, export_from_chroma


def docs():
    return [
        ("a", "Everton profile 2025", {"doc_type": "team_profile", "league": "EPL", "team": "Everton", "season": "2025/26"}),
        ("b", "Everton v Fulham", {"doc_type": "team_fixture", "league": "EPL", "team": "Everton", "season": "2025/26",
                                   "fixture": "Everton vs Fulham", "goals_for": 2}),
        ("c", "Everton v Spurs", {"doc_type": "team_fixture", "league": "EPL", "team": "Everton", "season": "2026/27",
                                  "fixture": "Tottenham vs Everton", "goals_for": 0}),
        ("d", "Chelsea v Fulham", {"doc_type": "team_fixture", "league": "EPL", "team": "Chelsea", "season": "2026/27",
                                   "fixture": "Chelsea vs Fulham", "goals_for": 1}),
    ]


@pytest.fixture
def store(tmp_path):
    s = DocumentStore(tmp_path / "docs.sqlite3")
    ids, texts, metas = zip(*docs())
    s.upsert(ids=list(ids), documents=list(texts), metadatas=list(metas))
    return s


def test_equality_filters_under_and_return_chroma_shaped_results(store):
    r = store.get(where={"$and": [{"league": "EPL"}, {"doc_type": "team_fixture"}, {"team": "Everton"}]},
                  include=["documents", "metadatas"], limit=500)
    assert r["ids"] == ["b", "c"]
    assert r["documents"] == ["Everton v Fulham", "Everton v Spurs"]
    assert r["metadatas"][0]["goals_for"] == 2


def test_in_filters_and_metadata_only_includes(store):
    r = store.get(where={"$and": [{"doc_type": "team_fixture"}, {"fixture": {"$in": ["Chelsea vs Fulham", "Nope"]}},
                                  {"league": "EPL"}, {"season": "2026/27"}]}, include=["metadatas"], limit=256)
    assert r["ids"] == ["d"] and r["documents"] is None and r["metadatas"][0]["team"] == "Chelsea"


def test_limit_keeps_insertion_order_like_chroma(store):
    assert store.get(where={"doc_type": "team_fixture"}, include=["metadatas"], limit=2)["ids"] == ["b", "c"]


def test_comparison_operators_on_any_metadata_field(store):
    assert store.get(where={"goals_for": {"$gte": 1}}, include=[])["ids"] == ["b", "d"]
    assert store.get(where={"$or": [{"team": "Chelsea"}, {"doc_type": "team_profile"}]}, include=[])["ids"] == ["a", "d"]
    assert store.get(where={"team": {"$ne": "Everton"}}, include=[])["ids"] == ["d"]


def test_upsert_updates_in_place_and_new_ids_go_last(store):
    store.upsert(ids=["e", "b"], documents=["new", "Everton v Fulham (corrected)"],
                 metadatas=[{"doc_type": "team_profile", "team": "Fulham"}, {**docs()[1][2], "goals_for": 3}])
    r = store.get(include=["documents", "metadatas"])
    assert r["ids"] == ["a", "b", "c", "d", "e"]
    assert r["metadatas"][1]["goals_for"] == 3 and store.count() == 5


def test_get_by_ids(store):
    assert store.get(ids=["d", "a"], include=[])["ids"] == ["a", "d"]


def test_unsupported_operator_is_an_error_not_a_silent_match(store):
    with pytest.raises(ValueError):
        store.get(where={"team": {"$regex": "Ev"}})


class FakeCollection:
    name = "football_top5"

    def __init__(self, rows):
        self.rows = rows

    def count(self):
        return len(self.rows)

    def get(self, *, limit, offset, include):
        page = self.rows[offset:offset + limit]
        return {"ids": [r[0] for r in page], "documents": [r[1] for r in page], "metadatas": [r[2] for r in page]}


def test_export_copies_every_document_in_collection_order(tmp_path):
    rows = [(f"id{i}", f"text {i}", {"doc_type": "team_fixture", "team": f"T{i % 3}"}) for i in range(23)]
    store = DocumentStore(tmp_path / "docs.sqlite3")
    assert export_from_chroma(FakeCollection(rows), store, batch=5) == 23
    assert store.get(include=[])["ids"] == [r[0] for r in rows]
    assert store.get(where={"team": "T1"}, include=[], limit=2)["ids"] == ["id1", "id4"]


def test_engine_lookups_use_the_store_when_switched_on(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "rag_ingest")]
    import lineup_context
    from core import team_resolution
    monkeypatch.setenv("DOCUMENT_STORE", "sqlite")
    monkeypatch.setenv("DOCUMENT_STORE_PATH", str(tmp_path / "docs.sqlite3"))
    monkeypatch.setattr(team_resolution, "_chroma_collection", None)
    for handle in (team_resolution.get_collection_handle(create_if_missing=False), lineup_context._get_chroma_collection()):
        assert isinstance(handle, DocumentStore) and handle.path == tmp_path / "docs.sqlite3"


def test_backend_writes_kb_documents_into_the_store(tmp_path):
    from Scripts.data_platform.vectors.document_store import DocumentStoreBackend
    backend = DocumentStoreBackend(DocumentStore(tmp_path / "docs.sqlite3"))
    result = backend.upsert([{"id": "k1", "text": "t", "metadata": {"doc_type": "team_profile", "team": "Hull"}}])
    assert result.upserted == 1 and backend.store.get(where={"team": "Hull"}, include=[])["ids"] == ["k1"]
    assert backend.delete(["k1"]) == 1 and backend.store.count() == 0


class Failing:
    name = "chroma"

    def upsert(self, docs):
        raise RuntimeError("OpenAI: no credits remaining")

    def delete(self, ids):
        raise RuntimeError("down")


def test_mirror_keeps_the_store_current_when_chroma_or_openai_fails(tmp_path, caplog):
    from Scripts.data_platform.vectors.document_store import DocumentStoreBackend, MirroredBackend
    store = DocumentStore(tmp_path / "docs.sqlite3")
    mirror = MirroredBackend(DocumentStoreBackend(store), Failing())
    assert mirror.upsert([{"id": "k1", "text": "t", "metadata": {"team": "Hull"}}]).upserted == 1
    assert store.count() == 1 and "no credits" in caplog.text


def test_mirror_fails_when_the_store_itself_fails(tmp_path):
    from Scripts.data_platform.vectors.document_store import MirroredBackend
    with pytest.raises(RuntimeError):
        MirroredBackend(Failing(), Failing()).upsert([{"id": "k1", "text": "t", "metadata": {}}])


def test_factory_builds_the_mirror_for_the_document_store(monkeypatch, tmp_path):
    from Scripts.data_platform.vectors.factory import build_backend
    from Scripts.data_platform.vectors.document_store import MirroredBackend
    monkeypatch.setenv("DOCUMENT_STORE_PATH", str(tmp_path / "docs.sqlite3"))
    backend = build_backend("sqlite+chroma")
    assert isinstance(backend, MirroredBackend) and backend.primary.store.path == tmp_path / "docs.sqlite3"


def test_normalized_files_load_like_the_incremental_embed(tmp_path):
    import json
    from Scripts.data_platform.vectors.document_store import load_normalized_documents
    store = DocumentStore(tmp_path / "docs.sqlite3")
    store.upsert(ids=["fx", "prof"], documents=["old fx", "old profile"],
                 metadatas=[{"doc_type": "team_fixture"}, {"doc_type": "team_profile"}])
    (tmp_path / "normalized_EPL_2026_27.json").write_text(json.dumps([
        {"id": "fx", "text": "new fx", "metadata": {"doc_type": "team_fixture"}},
        {"id": "prof", "text": "new profile", "metadata": {"doc_type": "team_profile"}},
        {"id": "new", "text": "brand new", "metadata": {"doc_type": "team_fixture"}},
        {"id": "new", "text": "duplicate in file", "metadata": {"doc_type": "team_fixture"}}]))
    (tmp_path / "normalized_EPL_2025_26.json").write_text("[]")
    counts = load_normalized_documents(store, tmp_path, season=2026)
    assert counts == {"files": 1, "written": 2, "skipped": 1}
    got = dict(zip(*[store.get(include=["documents"])[k] for k in ("ids", "documents")]))
    assert got == {"fx": "old fx", "prof": "new profile", "new": "brand new"}  # completed fixtures are immutable
