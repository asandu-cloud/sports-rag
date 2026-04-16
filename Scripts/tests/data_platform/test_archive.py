"""Raw payload archival: storage backends + metadata row."""

from __future__ import annotations

import gzip

from data_platform.storage import LocalDiskStorage, PayloadArchiver, build_storage
from data_platform.models import RawPayloadArchive


def test_local_disk_round_trip(tmp_path):
    storage = LocalDiskStorage(tmp_path)
    uri = storage.put("a/b/c.txt", b"hello", content_type="text/plain")
    assert uri.startswith("file://")
    assert storage.get("a/b/c.txt") == b"hello"
    assert storage.exists("a/b/c.txt")
    assert not storage.exists("missing")


def test_build_storage_defaults_to_local(settings):
    storage = build_storage(settings)
    assert storage.backend_name == "local"


def test_archiver_writes_row_and_dedupes(session_factory, settings):
    payload = {"foo": [1, 2, 3], "nested": {"k": "v"}}
    params = {"league": 39, "season": 2025}
    with session_factory() as s:
        archiver = PayloadArchiver(s, settings=settings)
        res1 = archiver.archive_json(
            provider="api_football", endpoint="/fixtures",
            params=params, payload=payload,
        )
        assert res1.was_new is True
        res2 = archiver.archive_json(
            provider="api_football", endpoint="/fixtures",
            params=params, payload=payload,
        )
        assert res2.was_new is False, "identical payload should dedupe"
        assert res2.archive_id == res1.archive_id

        # Changed payload → new row
        res3 = archiver.archive_json(
            provider="api_football", endpoint="/fixtures",
            params=params, payload={"foo": [9, 9]},
        )
        assert res3.was_new is True
        assert res3.archive_id != res1.archive_id

    with session_factory() as s:
        rows = s.query(RawPayloadArchive).all()
        assert len(rows) == 2
        for row in rows:
            assert row.storage_uri.startswith("file://")
            # Payload is gzipped by default
            path = row.storage_uri.replace("file://", "")
            with open(path, "rb") as fh:
                body = fh.read()
            decoded = gzip.decompress(body)
            assert decoded  # non-empty JSON


def test_archiver_replay(session_factory, settings):
    payload = {"x": [1, 2], "y": {"k": 9}}
    with session_factory() as s:
        archiver = PayloadArchiver(s, settings=settings)
        res = archiver.archive_json(
            provider="api_football", endpoint="/fixtures/statistics",
            params={"fixture": 1234}, payload=payload,
        )
        loaded = archiver.load_json(res.archive_id)
    assert loaded == payload
