from __future__ import annotations

import unittest
from unittest import mock

from Scripts.rag_ingest.core import team_resolution


class _FakeCollection:
    def __init__(self, metadatas):
        self._metadatas = metadatas

    def get(self, **kwargs):
        return {"documents": ["doc"] * len(self._metadatas), "metadatas": self._metadatas}


class RecentStatsCutoffTests(unittest.TestCase):
    def test_recent_fixture_rows_exclude_future_matches(self):
        metadatas = [
            {"fixture": "Future", "fixture_date": "2026-04-12", "season": "2025", "team": "Arsenal"},
            {"fixture": "Recent A", "fixture_date": "2026-04-08", "season": "2025", "team": "Arsenal"},
            {"fixture": "Recent B", "fixture_date": "2026-04-05", "season": "2025", "team": "Arsenal"},
        ]
        fake_collection = _FakeCollection(metadatas)

        with mock.patch.object(team_resolution, "get_collection_handle", return_value=fake_collection), \
             mock.patch.object(team_resolution, "_kb_team_variants", return_value=["Arsenal"]), \
             mock.patch.object(team_resolution, "get_team_profile_doc", return_value={"meta": {"season": "2025"}}):
            rows = team_resolution.get_recent_team_fixture_rows(
                "Arsenal",
                "EPL",
                limit=5,
                target_date="2026-04-10",
            )

        fixtures = [row["meta"]["fixture"] for row in rows]
        self.assertEqual(fixtures, ["Recent A", "Recent B"])
        self.assertNotIn("Future", fixtures)


if __name__ == "__main__":
    unittest.main()
