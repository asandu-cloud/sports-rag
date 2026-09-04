"""Regression coverage for incremental Chroma embedding decisions.

The executable embedding script is intentionally loaded with lightweight
stand-ins for its runtime dependencies.  These tests only exercise the
selection logic and never create an OpenAI client or call an embedding API.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "Scripts" / "rag_ingest" / "01_embed_and_upsert.py"


def _load_embedding_module():
    """Load the executable script without its network-backed dependencies."""
    fake_openai = ModuleType("openai")

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.embeddings = object()

    fake_openai.OpenAI = FakeOpenAI

    fake_dotenv = ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda: None

    fake_chroma_backend = ModuleType("chroma_backend")
    fake_chroma_backend.backend_description = lambda *_args, **_kwargs: "fake"
    fake_chroma_backend.env_first = lambda _name, default=None: default
    fake_chroma_backend.get_chroma_client = lambda *_args, **_kwargs: None

    module_name = "embedding_incremental_profiles_under_test"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "openai": fake_openai,
            "dotenv": fake_dotenv,
            "chroma_backend": fake_chroma_backend,
        },
    ):
        spec.loader.exec_module(module)
    return module


class IncrementalProfileEmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedding = _load_embedding_module()

    def test_existing_team_and_player_profiles_are_reembedded(self):
        """Season-to-date profile IDs are stable but their content is mutable."""
        collection = object()
        docs = [
            {
                "id": "team_profile:EPL:2026:42",
                "text": "updated team profile",
                "metadata": {"doc_type": "team_profile"},
            },
            {
                "id": "player_profile:EPL:2026:7",
                "text": "updated player profile",
                "metadata": {"doc_type": "player_profile"},
            },
        ]

        with mock.patch.object(self.embedding, "_upsert_batch") as upsert_batch:
            embedded, skipped = self.embedding.upsert_docs(
                collection,
                docs,
                skip_ids={doc["id"] for doc in docs},
            )

        self.assertEqual((embedded, skipped), (2, 0))
        upsert_batch.assert_called_once_with(collection, docs)

    def test_existing_completed_fixture_documents_are_skipped(self):
        """Completed fixtures remain immutable in the normal incremental run."""
        collection = object()
        docs = [
            {
                "id": "team_fixture:EPL:2026:100:42",
                "text": "completed team fixture",
                "metadata": {"doc_type": "team_fixture"},
            },
            {
                "id": "player_fixture:EPL:2026:100:7",
                "text": "completed player fixture",
                "metadata": {"doc_type": "player_fixture"},
            },
        ]

        with mock.patch.object(self.embedding, "_upsert_batch") as upsert_batch:
            embedded, skipped = self.embedding.upsert_docs(
                collection,
                docs,
                skip_ids={doc["id"] for doc in docs},
            )

        self.assertEqual((embedded, skipped), (0, 2))
        upsert_batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
