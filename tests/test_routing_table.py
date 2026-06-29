from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from llm_router.config import RoutingTableEntry
from llm_router.semantic import RoutingTable


class FakeEmbedder:
    model_name = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        mapping = {
            "math": [1.0, 0.0],
            "writing": [0.0, 1.0],
            "algebra": [0.9, 0.1],
        }
        return np.asarray([mapping[text] for text in texts], dtype=np.float32)


class RoutingTableTests(unittest.TestCase):
    def test_best_match_uses_cosine_similarity(self) -> None:
        entries = [
            RoutingTableEntry("math", "strong_model"),
            RoutingTableEntry("writing", "weak_model"),
        ]
        table = RoutingTable(entries, FakeEmbedder())

        entry, score = table.best_match("algebra")

        self.assertEqual(entry, entries[0])
        self.assertGreater(score, 0.99)

    def test_empty_table_does_not_load_embedder(self) -> None:
        embedder = FakeEmbedder()
        table = RoutingTable([], embedder)

        self.assertEqual(table.best_match("anything"), (None, 0.0))
        self.assertEqual(embedder.calls, 0)

    def test_valid_cache_skips_reembedding_table(self) -> None:
        entries = [RoutingTableEntry("math", "strong_model")]
        with tempfile.TemporaryDirectory() as directory:
            cache = str(Path(directory) / "routes")
            first = FakeEmbedder()
            RoutingTable(entries, first, cache)
            second = FakeEmbedder()
            RoutingTable(entries, second, cache)

        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 0)

    def test_changed_table_invalidates_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = str(Path(directory) / "routes")
            RoutingTable(
                [RoutingTableEntry("math", "strong_model")],
                FakeEmbedder(),
                cache,
            )
            embedder = FakeEmbedder()
            RoutingTable(
                [RoutingTableEntry("writing", "weak_model")],
                embedder,
                cache,
            )

        self.assertEqual(embedder.calls, 1)

    def test_entry_rejects_provider_model_name_as_target(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "must be 'weak_model' or 'strong_model'"
        ):
            RoutingTableEntry("math", "provider/model")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
