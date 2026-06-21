"""In-memory semantic routing table with semantic search and an optional disk cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import RoutingTableEntry
from .embedder import Embedder


class RoutingTable:
    def __init__(
        self,
        entries: list[RoutingTableEntry],
        embedder: Embedder,
        cache_path: str | None = None,
    ) -> None:
        self.entries = list(entries)
        self.embedder = embedder
        self.cache_path = cache_path
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.load_or_build()

    def load_or_build(self) -> None:
        if not self.entries:
            return

        table_hash = self._table_hash()
        vector_path, metadata_path = self._cache_paths()
        if vector_path and metadata_path and vector_path.exists() and metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (
                    metadata.get("table_hash") == table_hash
                    and metadata.get("embedding_model") == self.embedder.model_name
                ):
                    cached = np.load(vector_path, allow_pickle=False)
                    if cached.ndim == 2 and cached.shape[0] == len(self.entries):
                        self.embeddings = cached.astype(np.float32, copy=False)
                        return
            except (OSError, ValueError, json.JSONDecodeError):
                pass

        vectors = self.embedder.encode([entry.query for entry in self.entries])
        self.embeddings = _normalize_rows(vectors)

        if vector_path and metadata_path:
            vector_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(vector_path, self.embeddings, allow_pickle=False)
            metadata_path.write_text(
                json.dumps(
                    {
                        "table_hash": table_hash,
                        "embedding_model": self.embedder.model_name,
                        "entries": [asdict(entry) for entry in self.entries],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    def best_match(self, query: str) -> tuple[RoutingTableEntry | None, float]:
        if not self.entries:
            return None, 0.0

        query_vector = _normalize_rows(self.embedder.encode([query]))[0]
        scores = self.embeddings @ query_vector
        best_index = int(np.argmax(scores))
        return self.entries[best_index], float(scores[best_index])

    def _table_hash(self) -> str:
        payload = json.dumps(
            [asdict(entry) for entry in self.entries],
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_paths(self) -> tuple[Path | None, Path | None]:
        if not self.cache_path:
            return None, None
        base = Path(self.cache_path)
        if base.suffix in {".npy", ".json"}:
            base = base.with_suffix("")
        return base.with_suffix(".npy"), base.with_suffix(".json")


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)

