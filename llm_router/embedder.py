"""Small, lazy-loading Model2Vec embedder."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class Embedder:
    def __init__(
        self,
        model_name: str = "minishlab/potion-base-8M",
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from model2vec import StaticModel
            except ImportError as exc:
                raise ImportError(
                    "model2vec is required to compute embeddings; install route67"
                ) from exc
            self._model = StaticModel.from_pretrained(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            raise TypeError("encode expects a sequence of strings; use encode_one for one string")
        vectors = np.asarray(self.model.encode(list(texts)), dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("embedder returned an array with an unexpected shape")
        return vectors

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

