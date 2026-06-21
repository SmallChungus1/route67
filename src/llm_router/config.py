"""Configuration models for the router."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ModelTarget = Literal["weak_model", "strong_model"]
MODEL_TARGETS = frozenset({"weak_model", "strong_model"})


@dataclass(frozen=True, slots=True)
class RoutingTableEntry:
    query: str
    target: ModelTarget
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.target not in MODEL_TARGETS:
            raise ValueError(
                "routing target must be 'weak_model' or 'strong_model'"
            )


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    usage_notes: str | None = None


@dataclass(slots=True)
class RouterConfig:
    routing_table: list[RoutingTableEntry] = field(default_factory=list)
    similarity_threshold: float = 0.75
    weak_model: ModelSpec | None = None
    strong_model: ModelSpec | None = None
    embedding_cache_path: str | None = None
    log_path: str | None = None
    escalation_max_tokens: int = 10
    embedding_model: str = "minishlab/potion-base-8M"

    def __post_init__(self) -> None:
        if not -1.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1.0 and 1.0")
        if self.weak_model is None:
            raise ValueError("weak_model is required")
        if self.strong_model is None:
            raise ValueError("strong_model is required")
        if self.escalation_max_tokens < 1:
            raise ValueError("escalation_max_tokens must be at least 1")

    def resolve_target(self, target: ModelTarget) -> ModelSpec:
        if target == "weak_model":
            if self.weak_model is None:
                raise RuntimeError("weak_model is not configured")
            return self.weak_model
        if target == "strong_model":
            if self.strong_model is None:
                raise RuntimeError("strong_model is not configured")
            return self.strong_model
        raise ValueError("routing target must be 'weak_model' or 'strong_model'")
