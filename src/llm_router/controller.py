"""OpenAI-compatible public controller."""

from __future__ import annotations

from typing import Any

from .config import RouterConfig
from .embedder import Embedder
from .escalation import run_with_escalation
from .logging_utils import RoutingDecision, log_decision
from .routing_table import RoutingTable


class Controller:
    def __init__(
        self,
        config: RouterConfig,
        openai_client: Any | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.config = config
        self.client = openai_client or _default_openai_client()
        self.table = RoutingTable(
            config.routing_table,
            embedder or Embedder(config.embedding_model),
            config.embedding_cache_path,
        )
        self.chat = _ChatProxy(self)

    def chat_completions_create(self, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            raise NotImplementedError("Public streaming is not supported in route67 v1")
        messages = kwargs.get("messages")
        if not isinstance(messages, list):
            raise TypeError("messages must be provided as a list")

        query = extract_user_query(messages)
        entry, score = self.table.best_match(query)
        forwarded = {key: value for key, value in kwargs.items() if key != "model"}

        if entry is not None and score >= self.config.similarity_threshold:
            selected_model = self.config.resolve_target(entry.target)
            response = self.client.chat.completions.create(
                model=selected_model.name,
                **forwarded,
            )
            decision = RoutingDecision("table_match", selected_model.name, score)
        else:
            result = run_with_escalation(
                self.client,
                self.config,
                messages,
                request_kwargs=forwarded,
            )
            response = result.response
            decision = RoutingDecision(
                "escalated" if result.escalated else "weak_model_direct",
                result.used_model,
                score,
            )

        log_decision(self.config.log_path, query, decision)
        return response


def extract_user_query(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return str(content)
    raise ValueError("messages must contain at least one user message")


def _default_openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai is required when openai_client is not supplied") from exc
    return OpenAI()


class _ChatProxy:
    def __init__(self, controller: Controller) -> None:
        self.completions = _CompletionsProxy(controller)


class _CompletionsProxy:
    def __init__(self, controller: Controller) -> None:
        self._controller = controller

    def create(self, **kwargs: Any) -> Any:
        return self._controller.chat_completions_create(**kwargs)
