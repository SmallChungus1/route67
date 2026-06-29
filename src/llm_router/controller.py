"""OpenAI-compatible public controller."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import RouterConfig
from .routing import RoutingDecision, build_chat_request, execute_chat_request, extract_user_query
from .semantic import Embedder, RoutingTable


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
        request = build_chat_request(kwargs)
        response, decision = execute_chat_request(
            self.client,
            self.config,
            self.table,
            request,
        )
        log_decision(self.config.log_path, request.query, decision)
        return response


def _default_openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai is required when openai_client is not supplied") from exc
    return OpenAI()


def log_decision(log_path: str | None, query: str, decision: RoutingDecision) -> None:
    if not log_path:
        return

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_preview": " ".join(query.split())[:100],
        "method": decision.method,
        "model_used": decision.model,
        "similarity_score": round(decision.score, 6),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class _ChatProxy:
    def __init__(self, controller: Controller) -> None:
        self.completions = _CompletionsProxy(controller)


class _CompletionsProxy:
    def __init__(self, controller: Controller) -> None:
        self._controller = controller

    def create(self, **kwargs: Any) -> Any:
        return self._controller.chat_completions_create(**kwargs)
