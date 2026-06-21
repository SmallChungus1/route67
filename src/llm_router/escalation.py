"""Sequential weak-to-strong escalation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from .config import RouterConfig
from .prompts import build_escalation_prompt

SENTINEL = "ESCALATE"


@dataclass(frozen=True, slots=True)
class EscalationResult:
    used_model: str
    response: Any
    escalated: bool


def run_with_escalation(
    client: Any,
    config: RouterConfig,
    messages: list[dict[str, Any]],
    request_kwargs: dict[str, Any] | None = None,
) -> EscalationResult:
    request_kwargs = dict(request_kwargs or {})
    request_kwargs.pop("model", None)
    request_kwargs.pop("messages", None)
    request_kwargs.pop("stream", None)

    prompt = build_escalation_prompt(
        config.weak_model.usage_notes,
        config.strong_model,
        config.routing_table,
    )
    weak_messages = [{"role": "system", "content": prompt}, *messages]
    weak_stream = client.chat.completions.create(
        model=config.weak_model.name,
        messages=weak_messages,
        stream=True,
        **request_kwargs,
    )

    chunks: list[Any] = []
    preview = ""
    decision_made = False
    try:
        for chunk in weak_stream:
            chunks.append(chunk)
            preview += _chunk_text(chunk)
            if not decision_made and _decision_boundary_reached(
                preview, config.escalation_max_tokens
            ):
                decision_made = True
                if _is_escalation(preview):
                    _close_stream(weak_stream)
                    strong_response = client.chat.completions.create(
                        model=config.strong_model.name,
                        messages=messages,
                        stream=False,
                        **request_kwargs,
                    )
                    return EscalationResult(
                        used_model=config.strong_model.name,
                        response=strong_response,
                        escalated=True,
                    )
    finally:
        _close_stream(weak_stream)

    if _is_escalation(preview):
        strong_response = client.chat.completions.create(
            model=config.strong_model.name,
            messages=messages,
            stream=False,
            **request_kwargs,
        )
        return EscalationResult(config.strong_model.name, strong_response, True)

    response = _assemble_chat_completion(chunks, config.weak_model.name)
    return EscalationResult(config.weak_model.name, response, False)


def _decision_boundary_reached(text: str, max_tokens: int) -> bool:
    stripped = text.lstrip()
    if "\n" in stripped or "\r" in stripped:
        return True
    if stripped.lower().startswith(SENTINEL.lower()) and len(stripped) >= len(SENTINEL):
        return True
    if stripped and not SENTINEL.lower().startswith(stripped.lower()):
        return True
    return len(stripped.split()) >= max_tokens


def _is_escalation(text: str) -> bool:
    return text.lstrip().lower().startswith(SENTINEL.lower())


def _chunk_text(chunk: Any) -> str:
    choices = _get(chunk, "choices", []) or []
    if not choices:
        return ""
    delta = _get(choices[0], "delta")
    return _get(delta, "content", "") or ""


def _close_stream(stream: Any) -> None:
    close = getattr(stream, "close", None)
    if callable(close):
        close()


def _assemble_chat_completion(chunks: list[Any], model: str) -> Any:
    message: dict[str, Any] = {"role": "assistant", "content": ""}
    response_extensions: dict[str, Any] = {}
    choice_extensions: dict[str, Any] = {}
    finish_reason = "stop"
    completion_id = "route67-weak"
    created = int(time.time())
    system_fingerprint = None
    usage = None

    for chunk in chunks:
        chunk_payload = _dump(chunk)
        if isinstance(chunk_payload, dict):
            for key, value in chunk_payload.items():
                if key not in {
                    "id",
                    "object",
                    "created",
                    "model",
                    "choices",
                    "usage",
                    "system_fingerprint",
                } and value is not None:
                    response_extensions[key] = value
        completion_id = _get(chunk, "id", completion_id) or completion_id
        created = _get(chunk, "created", created) or created
        system_fingerprint = _get(chunk, "system_fingerprint", system_fingerprint)
        usage = _get(chunk, "usage", usage)
        choices = _get(chunk, "choices", []) or []
        if not choices:
            continue
        choice = choices[0]
        delta = _get(choice, "delta")
        delta_payload = _dump(delta)
        if isinstance(delta_payload, dict):
            message = _merge_stream_value(message, delta_payload)
        choice_payload = _dump(choice)
        if isinstance(choice_payload, dict):
            for key, value in choice_payload.items():
                if key not in {"index", "delta", "finish_reason"} and value is not None:
                    choice_extensions[key] = value
        finish_reason = _get(choice, "finish_reason", finish_reason) or finish_reason

    payload = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
                "logprobs": None,
                **choice_extensions,
            }
        ],
        "usage": _dump(usage) if usage is not None else None,
        "system_fingerprint": system_fingerprint,
        **response_extensions,
    }
    try:
        from openai.types.chat import ChatCompletion

        return ChatCompletion.model_validate(payload)
    except (ImportError, TypeError, ValueError):
        # Compatible providers can add values before the OpenAI SDK schema knows
        # about them. Preserve the response shape instead of rejecting the data.
        return _namespace(payload)


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _dump(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)
    attributes = getattr(value, "__dict__", None)
    return dict(attributes) if isinstance(attributes, dict) else value


def _merge_stream_value(current: Any, incoming: Any, key: str | None = None) -> Any:
    """Merge streamed OpenAI-format deltas without dropping provider extensions."""
    if incoming is None:
        return current
    if current is None:
        return incoming
    if isinstance(current, dict) and isinstance(incoming, dict):
        merged = dict(current)
        for child_key, value in incoming.items():
            merged[child_key] = _merge_stream_value(
                merged.get(child_key), value, child_key
            )
        return merged
    if isinstance(current, list) and isinstance(incoming, list):
        return _merge_stream_lists(current, incoming)
    if isinstance(current, str) and isinstance(incoming, str):
        if key in {"role", "name"}:
            return incoming
        return current + incoming
    return incoming


def _merge_stream_lists(current: list[Any], incoming: list[Any]) -> list[Any]:
    merged = list(current)
    positions = {
        item.get("index"): position
        for position, item in enumerate(merged)
        if isinstance(item, dict) and item.get("index") is not None
    }
    for item in incoming:
        if isinstance(item, dict) and item.get("index") in positions:
            position = positions[item["index"]]
            merged[position] = _merge_stream_value(merged[position], item)
        else:
            merged.append(item)
            if isinstance(item, dict) and item.get("index") is not None:
                positions[item["index"]] = len(merged) - 1
    return merged


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value
