from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def chunk(
    content: str = "",
    *,
    finish_reason: str | None = None,
    role: str | None = None,
) -> Any:
    return SimpleNamespace(
        id="chatcmpl-test",
        created=123,
        system_fingerprint=None,
        usage=None,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, role=role),
                finish_reason=finish_reason,
            )
        ],
    )


class FakeStream:
    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks
        self.consumed = 0
        self.closed = False

    def __iter__(self):
        for item in self.chunks:
            self.consumed += 1
            yield item

    def close(self) -> None:
        self.closed = True


class FakeCompletions:
    def __init__(self, weak_stream: FakeStream, strong_response: Any = None) -> None:
        self.weak_stream = weak_stream
        self.strong_response = strong_response or SimpleNamespace(model="strong")
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return self.weak_stream
        return self.strong_response


class FakeClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)

