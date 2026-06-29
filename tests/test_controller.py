from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from llm_router.config import ModelSpec, RouterConfig, RoutingTableEntry
from llm_router.controller import Controller, extract_user_query
from llm_router.routing import EscalationResult
from tests.helpers import FakeClient, FakeCompletions, FakeStream


class ConstantEmbedder:
    model_name = "constant"

    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = list(vectors)

    def encode(self, texts: list[str]) -> np.ndarray:
        count = len(texts)
        result = self.vectors[:count]
        self.vectors = self.vectors[count:]
        return np.asarray(result, dtype=np.float32)


def config(**overrides) -> RouterConfig:
    values = {
        "routing_table": [RoutingTableEntry("example", "strong_model")],
        "similarity_threshold": 0.75,
        "weak_model": ModelSpec("weak"),
        "strong_model": ModelSpec("table-model"),
    }
    values.update(overrides)
    return RouterConfig(**values)


class ControllerTests(unittest.TestCase):
    def test_exact_threshold_routes_to_table_model(self) -> None:
        response = SimpleNamespace(model="table-model")
        completions = FakeCompletions(FakeStream([]), response)
        embedder = ConstantEmbedder([[1.0, 0.0], [0.75, 0.6614378]])
        controller = Controller(config(), FakeClient(completions), embedder)

        actual = controller.chat.completions.create(
            model="ignored",
            messages=[{"role": "user", "content": "query"}],
        )

        self.assertIs(actual, response)
        self.assertEqual(completions.calls[0]["model"], "table-model")

    def test_provider_specific_request_options_are_forwarded(self) -> None:
        response = SimpleNamespace(model="table-model")
        completions = FakeCompletions(FakeStream([]), response)
        embedder = ConstantEmbedder([[1.0, 0.0], [1.0, 0.0]])
        controller = Controller(config(), FakeClient(completions), embedder)
        messages = [{"role": "user", "content": "query"}]
        extra_body = {"reasoning": {"enabled": True}}

        controller.chat.completions.create(
            messages=messages,
            extra_body=extra_body,
            extra_headers={"X-Provider": "example"},
        )

        self.assertEqual(completions.calls[0]["messages"], messages)
        self.assertEqual(completions.calls[0]["extra_body"], extra_body)
        self.assertEqual(
            completions.calls[0]["extra_headers"], {"X-Provider": "example"}
        )

    @patch("llm_router.routing.run_with_escalation")
    def test_below_threshold_uses_escalation_path(self, escalate) -> None:
        weak_response = SimpleNamespace(model="weak")
        escalate.return_value = EscalationResult("weak", weak_response, False)
        embedder = ConstantEmbedder([[1.0, 0.0], [0.74, 0.6726069]])
        controller = Controller(
            config(),
            FakeClient(FakeCompletions(FakeStream([]))),
            embedder,
        )

        actual = controller.chat_completions_create(
            messages=[{"role": "user", "content": "query"}]
        )

        self.assertIs(actual, weak_response)
        escalate.assert_called_once()

    def test_last_user_message_is_routed(self) -> None:
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": [{"type": "text", "text": "last"}]},
        ]
        self.assertEqual(extract_user_query(messages), "last")

    def test_non_text_user_content_falls_back_to_stringified_payload(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "https://x"}}],
            }
        ]

        actual = extract_user_query(messages)

        self.assertIn("image_url", actual)

    @patch("llm_router.routing.run_with_escalation")
    def test_decision_is_logged(self, escalate) -> None:
        escalate.return_value = EscalationResult(
            "strong", SimpleNamespace(model="strong"), True
        )
        with tempfile.TemporaryDirectory() as directory:
            log_path = str(Path(directory) / "routing.jsonl")
            embedder = ConstantEmbedder([[1.0, 0.0], [0.0, 1.0]])
            controller = Controller(
                config(log_path=log_path),
                FakeClient(FakeCompletions(FakeStream([]))),
                embedder,
            )
            controller.chat_completions_create(
                messages=[{"role": "user", "content": "query"}]
            )
            record = Path(log_path).read_text(encoding="utf-8")

        self.assertIn('"method": "escalated"', record)
        self.assertIn('"model_used": "strong"', record)

    def test_public_streaming_is_explicitly_rejected(self) -> None:
        controller = Controller(
            config(routing_table=[]),
            FakeClient(FakeCompletions(FakeStream([]))),
            ConstantEmbedder([]),
        )
        with self.assertRaises(NotImplementedError):
            controller.chat_completions_create(
                messages=[{"role": "user", "content": "query"}],
                stream=True,
            )


if __name__ == "__main__":
    unittest.main()
