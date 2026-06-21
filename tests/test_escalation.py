from __future__ import annotations

import unittest
from types import SimpleNamespace

from llm_router.config import ModelSpec, RouterConfig, RoutingTableEntry
from llm_router.escalation import run_with_escalation
from tests.helpers import FakeClient, FakeCompletions, FakeStream, chunk


def config() -> RouterConfig:
    return RouterConfig(
        routing_table=[
            RoutingTableEntry(
                "Prove a difficult theorem",
                "strong_model",
                "Requires rigorous multi-step reasoning.",
            )
        ],
        weak_model=ModelSpec("weak", "Weak at proofs."),
        strong_model=ModelSpec("strong", "Use for proofs."),
    )


class EscalationTests(unittest.TestCase):
    def test_weak_answer_is_returned_without_restart(self) -> None:
        stream = FakeStream([chunk("Hello"), chunk(" world", finish_reason="stop")])
        completions = FakeCompletions(stream)

        result = run_with_escalation(
            FakeClient(completions),
            config(),
            [{"role": "user", "content": "Hi"}],
        )

        self.assertFalse(result.escalated)
        self.assertEqual(result.used_model, "weak")
        self.assertEqual(result.response.choices[0].message.content, "Hello world")
        self.assertEqual(len(completions.calls), 1)
        self.assertTrue(stream.closed)
        system_prompt = completions.calls[0]["messages"][0]["content"]
        self.assertIn("Your limits: Weak at proofs.", system_prompt)
        self.assertIn("- Prove a difficult theorem", system_prompt)
        self.assertIn("Requires rigorous multi-step reasoning.", system_prompt)

    def test_sentinel_closes_stream_and_calls_strong_model(self) -> None:
        stream = FakeStream([chunk("  esc"), chunk("ALATE"), chunk("discard me")])
        strong_response = SimpleNamespace(model="strong", answer="done")
        completions = FakeCompletions(stream, strong_response)
        messages = [{"role": "user", "content": "Hard question"}]

        result = run_with_escalation(FakeClient(completions), config(), messages)

        self.assertTrue(result.escalated)
        self.assertIs(result.response, strong_response)
        self.assertTrue(stream.closed)
        self.assertEqual(stream.consumed, 2)
        self.assertEqual(completions.calls[1]["model"], "strong")
        self.assertEqual(completions.calls[1]["messages"], messages)

    def test_sentinel_is_case_insensitive_with_leading_whitespace(self) -> None:
        stream = FakeStream([chunk("\n\tEsCaLaTe\n")])
        result = run_with_escalation(
            FakeClient(FakeCompletions(stream)),
            config(),
            [{"role": "user", "content": "Hard"}],
        )
        self.assertTrue(result.escalated)

    def test_provider_options_are_forwarded_to_weak_and_strong_models(self) -> None:
        stream = FakeStream([chunk("ESCALATE")])
        completions = FakeCompletions(stream, SimpleNamespace(model="strong"))
        options = {
            "extra_body": {"reasoning": {"enabled": True}},
            "extra_headers": {"X-Provider": "example"},
        }

        run_with_escalation(
            FakeClient(completions),
            config(),
            [{"role": "user", "content": "Hard"}],
            request_kwargs=options,
        )

        self.assertEqual(completions.calls[0]["extra_body"], options["extra_body"])
        self.assertEqual(completions.calls[1]["extra_body"], options["extra_body"])
        self.assertEqual(
            completions.calls[0]["extra_headers"], options["extra_headers"]
        )
        self.assertEqual(
            completions.calls[1]["extra_headers"], options["extra_headers"]
        )

    def test_weak_response_preserves_provider_specific_message_fields(self) -> None:
        stream = FakeStream(
            [
                SimpleNamespace(
                    id="openrouter-test",
                    created=123,
                    provider="Example Provider",
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                role="assistant",
                                content="There are ",
                                reasoning_details=[
                                    {"type": "reasoning.text", "text": "Count "}
                                ],
                            ),
                            finish_reason=None,
                        )
                    ],
                ),
                SimpleNamespace(
                    id="openrouter-test",
                    created=123,
                    provider="Example Provider",
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content="three.",
                                reasoning_details=[
                                    {"type": "reasoning.text", "text": "carefully."}
                                ],
                            ),
                            finish_reason="stop",
                        )
                    ],
                ),
            ]
        )

        result = run_with_escalation(
            FakeClient(FakeCompletions(stream)),
            config(),
            [{"role": "user", "content": "How many r's?"}],
        )

        message = result.response.choices[0].message
        self.assertEqual(message.content, "There are three.")
        self.assertEqual(
            message.reasoning_details,
            [
                {"type": "reasoning.text", "text": "Count "},
                {"type": "reasoning.text", "text": "carefully."},
            ],
        )
        self.assertEqual(result.response.provider, "Example Provider")

    def test_indexed_tool_call_fragments_are_assembled(self) -> None:
        stream = FakeStream(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta={
                                "content": "",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "weather",
                                            "arguments": '{"city":"',
                                        },
                                    }
                                ],
                            },
                            finish_reason=None,
                        )
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta={
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": 'Paris"}'},
                                    }
                                ]
                            },
                            finish_reason="tool_calls",
                        )
                    ]
                ),
            ]
        )

        result = run_with_escalation(
            FakeClient(FakeCompletions(stream)),
            config(),
            [{"role": "user", "content": "Weather?"}],
        )

        call = result.response.choices[0].message.tool_calls[0]
        self.assertEqual(call.function.arguments, '{"city":"Paris"}')

    def test_provider_extension_outside_openai_schema_is_still_returned(self) -> None:
        stream = FakeStream(
            [
                SimpleNamespace(
                    provider="Example Provider",
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="Partial result"),
                            finish_reason="provider_specific_reason",
                        )
                    ],
                )
            ]
        )

        result = run_with_escalation(
            FakeClient(FakeCompletions(stream)),
            config(),
            [{"role": "user", "content": "Hello"}],
        )

        self.assertEqual(
            result.response.choices[0].finish_reason,
            "provider_specific_reason",
        )
        self.assertEqual(result.response.provider, "Example Provider")


if __name__ == "__main__":
    unittest.main()
