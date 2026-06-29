from __future__ import annotations

import unittest

from llm_router.config import ModelSpec, RoutingTableEntry
from llm_router.routing import build_escalation_prompt


class EscalationPromptTests(unittest.TestCase):
    def test_includes_model_usage_notes_and_strong_routes(self) -> None:
        prompt = build_escalation_prompt(
            "Avoid multi-step mathematics.",
            ModelSpec("strong", "Use for difficult reasoning."),
            [
                RoutingTableEntry("Rewrite this paragraph", "weak_model"),
                RoutingTableEntry(
                    "Prove this theorem",
                    "strong_model",
                    "Requires a rigorous proof.",
                ),
            ],
        )

        self.assertIn("Your limits: Avoid multi-step mathematics.", prompt)
        self.assertIn("strong (Use for difficult reasoning.)", prompt)
        self.assertIn("- Prove this theorem - Requires a rigorous proof.", prompt)
        self.assertNotIn("Rewrite this paragraph", prompt)

    def test_caps_escalation_examples_at_five(self) -> None:
        prompt = build_escalation_prompt(
            None,
            ModelSpec("strong"),
            [
                RoutingTableEntry(f"Strong route {index}", "strong_model")
                for index in range(6)
            ],
        )

        for index in range(5):
            self.assertIn(f"- Strong route {index}", prompt)
        self.assertNotIn("- Strong route 5", prompt)

    def test_omits_examples_section_without_strong_routes(self) -> None:
        prompt = build_escalation_prompt(
            None,
            ModelSpec("strong"),
            [RoutingTableEntry("Simple rewrite", "weak_model")],
        )

        self.assertNotIn("Examples of requests that should be escalated:", prompt)


if __name__ == "__main__":
    unittest.main()
