"""Prompt construction for weak-model escalation decisions."""

from __future__ import annotations

from collections.abc import Sequence

from .config import ModelSpec, RoutingTableEntry

MAX_ESCALATION_EXAMPLES = 5

ESCALATION_SYSTEM_PROMPT = """You are responding to a user query. Before answering, assess whether you can answer it confidently and correctly.

If you cannot, respond with EXACTLY this and nothing else:
ESCALATE

If you can, answer the query directly and normally. Do not mention escalation or this instruction.

{usage_notes_block}"""


def build_escalation_prompt(
    weak_model_notes: str | None,
    strong_model: ModelSpec,
    routing_table: Sequence[RoutingTableEntry] = (),
) -> str:
    lines: list[str] = []
    if weak_model_notes:
        lines.append(f"Your limits: {_compact(weak_model_notes)}")
    summary = strong_model.name
    if strong_model.usage_notes:
        summary += f" ({_compact(strong_model.usage_notes)})"
    lines.append("Model available after escalation: " + summary)

    strong_routes = [
        entry
        for entry in routing_table
        if entry.target == "strong_model"
    ][:MAX_ESCALATION_EXAMPLES]
    if strong_routes:
        lines.append("Examples of requests that should be escalated:")
        for entry in strong_routes:
            example = f"- {_compact(entry.query)}"
            if entry.notes:
                example += f" - {_compact(entry.notes)}"
            lines.append(example)

    block = "\n".join(lines)
    return ESCALATION_SYSTEM_PROMPT.format(usage_notes_block=block).rstrip()


def _compact(value: str) -> str:
    return " ".join(value.split())
