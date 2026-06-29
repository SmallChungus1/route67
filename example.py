"""Minimal route67 example using the default OpenAI client."""

import os

from openai import OpenAI
from llm_router import Controller, ModelSpec, RouterConfig, RoutingTableEntry

openrouter = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

config = RouterConfig(
    routing_table=[
        RoutingTableEntry(
            "Answer questions about a country",
            "weak_model",
        ),
        RoutingTableEntry(
            "How to derive the Boolean Satisfiability, a NP problem",
            "strong_model",
            notes="Requires careful multi-step reasoning.",
        ),
    ],
    weak_model=ModelSpec(
        "liquid/lfm-2-24b-a2b",
        usage_notes="Best for straightforward factual and writing questions.",
    ),
    strong_model=ModelSpec(
        "deepseek/deepseek-v4-flash",
        usage_notes="Use for difficult reasoning, mathematics, and verification.",
    ),
    similarity_threshold=0.11 #obtained from running bechmark_router.py to find best threshold
)

client = Controller(config, openai_client=openrouter)
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "How to solve the NP problem of Traveling salesman",
        }
    ],
    extra_body={"reasoning": {"enabled": True}},
)

print(f"Response model name: {response.model}")
print(response.choices[0].message.content)