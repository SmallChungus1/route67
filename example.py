"""Route67 usage with open router example"""
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
            "Solve a difficult reasoning or math problem",
            "strong_model",
            notes="Requires careful multi-step reasoning.",
        ),
    ],
    weak_model=ModelSpec(
        "google/gemma-3-4b-it",
        usage_notes="Best for straightforward factual and writing questions.",
    ),
    strong_model=ModelSpec(
        "deepseek/deepseek-v4-flash",
        usage_notes="Use for difficult reasoning, mathematics, and verification.",
    ),
)

client = Controller(config, openai_client=openrouter)
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "What are some interesting facts about South Korea?",
        }
    ],
    extra_body={"reasoning": {"enabled": True}},
)

print(f"Response model name: {response.model}")
print(response.choices[0].message.content)
