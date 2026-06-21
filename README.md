# route67

`route67` is a LLM router for OpenAI-compatible chat
completions format. It uses a user-defined routing table for user defined question-model routing via semantic similarity, as a fallback a weak model answer or explicitly escalate to a strong model.

## Install

route67 requires Python 3.10 or newer. Choose either the standard Python workflow
or the `uv` workflow.

### Using `python -m venv`

Create and activate a virtual environment:

```console
python -m venv .venv
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```console
# macOS/Linux
source .venv/bin/activate
```

Then install route67 and its dependencies:

```console
python -m pip install --upgrade pip
python -m pip install -e .
```

To also install the test dependencies, use `python -m pip install -e ".[test]"`.

### Using `uv`

With [`uv`](https://docs.astral.sh/uv/) installed, create the environment and
install the project from the lockfile:

```console
uv sync
```

Run commands inside the environment with `uv run`, for example
`uv run python example.py`. To include test dependencies, use
`uv sync --extra test`.

## Get started

Set an OpenAI API key in your environment:

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY = "your-api-key"
```

```console
# macOS/Linux
export OPENAI_API_KEY="your-api-key"
```

Create `example.py`:

```python
from llm_router import Controller, ModelSpec, RouterConfig, RoutingTableEntry

config = RouterConfig(
    routing_table=[
        RoutingTableEntry(
            "Prove this theorem",
            "strong_model",
            notes="Requires a rigorous multi-step proof.",
        ),
        RoutingTableEntry("Rewrite this paragraph", "weak_model"),
    ],
    weak_model=ModelSpec(
        "gpt-5-mini",
        usage_notes="Avoid difficult multi-step proofs.",
    ),
    strong_model=ModelSpec(
        "gpt-5",
        usage_notes="Use for rigorous proofs and difficult reasoning.",
    ),
    embedding_cache_path=".cache/routes",
    log_path=".cache/routing.jsonl",
)

client = Controller(config)
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Prove that sqrt(2) is irrational."}]
)
print(response.choices[0].message.content)
```

Run it with the activated standard virtual environment:

```console
python example.py
```

Or with `uv`:

```console
uv run python example.py
```

### OpenAI-compatible providers

route67 can use any provider exposed through an OpenAI-compatible client. Create
the provider's client normally and inject it into the controller. Model names in
the routing configuration are passed to that provider unchanged.

For example, with OpenRouter:

```python
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
        "openai/gpt-4.1-mini",
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
            "content": "How many r's are in the word 'strawberry'?",
        }
    ],
    extra_body={"reasoning": {"enabled": True}},
)
```

Provider-specific request options such as `extra_body` and `extra_headers` are
forwarded unchanged. Provider-specific response fields, including
`reasoning_details`, are also preserved. To continue a provider's reasoning,
pass its assistant message fields back unmodified in the next request.

Routing table entries target only `"weak_model"` or `"strong_model"`. Provider
model names live in `ModelSpec`, so switching models or providers does not
require rewriting the routing table.

`ModelSpec.usage_notes` are added to the weak model's escalation system prompt.
The prompt also includes up to five routing-table entries targeting
`"strong_model"` as examples of requests that should be escalated. Add concise
`notes` to those entries when the reason for escalation is useful context.

Your first request will download the `minishlab/potion-base-8M` from HuggingFace. The model is lazy-loaded,
so constructing a controller with an empty routing table does not download it.
