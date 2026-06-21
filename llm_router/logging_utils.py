"""Structured JSONL routing-decision logging."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    method: Literal["table_match", "weak_model_direct", "escalated"]
    model: str
    score: float


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

