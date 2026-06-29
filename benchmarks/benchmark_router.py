"""Offline benchmark for route67 semantic routing."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_router.config import RoutingTableEntry
from llm_router.semantic import Embedder, RoutingTable


@dataclass(frozen=True, slots=True)
class EvalExample:
    id: int
    query: str
    expected_target: str
    category: str
    bucket: str
    tags: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class EvalResult:
    example: EvalExample
    score: float
    matched_query: str | None
    matched_target: str | None
    routed_target: str | None
    predicted_target: str


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    routing_table = load_routing_table(args.routing_table)

    embedder = Embedder(args.embedding_model)
    table = RoutingTable(routing_table, embedder, args.embedding_cache_path)

    if args.threshold_sweep:
        run_threshold_sweep(dataset, table, args)
        return

    results = evaluate_dataset(
        dataset,
        table,
        threshold=args.threshold,
        fallback_target=args.fallback_target,
    )
    print_summary(
        results,
        args.threshold,
        args.fallback_target,
        routing_table,
        args.show_mistakes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark route67's semantic router against a labeled dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("benchmarks/router_eval_dataset.jsonl"),
        help="JSONL file with queries and expected targets.",
    )
    parser.add_argument(
        "--routing-table",
        type=Path,
        default=Path("benchmarks/router_seed_table.json"),
        help="JSON file with routing-table seed entries used for evaluation.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Similarity threshold used to route to a table entry.",
    )
    parser.add_argument(
        "--fallback-target",
        choices=("weak_model", "strong_model"),
        default="weak_model",
        help="Target to assume when no table entry clears the threshold.",
    )
    parser.add_argument(
        "--threshold-sweep",
        action="store_true",
        help="Evaluate a configurable range of thresholds and report the best one.",
    )
    parser.add_argument(
        "--threshold-sweep-start",
        type=float,
        default=0.05,
        help="Lower bound for threshold sweep, inclusive.",
    )
    parser.add_argument(
        "--threshold-sweep-end",
        type=float,
        default=0.95,
        help="Upper bound for threshold sweep, inclusive.",
    )
    parser.add_argument(
        "--threshold-sweep-step",
        type=float,
        default=0.01,
        help="Step size for threshold sweep.",
    )
    parser.add_argument(
        "--optimize-metric",
        choices=(
            "strong_f1",
            "accuracy",
            "balanced_accuracy",
            "strong_recall",
            "strong_precision",
        ),
        default="strong_f1",
        help="Metric used to pick the best threshold during a sweep.",
    )
    parser.add_argument(
        "--embedding-model",
        default="minishlab/potion-base-8M",
        help="Embedding model to use for semantic retrieval.",
    )
    parser.add_argument(
        "--embedding-cache-path",
        default=".cache/benchmark_routes",
        help="Optional cache path for routing-table embeddings.",
    )
    parser.add_argument(
        "--show-mistakes",
        type=int,
        default=10,
        help="Number of misrouted examples to print.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            examples.append(
                EvalExample(
                    id=int(payload["id"]),
                    query=str(payload["query"]),
                    expected_target=str(payload["expected_target"]),
                    category=str(payload["category"]),
                    bucket=str(payload["bucket"]),
                    tags=tuple(str(tag) for tag in payload.get("tags", [])),
                    reason=str(payload["reason"]),
                )
            )
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def load_routing_table(path: Path) -> list[RoutingTableEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        RoutingTableEntry(
            query=str(item["query"]),
            target=str(item["target"]),
            notes=None if item.get("notes") is None else str(item["notes"]),
        )
        for item in payload
    ]


def evaluate_dataset(
    examples: list[EvalExample],
    table: RoutingTable,
    *,
    threshold: float,
    fallback_target: str,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for example in examples:
        entry, score = table.best_match(example.query)
        routed_target = entry.target if entry is not None and score >= threshold else None
        predicted_target = routed_target or fallback_target
        results.append(
            EvalResult(
                example=example,
                score=score,
                matched_query=None if entry is None else entry.query,
                matched_target=None if entry is None else entry.target,
                routed_target=routed_target,
                predicted_target=predicted_target,
            )
        )
    return results


def run_threshold_sweep(
    examples: list[EvalExample],
    table: RoutingTable,
    args: argparse.Namespace,
) -> None:
    thresholds = build_thresholds(
        args.threshold_sweep_start,
        args.threshold_sweep_end,
        args.threshold_sweep_step,
    )
    print(
        "threshold\taccuracy\tbalanced\tstrong_f1\tprecision\trecall\tcoverage"
    )
    best_threshold: float | None = None
    best_metrics: dict[str, float] | None = None
    best_results: list[EvalResult] | None = None
    best_candidate: tuple[float, float, float] | None = None
    for threshold in thresholds:
        results = evaluate_dataset(
            examples,
            table,
            threshold=threshold,
            fallback_target=args.fallback_target,
        )
        metrics = summarize_results(results)
        print(
            f"{threshold:.2f}\t"
            f"{metrics['accuracy']:.3f}\t"
            f"{metrics['balanced_accuracy']:.3f}\t"
            f"{metrics['strong_f1']:.3f}\t"
            f"{metrics['strong_precision']:.3f}\t"
            f"{metrics['strong_recall']:.3f}\t"
            f"{metrics['coverage']:.3f}"
        )
        candidate = (
            metrics[args.optimize_metric],
            metrics["accuracy"],
            metrics["strong_recall"],
        )
        if best_candidate is None or candidate > best_candidate:
            best_candidate = candidate
            best_threshold = threshold
            best_metrics = metrics
            best_results = results

    if best_threshold is None or best_metrics is None or best_results is None:
        raise RuntimeError("Threshold sweep produced no results")

    print(
        "\nBest threshold "
        f"by {args.optimize_metric}: {best_threshold:.2f}"
    )
    print(
        "Best metrics: "
        f"accuracy={best_metrics['accuracy']:.3f}, "
        f"balanced_accuracy={best_metrics['balanced_accuracy']:.3f}, "
        f"strong_f1={best_metrics['strong_f1']:.3f}, "
        f"strong_precision={best_metrics['strong_precision']:.3f}, "
        f"strong_recall={best_metrics['strong_recall']:.3f}, "
        f"coverage={best_metrics['coverage']:.3f}"
    )
    print_summary(
        best_results,
        best_threshold,
        args.fallback_target,
        load_routing_table(args.routing_table),
        args.show_mistakes,
    )


def build_thresholds(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("threshold sweep step must be positive")
    if end < start:
        raise ValueError("threshold sweep end must be greater than or equal to start")
    thresholds: list[float] = []
    value = start
    while value <= end + (step / 10):
        thresholds.append(round(value, 10))
        value += step
    return thresholds


def summarize_results(results: list[EvalResult]) -> dict[str, float]:
    total = len(results)
    correct = sum(
        1 for result in results if result.predicted_target == result.example.expected_target
    )
    covered = sum(1 for result in results if result.routed_target is not None)
    true_positive = sum(
        1
        for result in results
        if result.predicted_target == "strong_model"
        and result.example.expected_target == "strong_model"
    )
    predicted_positive = sum(
        1 for result in results if result.predicted_target == "strong_model"
    )
    actual_positive = sum(
        1 for result in results if result.example.expected_target == "strong_model"
    )
    true_negative = sum(
        1
        for result in results
        if result.predicted_target == "weak_model"
        and result.example.expected_target == "weak_model"
    )
    actual_negative = sum(
        1 for result in results if result.example.expected_target == "weak_model"
    )
    strong_precision = true_positive / predicted_positive if predicted_positive else 0.0
    strong_recall = true_positive / actual_positive if actual_positive else 0.0
    weak_recall = true_negative / actual_negative if actual_negative else 0.0
    strong_f1 = (
        2 * strong_precision * strong_recall / (strong_precision + strong_recall)
        if (strong_precision + strong_recall)
        else 0.0
    )
    return {
        "accuracy": correct / total,
        "coverage": covered / total,
        "balanced_accuracy": (strong_recall + weak_recall) / 2,
        "strong_precision": strong_precision,
        "strong_recall": strong_recall,
        "strong_f1": strong_f1,
    }


def print_summary(
    results: list[EvalResult],
    threshold: float,
    fallback_target: str,
    routing_table: list[RoutingTableEntry],
    show_mistakes: int,
) -> None:
    metrics = summarize_results(results)
    print(f"Dataset size: {len(results)}")
    print(f"Seed routes: {len(routing_table)}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Fallback target: {fallback_target}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"Coverage: {metrics['coverage']:.3f}")
    print(f"Strong F1: {metrics['strong_f1']:.3f}")
    print(f"Strong precision: {metrics['strong_precision']:.3f}")
    print(f"Strong recall: {metrics['strong_recall']:.3f}")

    print("\nConfusion matrix:")
    confusion = Counter(
        (result.example.expected_target, result.predicted_target) for result in results
    )
    for expected in ("weak_model", "strong_model"):
        for predicted in ("weak_model", "strong_model"):
            print(f"  expected={expected:<12} predicted={predicted:<12} count={confusion[(expected, predicted)]}")

    print("\nCategory accuracy:")
    by_category: dict[str, list[EvalResult]] = defaultdict(list)
    for result in results:
        by_category[result.example.category].append(result)
    for category in sorted(by_category):
        category_results = by_category[category]
        category_accuracy = sum(
            1
            for result in category_results
            if result.predicted_target == result.example.expected_target
        ) / len(category_results)
        print(f"  {category:<16} {category_accuracy:.3f} ({len(category_results)})")

    mistakes = [
        result
        for result in results
        if result.predicted_target != result.example.expected_target
    ]
    if not mistakes:
        return

    print("\nSample mistakes:")
    for result in mistakes[:show_mistakes]:
        print(f"- id={result.example.id} category={result.example.category} score={result.score:.3f}")
        print(f"  query: {result.example.query}")
        print(
            "  expected: "
            f"{result.example.expected_target} predicted: {result.predicted_target}"
        )
        print(
            "  matched route: "
            f"{result.matched_target or 'none'} | {result.matched_query or 'none'}"
        )
        print(f"  rationale: {result.example.reason}")


if __name__ == "__main__":
    main()
