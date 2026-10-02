"""Production thresholds and aggregate evaluation reports."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from membership_rag.evaluation.models import EvaluationCase, QueryEvaluation
from membership_rag.guardrails import inspect_query


@dataclass(frozen=True, slots=True)
class EvaluationThresholds:
    minimum_total_cases: int = 60
    minimum_answerable_cases: int = 30
    minimum_unanswerable_cases: int = 20
    minimum_hit_at_1: float = 0.90
    minimum_hit_at_5: float = 0.95
    minimum_mrr: float = 0.90
    minimum_ndcg_at_5: float = 0.90
    minimum_abstention_rate: float = 1.0
    maximum_p95_latency_ms: float = 1_000.0
    maximum_request_errors: int = 0
    maximum_acl_leakage: int = 0
    maximum_metadata_errors: int = 0

    def __post_init__(self) -> None:
        counts = (
            self.minimum_total_cases,
            self.minimum_answerable_cases,
            self.minimum_unanswerable_cases,
            self.maximum_request_errors,
            self.maximum_acl_leakage,
            self.maximum_metadata_errors,
        )
        if any(value < 0 for value in counts):
            raise ValueError("evaluation count thresholds cannot be negative")

        rates = (
            self.minimum_hit_at_1,
            self.minimum_hit_at_5,
            self.minimum_mrr,
            self.minimum_ndcg_at_5,
            self.minimum_abstention_rate,
        )
        if any(not 0 <= value <= 1 for value in rates):
            raise ValueError("evaluation rate thresholds must be between 0 and 1")
        if self.maximum_p95_latency_ms <= 0:
            raise ValueError("maximum_p95_latency_ms must be greater than zero")

    @classmethod
    def load(cls, path: Path) -> EvaluationThresholds:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    passed: bool
    failures: tuple[str, ...]
    metrics: dict[str, int | float]
    thresholds: EvaluationThresholds

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "metrics": self.metrics,
            "thresholds": asdict(self.thresholds),
        }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _mean_score(
    results: Sequence[QueryEvaluation],
    getter: Callable[[QueryEvaluation], float | None],
) -> float:
    # Errors receive zero. They must not disappear from the quality denominator.
    if not results:
        return 0.0
    return mean(0.0 if result.error else float(getter(result) or 0.0) for result in results)


def build_evaluation_report(
    cases: Sequence[EvaluationCase],
    results: Sequence[QueryEvaluation],
    thresholds: EvaluationThresholds,
) -> EvaluationReport:
    if len(cases) != len(results):
        raise ValueError("cases and results must have the same length")

    answerable = [result for result in results if result.answerable]
    unanswerable_pairs = [
        (case, result)
        for case, result in zip(cases, results, strict=True)
        if not case.answerable
    ]
    abstained = sum(
        not inspect_query(case.query).allowed or not result.retrieved_results
        for case, result in unanswerable_pairs
    )
    unanswerable_count = len(unanswerable_pairs)
    retrieval_latencies = [
        result.latency_ms
        for case, result in zip(cases, results, strict=True)
        if inspect_query(case.query).allowed
    ]

    metrics: dict[str, int | float] = {
        "total_cases": len(cases),
        "answerable_cases": len(answerable),
        "unanswerable_cases": unanswerable_count,
        "hit_at_1": _mean_score(answerable, lambda result: result.hit_at_k.get(1)),
        "hit_at_5": _mean_score(answerable, lambda result: result.hit_at_k.get(5)),
        "mrr": _mean_score(answerable, lambda result: result.reciprocal_rank),
        "ndcg_at_5": _mean_score(
            answerable, lambda result: result.ndcg_at_k.get(5)
        ),
        "abstention_rate": (
            abstained / unanswerable_count if unanswerable_count else 0.0
        ),
        # Blocked requests make no AWS call and must not dilute retrieval latency.
        "p50_latency_ms": _percentile(retrieval_latencies, 0.50),
        "p95_latency_ms": _percentile(retrieval_latencies, 0.95),
        "request_errors": sum(result.error is not None for result in results),
        "acl_leakage": sum(result.unauthorized_result_count for result in results),
        "metadata_errors": sum(len(result.metadata_errors) for result in results),
    }

    checks = {
        "not enough total cases": metrics["total_cases"]
        >= thresholds.minimum_total_cases,
        "not enough answerable cases": metrics["answerable_cases"]
        >= thresholds.minimum_answerable_cases,
        "not enough unanswerable cases": metrics["unanswerable_cases"]
        >= thresholds.minimum_unanswerable_cases,
        "Hit@1 below threshold": metrics["hit_at_1"]
        >= thresholds.minimum_hit_at_1,
        "Hit@5 below threshold": metrics["hit_at_5"]
        >= thresholds.minimum_hit_at_5,
        "MRR below threshold": metrics["mrr"] >= thresholds.minimum_mrr,
        "nDCG@5 below threshold": metrics["ndcg_at_5"]
        >= thresholds.minimum_ndcg_at_5,
        "abstention rate below threshold": metrics["abstention_rate"]
        >= thresholds.minimum_abstention_rate,
        "p95 latency above threshold": metrics["p95_latency_ms"]
        <= thresholds.maximum_p95_latency_ms,
        "too many request errors": metrics["request_errors"]
        <= thresholds.maximum_request_errors,
        "ACL leakage detected": metrics["acl_leakage"]
        <= thresholds.maximum_acl_leakage,
        "metadata errors detected": metrics["metadata_errors"]
        <= thresholds.maximum_metadata_errors,
    }
    failures = tuple(message for message, passed in checks.items() if not passed)
    return EvaluationReport(not failures, failures, metrics, thresholds)
