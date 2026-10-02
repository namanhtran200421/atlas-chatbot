"""Evaluate retrieval results against labelled query cases."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import Protocol

from membership_rag.bedrock.retrieval import RetrievedChunk
from membership_rag.evaluation.metrics import (
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from membership_rag.evaluation.models import (
    EvaluationCase,
    QueryEvaluation,
    RankedRetrievalResult,
)

DEFAULT_K_VALUES = (1, 3, 5, 10, 20)
REQUIRED_METADATA_FIELDS = ("document_id", "access_class", "content_type")


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        allowed_access_classes: Sequence[str],
        number_of_results: int,
    ) -> list[RetrievedChunk]: ...


class RetrievalEvaluator:
    def __init__(
        self,
        retriever: Retriever,
        k_values: Sequence[int] = DEFAULT_K_VALUES,
    ) -> None:
        if not k_values:
            raise ValueError("k_values must not be empty")
        if any(k <= 0 for k in k_values):
            raise ValueError("all k must be greater than 0")

        self.retriever = retriever
        self.k_values = tuple(sorted(set(k_values)))

    def evaluate_case(self, case: EvaluationCase) -> QueryEvaluation:
        started_at = perf_counter()
        try:
            chunks = self.retriever.retrieve(
                case.query,
                allowed_access_classes=case.allowed_access_classes,
                number_of_results=max(self.k_values),
            )
        # Every retriever failure is data for the production gate. The report
        # records it instead of allowing one exception to hide the other cases.
        except Exception as exc:  # noqa: BLE001
            latency_ms = (perf_counter() - started_at) * 1000
            return self.error_result(case, latency_ms, str(exc))

        latency_ms = (perf_counter() - started_at) * 1000
        return self.evaluate_chunks(case, chunks, latency_ms=latency_ms)

    def evaluate_cases(self, cases: Sequence[EvaluationCase]) -> list[QueryEvaluation]:
        return [self.evaluate_case(case) for case in cases]

    def evaluate_chunks(
        self,
        case: EvaluationCase,
        chunks: Sequence[RetrievedChunk],
        *,
        latency_ms: float,
        error: str | None = None,
    ) -> QueryEvaluation:
        """Score already-retrieved chunks; used by deterministic offline evals."""

        if error is not None:
            return self.error_result(case, latency_ms, error)

        ranked_results = self.build_ranked_results(chunks)
        document_ids = [
            result.document_id
            for result in ranked_results
            if result.document_id is not None
        ]

        hit_scores: dict[int, float | None] = {}
        recall_scores: dict[int, float | None] = {}
        ndcg_scores: dict[int, float | None] = {}

        for k in self.k_values:
            if case.answerable:
                hit_scores[k] = hit_at_k(document_ids, case.expected_document_ids, k)
                recall_scores[k] = recall_at_k(
                    document_ids, case.expected_document_ids, k
                )
            else:
                hit_scores[k] = None
                recall_scores[k] = None

            ndcg_scores[k] = (
                ndcg_at_k(document_ids, case.relevance_grades, k)
                if case.relevance_grades
                else None
            )

        reciprocal_rank_score = (
            reciprocal_rank(document_ids, case.expected_document_ids)
            if case.answerable
            else None
        )

        return QueryEvaluation(
            query_id=case.query_id,
            query=case.query,
            category=case.category,
            answerable=case.answerable,
            latency_ms=latency_ms,
            retrieved_results=ranked_results,
            hit_at_k=hit_scores,
            recall_at_k=recall_scores,
            reciprocal_rank=reciprocal_rank_score,
            ndcg_at_k=ndcg_scores,
            unauthorized_result_count=self.count_unauthorized_results(
                chunks, case.allowed_access_classes
            ),
            metadata_errors=self.find_metadata_errors(chunks),
        )

    @staticmethod
    def build_ranked_results(
        chunks: Sequence[RetrievedChunk],
    ) -> tuple[RankedRetrievalResult, ...]:
        return tuple(
            RankedRetrievalResult(
                rank=rank,
                document_id=chunk.metadata.get("document_id"),
                chunk_id=chunk.metadata.get("chunk_id"),
                score=chunk.score,
                access_class=chunk.metadata.get("access_class"),
                source_type=chunk.metadata.get("source_type"),
                content_type=chunk.metadata.get("content_type"),
                title=chunk.metadata.get("title"),
            )
            for rank, chunk in enumerate(chunks, start=1)
        )

    @staticmethod
    def find_metadata_errors(chunks: Sequence[RetrievedChunk]) -> tuple[str, ...]:
        errors: list[str] = []
        for rank, chunk in enumerate(chunks, start=1):
            for field in REQUIRED_METADATA_FIELDS:
                if not chunk.metadata.get(field):
                    errors.append(f"rank={rank}:missing:{field}")
        return tuple(errors)

    @staticmethod
    def count_unauthorized_results(
        chunks: Sequence[RetrievedChunk],
        allowed_access_classes: Sequence[str],
    ) -> int:
        allowed = set(allowed_access_classes)
        return sum(
            chunk.metadata.get("access_class") not in allowed for chunk in chunks
        )

    def error_result(
        self,
        case: EvaluationCase,
        latency_ms: float,
        error: str,
    ) -> QueryEvaluation:
        empty_scores: dict[int, float | None] = {k: None for k in self.k_values}
        return QueryEvaluation(
            query_id=case.query_id,
            query=case.query,
            category=case.category,
            answerable=case.answerable,
            latency_ms=latency_ms,
            retrieved_results=(),
            hit_at_k=empty_scores.copy(),
            recall_at_k=empty_scores.copy(),
            reciprocal_rank=None,
            ndcg_at_k=empty_scores.copy(),
            unauthorized_result_count=0,
            metadata_errors=(),
            error=error,
        )
