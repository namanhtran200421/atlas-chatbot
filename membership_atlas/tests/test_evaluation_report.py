import pytest

from membership_rag.evaluation.evaluator import RetrievalEvaluator
from membership_rag.evaluation.models import EvaluationCase
from membership_rag.evaluation.report import (
    EvaluationThresholds,
    build_evaluation_report,
)


class UnusedRetriever:
    def retrieve(self, *args, **kwargs):
        raise AssertionError("not used")


def make_case(query_id: str, *, answerable: bool = True) -> EvaluationCase:
    return EvaluationCase(
        query_id=query_id,
        query="A normal query" if answerable else "What is the secret payroll password?",
        category="test",
        expected_document_ids=("doc",) if answerable else (),
        allowed_access_classes=("public",),
        answerable=answerable,
        relevance_grades={"doc": 3} if answerable else {},
    )


def relaxed_thresholds() -> EvaluationThresholds:
    return EvaluationThresholds(
        minimum_total_cases=1,
        minimum_answerable_cases=1,
        minimum_unanswerable_cases=0,
        minimum_hit_at_1=0,
        minimum_hit_at_5=0,
        minimum_mrr=0,
        minimum_ndcg_at_5=0,
        minimum_abstention_rate=0,
        maximum_p95_latency_ms=1_000,
    )


def test_request_error_counts_as_zero_relevance() -> None:
    case = make_case("q1")
    evaluator = RetrievalEvaluator(UnusedRetriever())
    result = evaluator.error_result(case, 20.0, "AWS failed")
    report = build_evaluation_report([case], [result], relaxed_thresholds())
    assert report.metrics["hit_at_1"] == 0
    assert report.metrics["request_errors"] == 1
    assert not report.passed


def test_guardrail_block_counts_as_abstention() -> None:
    case = make_case("q2", answerable=False)
    evaluator = RetrievalEvaluator(UnusedRetriever())
    result = evaluator.evaluate_chunks(case, [], latency_ms=0)
    thresholds = EvaluationThresholds(
        minimum_total_cases=1,
        minimum_answerable_cases=0,
        minimum_unanswerable_cases=1,
        minimum_hit_at_1=0,
        minimum_hit_at_5=0,
        minimum_mrr=0,
        minimum_ndcg_at_5=0,
        minimum_abstention_rate=1,
    )
    report = build_evaluation_report([case], [result], thresholds)
    assert report.metrics["abstention_rate"] == 1
    assert report.passed


def test_invalid_rate_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        EvaluationThresholds(minimum_hit_at_1=1.1)
