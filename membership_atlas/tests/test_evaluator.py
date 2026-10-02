from membership_rag.bedrock.retrieval import (
    RetrievedChunk,
)
from membership_rag.evaluation.evaluator import (
    RetrievalEvaluator,
)
from membership_rag.evaluation.models import (
    EvaluationCase,
)


class FakeRetriever:
    def __init__(
        self,
        results,
    ) -> None:
        self.results = results

        self.calls = []

    def retrieve(
        self,
        query,
        *,
        allowed_access_classes,
        number_of_results,
    ):
        self.calls.append(
            {
                "query": query,
                "allowed_access_classes": (
                    allowed_access_classes
                ),
                "number_of_results": (
                    number_of_results
                ),
            }
        )

        return self.results


def make_case() -> EvaluationCase:
    return EvaluationCase(
        query_id="q001",
        query="What is Buddhism?",
        category="anthropedia",
        expected_document_ids=(
            "doc_buddhism",
        ),
        expected_source_types=(
            "anthropedia",
        ),
        expected_content_types=(
            "anthropedia",
        ),
        allowed_access_classes=(
            "member_restricted",
        ),
        answerable=True,
        relevance_grades={
            "doc_buddhism": 3,
        },
    )


def make_chunk(
    document_id: str,
    access_class: str = (
        "member_restricted"
    ),
) -> RetrievedChunk:
    return RetrievedChunk(
        text="example",
        score=0.9,
        metadata={
            "document_id": document_id,
            "chunk_id": (
                f"chunk_{document_id}"
            ),
            "access_class": access_class,
            "source_type": "anthropedia",
            "content_type": "anthropedia",
            "title": "Example",
        },
        document_id=document_id,
        source_uri=None,
    )


def test_evaluator_retrieves_only_once() -> None:
    retriever = FakeRetriever(
        [
            make_chunk(
                "doc_buddhism"
            ),
        ]
    )

    evaluator = RetrievalEvaluator(
        retriever
    )

    evaluator.evaluate_case(
        make_case()
    )

    assert len(
        retriever.calls
    ) == 1

    assert (
        retriever.calls[0][
            "number_of_results"
        ]
        == 20
    )


def test_evaluator_calculates_metrics() -> None:
    retriever = FakeRetriever(
        [
            make_chunk(
                "wrong_doc"
            ),
            make_chunk(
                "doc_buddhism"
            ),
        ]
    )

    evaluator = RetrievalEvaluator(
        retriever
    )

    result = (
        evaluator.evaluate_case(
            make_case()
        )
    )

    assert (
        result.hit_at_k[1]
        == 0.0
    )

    assert (
        result.hit_at_k[3]
        == 1.0
    )

    assert (
        result.reciprocal_rank
        == 0.5
    )


def test_evaluator_detects_unauthorized_result() -> None:
    retriever = FakeRetriever(
        [
            make_chunk(
                "doc_buddhism",
                access_class="public",
            ),
        ]
    )

    evaluator = RetrievalEvaluator(
        retriever
    )

    result = (
        evaluator.evaluate_case(
            make_case()
        )
    )

    assert (
        result.unauthorized_result_count
        == 1
    )


def test_evaluator_detects_missing_metadata() -> None:
    chunk = RetrievedChunk(
        text="example",
        score=0.9,
        metadata={
            "document_id": "doc_buddhism",
        },
        document_id="doc_buddhism",
        source_uri=None,
    )

    retriever = FakeRetriever(
        [
            chunk,
        ]
    )

    evaluator = RetrievalEvaluator(
        retriever
    )

    result = (
        evaluator.evaluate_case(
            make_case()
        )
    )

    assert (
        "rank=1:missing:access_class"
        in result.metadata_errors
    )

    assert (
        "rank=1:missing:content_type"
        in result.metadata_errors
    )