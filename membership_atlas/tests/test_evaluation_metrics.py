import pytest

from membership_rag.evaluation.metrics import (
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_hit_at_k_success() -> None:
    retrieved = [
        "doc_a",
        "doc_b",
        "doc_c",
    ]

    relevant = [
        "doc_b",
    ]

    assert hit_at_k(
        retrieved,
        relevant,
        3,
    ) == 1.0


def test_hit_at_k_failure() -> None:
    retrieved = [
        "doc_a",
        "doc_b",
    ]

    relevant = [
        "doc_c",
    ]

    assert hit_at_k(
        retrieved,
        relevant,
        2,
    ) == 0.0


def test_recall_at_k() -> None:
    retrieved = [
        "doc_a",
        "doc_b",
        "doc_c",
    ]

    relevant = [
        "doc_b",
        "doc_c",
        "doc_d",
    ]

    assert recall_at_k(
        retrieved,
        relevant,
        3,
    ) == pytest.approx(
        2 / 3
    )


def test_reciprocal_rank() -> None:
    retrieved = [
        "doc_a",
        "doc_b",
        "doc_c",
    ]

    relevant = [
        "doc_c",
    ]

    assert reciprocal_rank(
        retrieved,
        relevant,
    ) == pytest.approx(
        1 / 3
    )


def test_duplicate_documents_do_not_change_rank() -> None:
    retrieved = [
        "doc_a",
        "doc_a",
        "doc_b",
    ]

    relevant = [
        "doc_b",
    ]

    assert reciprocal_rank(
        retrieved,
        relevant,
    ) == pytest.approx(
        1/3
    )


def test_ndcg_perfect_ranking() -> None:
    retrieved = [
        "doc_a",
        "doc_b",
        "doc_c",
    ]

    grades = {
        "doc_a": 3,
        "doc_b": 2,
        "doc_c": 1,
    }

    assert ndcg_at_k(
        retrieved,
        grades,
        3,
    ) == pytest.approx(
        1.0
    )


def test_ndcg_penalizes_bad_order() -> None:
    grades = {
        "doc_a": 3,
        "doc_b": 2,
        "doc_c": 1,
    }

    perfect = ndcg_at_k(
        [
            "doc_a",
            "doc_b",
            "doc_c",
        ],
        grades,
        3,
    )

    reversed_ranking = ndcg_at_k(
        [
            "doc_c",
            "doc_b",
            "doc_a",
        ],
        grades,
        3,
    )

    assert (
        reversed_ranking
        < perfect
    )


def test_ndcg_duplicate_consumes_rank_position() -> None:
    duplicate_score = ndcg_at_k(
        ["wrong", "wrong", "relevant"],
        {"relevant": 3},
        3,
    )
    perfect_score = ndcg_at_k(["relevant"], {"relevant": 3}, 3)
    assert duplicate_score < perfect_score
