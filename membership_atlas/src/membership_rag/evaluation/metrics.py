"""Standard ranking metrics used by the retrieval evaluation suite."""

from collections.abc import Mapping, Sequence
from math import log2


def unique_preserve_order(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def hit_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")
    relevant = set(relevant_ids)
    return float(any(document_id in relevant for document_id in retrieved_ids[:k]))


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    found = set(retrieved_ids[:k]) & relevant
    return len(found) / len(relevant)


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
) -> float:
    relevant = set(relevant_ids)
    for rank, document_id in enumerate(retrieved_ids, start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevance_grades: Mapping[str, int],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")

    dcg = 0.0
    seen_documents: set[str] = set()
    for rank, document_id in enumerate(retrieved_ids[:k], start=1):
        relevance = 0
        if document_id not in seen_documents:
            relevance = relevance_grades.get(document_id, 0)
            seen_documents.add(document_id)
        dcg += ((2**relevance) - 1) / log2(rank + 1)

    ideal_grades = sorted(relevance_grades.values(), reverse=True)[:k]
    ideal_dcg = sum(
        ((2**relevance) - 1) / log2(rank + 1)
        for rank, relevance in enumerate(ideal_grades, start=1)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0
