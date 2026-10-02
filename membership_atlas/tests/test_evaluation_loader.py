import json
from pathlib import Path

import pytest

from membership_rag.evaluation.loader import (
    load_evaluation_cases,
)


def write_cases(
    path: Path,
    cases: list[dict],
) -> None:
    path.write_text(
        "\n".join(
            json.dumps(case)
            for case in cases
        ),
        encoding="utf-8",
    )


def valid_case() -> dict:
    return {
        "query_id": "q001",
        "query": "What is Buddhism?",
        "category": "anthropedia",
        "expected_document_ids": [
            "anthropedia:28835",
        ],
        "expected_source_types": [
            "anthropedia",
        ],
        "expected_content_types": [
            "anthropedia",
        ],
        "allowed_access_classes": [
            "member_restricted",
        ],
        "answerable": True,
        "relevance_grades": {
            "anthropedia:28835": 3,
        },
        "notes": "Test query",
    }


def test_loader_reads_case(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "cases.jsonl"
    )

    write_cases(
        path,
        [
            valid_case(),
        ],
    )

    cases = (
        load_evaluation_cases(
            path
        )
    )

    assert len(cases) == 1

    assert (
        cases[0].query_id
        == "q001"
    )

    assert (
        cases[0].expected_document_ids
        == (
            "anthropedia:28835",
        )
    )


def test_duplicate_query_id_is_rejected(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "cases.jsonl"
    )

    first = valid_case()

    second = valid_case()

    write_cases(
        path,
        [
            first,
            second,
        ],
    )

    with pytest.raises(
        ValueError,
        match="duplicate query_id",
    ):
        load_evaluation_cases(
            path
        )


def test_answerable_query_requires_document(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "cases.jsonl"
    )

    case = valid_case()

    case[
        "expected_document_ids"
    ] = []

    case[
        "relevance_grades"
    ] = {}

    write_cases(
        path,
        [
            case,
        ],
    )

    with pytest.raises(
        ValueError,
    ):
        load_evaluation_cases(
            path
        )


def test_unanswerable_query_cannot_have_document(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "cases.jsonl"
    )

    case = valid_case()

    case[
        "answerable"
    ] = False

    write_cases(
        path,
        [
            case,
        ],
    )

    with pytest.raises(
        ValueError,
    ):
        load_evaluation_cases(
            path
        )


def test_relevance_grades_must_match_documents(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "cases.jsonl"
    )

    case = valid_case()

    case[
        "relevance_grades"
    ] = {
        "wrong_document": 3,
    }

    write_cases(
        path,
        [
            case,
        ],
    )

    with pytest.raises(
        ValueError,
    ):
        load_evaluation_cases(
            path
        )