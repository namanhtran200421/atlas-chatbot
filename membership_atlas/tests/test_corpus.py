import json
from pathlib import Path

import pytest

from membership_rag.corpus import (
    load_documents,
    record_to_document,
)


def test_record_to_document() -> None:
    record = {
        "schema_version": "1.1",
        "document_id": "anthropedia:29676",
        "source": "anthropedia",
        "source_id": 29676,
        "content_type": "anthropedia",
        "access_class": "member_restricted",
        "title": "African Traditional Religions",
        "content_format": "text/markdown",
        "content_markdown": (
            "# African Traditional Religions\n\nSome useful information."
        ),
        "modified_at": "2024-07-30T15:34:30+10:00",
        "url": "https://example.com/article",
    }

    document = record_to_document(record)

    assert document is not None
    assert document.document_id == "anthropedia:29676"
    assert document.source_type == "anthropedia"
    assert document.content_type == "anthropedia"
    assert document.access_class == "member_restricted"


def test_empty_markdown_is_skipped() -> None:
    record = {
        "document_id": "anthropedia:1",
        "source": "anthropedia",
        "source_id": 1,
        "content_type": "anthropedia",
        "access_class": "member_restricted",
        "title": "Empty",
        "content_format": "text/markdown",
        "content_markdown": "",
    }

    assert record_to_document(record) is None


def test_non_markdown_content_is_rejected() -> None:
    record = {
        "document_id": "anthropedia:1",
        "source": "anthropedia",
        "source_id": 1,
        "content_type": "anthropedia",
        "access_class": "member_restricted",
        "title": "Test",
        "content_format": "text/html",
        "content_markdown": "# Test",
    }

    with pytest.raises(ValueError):
        record_to_document(record)


def test_loader_reads_jsonl(
    tmp_path: Path,
) -> None:
    record = {
        "document_id": "anthropedia:1",
        "source": "anthropedia",
        "source_id": 1,
        "content_type": "anthropedia",
        "access_class": "member_restricted",
        "title": "Test",
        "content_format": "text/markdown",
        "content_markdown": "# Test\n\nContent.",
    }

    path = tmp_path / "anthropedia.jsonl"

    path.write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].document_id == "anthropedia:1"
