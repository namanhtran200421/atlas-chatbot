import json
from dataclasses import replace
from pathlib import Path

from membership_rag.chunking import (
    Chunk,
    write_bedrock_chunks,
)


def make_chunk() -> Chunk:
    return Chunk(
        document_id="anthropedia:1",
        section_id="sec_123",
        parent_id="anthropedia:1",
        chunk_id="chk_123",
        chunk_hash="abc123",
        source_id="1",
        source_type="anthropedia",
        content_type="anthropedia",
        access_class="member_restricted",
        title="Buddhism",
        heading="Buddhism",
        text=("# Buddhism\n\nBuddhism is a religious and philosophical tradition."),
        url="https://example.com/buddhism",
        modified_at="2026-09-09T00:00:00Z",
    )


def test_writer_creates_markdown_file(
    tmp_path: Path,
) -> None:
    chunk = make_chunk()

    write_bedrock_chunks(
        [chunk],
        tmp_path,
    )

    path = tmp_path / "chk_123.md"

    assert path.exists()

    assert path.read_text(encoding="utf-8").startswith("# Buddhism")


def test_writer_creates_metadata_sidecar(
    tmp_path: Path,
) -> None:
    chunk = make_chunk()

    write_bedrock_chunks(
        [chunk],
        tmp_path,
    )

    path = tmp_path / "chk_123.md.metadata.json"

    assert path.exists()

    metadata = json.loads(path.read_text(encoding="utf-8"))

    attributes = metadata["metadataAttributes"]

    assert attributes["chunk_id"] == "chk_123"

    assert attributes["access_class"] == "member_restricted"

    assert attributes["document_id"] == "anthropedia:1"


def test_writer_creates_one_pair_per_chunk(
    tmp_path: Path,
) -> None:
    first = make_chunk()

    second = replace(
        first,
        chunk_id="chk_456",
        chunk_hash="def456",
    )

    write_bedrock_chunks(
        [first, second],
        tmp_path,
    )

    files = list(tmp_path.iterdir())

    assert len(files) == 4
