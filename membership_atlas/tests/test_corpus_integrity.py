import json
from pathlib import Path

from membership_rag.corpus.integrity import check_corpus_directory


def write_pair(path: Path, chunk_id: str = "chunk-1") -> None:
    (path / f"{chunk_id}.md").write_text("# Useful content\n", encoding="utf-8")
    metadata = {
        "metadataAttributes": {
            "chunk_id": chunk_id,
            "document_id": "doc-1",
            "access_class": "public",
            "content_type": "page",
        }
    }
    (path / f"{chunk_id}.md.metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def test_valid_corpus_passes(tmp_path: Path) -> None:
    write_pair(tmp_path)
    report = check_corpus_directory(tmp_path)
    assert report.passed
    assert report.markdown_files == 1


def test_missing_metadata_fails(tmp_path: Path) -> None:
    (tmp_path / "chunk-1.md").write_text("Content", encoding="utf-8")
    report = check_corpus_directory(tmp_path)
    assert not report.passed
    assert "missing metadata sidecar" in report.errors[0]
