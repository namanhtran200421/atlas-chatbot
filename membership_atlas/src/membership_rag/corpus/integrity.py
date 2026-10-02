"""Integrity checks for a generated Bedrock chunk directory."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from membership_rag.guardrails import KNOWN_ACCESS_CLASSES

REQUIRED_METADATA = ("chunk_id", "document_id", "access_class", "content_type")


@dataclass(frozen=True, slots=True)
class CorpusIntegrityReport:
    passed: bool
    markdown_files: int
    metadata_files: int
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_corpus_directory(path: Path) -> CorpusIntegrityReport:
    errors: list[str] = []
    markdown_paths = sorted(path.glob("*.md"))
    metadata_paths = sorted(path.glob("*.md.metadata.json"))
    markdown_stems = {item.name.removesuffix(".md") for item in markdown_paths}
    metadata_stems = {
        item.name.removesuffix(".md.metadata.json") for item in metadata_paths
    }

    for stem in sorted(markdown_stems - metadata_stems):
        errors.append(f"missing metadata sidecar: {stem}.md")
    for stem in sorted(metadata_stems - markdown_stems):
        errors.append(f"missing markdown file: {stem}.md")

    seen_chunk_ids: set[str] = set()
    for metadata_path in metadata_paths:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid metadata {metadata_path.name}: {exc}")
            continue

        metadata = payload.get("metadataAttributes") or {}
        for field in REQUIRED_METADATA:
            if not metadata.get(field):
                errors.append(f"{metadata_path.name}: missing {field}")

        chunk_id = metadata.get("chunk_id")
        if isinstance(chunk_id, str):
            if chunk_id in seen_chunk_ids:
                errors.append(f"duplicate chunk_id: {chunk_id}")
            seen_chunk_ids.add(chunk_id)
            if metadata_path.name != f"{chunk_id}.md.metadata.json":
                errors.append(f"{metadata_path.name}: chunk_id does not match filename")

        access_class = metadata.get("access_class")
        if access_class and access_class not in KNOWN_ACCESS_CLASSES:
            errors.append(f"{metadata_path.name}: unknown access_class {access_class!r}")

    for markdown_path in markdown_paths:
        if not markdown_path.read_text(encoding="utf-8").strip():
            errors.append(f"empty chunk: {markdown_path.name}")

    if not markdown_paths:
        errors.append("no markdown chunks found")

    return CorpusIntegrityReport(
        passed=not errors,
        markdown_files=len(markdown_paths),
        metadata_files=len(metadata_paths),
        errors=tuple(errors),
    )
