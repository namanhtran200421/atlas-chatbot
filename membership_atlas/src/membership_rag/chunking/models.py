from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NormalizedDocument:
    document_id: str
    source_id: str
    source_type: str
    content_type: str
    access_class: str

    title: str
    content_markdown: str

    url: str | None = None
    modified_at: str | None = None


@dataclass(frozen=True, slots=True)
class MarkdownSection:
    section_id: str
    parent_id: str
    level: int

    heading: str | None
    heading_path: tuple[str, ...]

    body: str
    ordinal: int


@dataclass(frozen=True, slots=True)
class Chunk:
    document_id: str
    section_id: str
    parent_id: str

    chunk_id: str
    chunk_hash: str

    source_id: str
    source_type: str
    content_type: str
    access_class: str

    title: str
    heading: str | None

    text: str

    url: str | None = None
    modified_at: str | None = None