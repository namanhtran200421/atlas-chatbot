"""Readable local ranking rules applied after Bedrock retrieval.

The AWS similarity score remains the main signal. Small boosts help obvious
content types and title matches without filtering other document types out.
That soft behaviour is important: a hard content-type filter caused a held-out
membership page to disappear completely during the original evaluation.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from membership_rag.bedrock.retrieval import RetrievedChunk

_WORD = re.compile(r"[a-z0-9]+")
_IGNORED_WORDS = {
    "a",
    "about",
    "and",
    "for",
    "how",
    "i",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "what",
    "which",
}

_CONTENT_TYPE_CUES: dict[str, tuple[str, ...]] = {
    "membership_plan": ("membership plan", "membership option", "annual price"),
    "faq": ("frequently asked", "faq"),
    "research": ("research", "study", "paper", "evidence"),
    "cign_event": ("workshop", "session", "event"),
    "calendar_event_definition": ("observance", "festival", "calendar"),
    "info_hub": ("guidance", "workplace", "how can our organisation"),
    "anthropedia": ("religion", "worldview", "faith", "belief"),
}


def predicted_content_types(query: str) -> frozenset[str]:
    lowered = query.casefold()
    return frozenset(
        content_type
        for content_type, cues in _CONTENT_TYPE_CUES.items()
        if any(cue in lowered for cue in cues)
    )


def _important_words(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.casefold()) if word not in _IGNORED_WORDS}


def _ranking_score(chunk: RetrievedChunk, query: str) -> float:
    base_score = float(chunk.score or 0.0)
    metadata = chunk.metadata
    content_type = metadata.get("content_type")
    title = str(metadata.get("title") or "")

    boost = 0.0
    if content_type in predicted_content_types(query):
        boost += 0.035

    query_words = _important_words(query)
    title_words = _important_words(title)
    if query_words and title_words:
        overlap = len(query_words & title_words) / len(title_words)
        boost += min(overlap * 0.08, 0.08)

    if title and title.casefold() in query.casefold():
        boost += 0.08
    return base_score + boost


def deduplicate_and_rank(
    chunks: Sequence[RetrievedChunk],
    query: str,
    *,
    limit: int,
) -> list[RetrievedChunk]:
    """Deduplicate documents and apply a stable, explainable local rerank."""

    unique: list[RetrievedChunk] = []
    seen_document_ids: set[str] = set()

    for chunk in chunks:
        document_id = chunk.document_id or chunk.metadata.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            continue
        if document_id in seen_document_ids:
            continue
        seen_document_ids.add(document_id)
        unique.append(chunk)

    ranked = sorted(
        enumerate(unique),
        key=lambda item: (-_ranking_score(item[1], query), item[0]),
    )
    return [chunk for _, chunk in ranked[:limit]]
