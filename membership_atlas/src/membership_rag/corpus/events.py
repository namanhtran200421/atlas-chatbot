from __future__ import annotations

from collections import defaultdict
from typing import Any


def merge_event_occurrences(
    event_records: list[dict[str, Any]],
    occurrence_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Enrich MEC event definitions with their calendar occurrences.

    Records are joined by source_id. Calendar occurrence
    records are not returned independently.
    """

    occurrences_by_source_id: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for occurrence in occurrence_records:
        source_id = _source_id(
            occurrence
        )

        occurrences_by_source_id[
            source_id
        ].append(occurrence)

    merged: list[dict[str, Any]] = []

    for event in event_records:
        source_id = _source_id(
            event
        )

        occurrences = (
            occurrences_by_source_id.get(
                source_id,
                [],
            )
        )

        merged_event = dict(event)

        if occurrences:
            merged_event[
                "content_markdown"
            ] = _append_occurrences(
                event,
                occurrences,
            )

        merged.append(
            merged_event
        )

    return merged


def _append_occurrences(
    event: dict[str, Any],
    occurrences: list[dict[str, Any]],
) -> str:
    event_markdown = str(
        event.get("content_markdown") or ""
    ).strip()

    occurrence_blocks: list[str] = []

    seen: set[str] = set()

    for occurrence in occurrences:
        markdown = str(
            occurrence.get(
                "content_markdown"
            )
            or ""
        ).strip()

        if not markdown:
            continue

        markdown = _remove_first_heading(
            markdown
        )

        if not markdown:
            continue

        if markdown in seen:
            continue

        seen.add(markdown)
        occurrence_blocks.append(
            markdown
        )

    if not occurrence_blocks:
        return event_markdown

    occurrence_text = "\n\n".join(
        occurrence_blocks
    )

    if not any(block.lstrip().startswith("#") for block in occurrence_blocks):
        occurrence_text = f"## Calendar occurrences\n\n{occurrence_text}"

    return (
        f"{event_markdown}\n\n"
        f"{occurrence_text}"
    ).strip()


def _remove_first_heading(
    markdown: str,
) -> str:
    """
    Avoid nesting the occurrence document's own title
    inside the event document.
    """

    lines = markdown.splitlines()

    if (
        lines
        and lines[0].startswith("# ")
    ):
        lines = lines[1:]

    return "\n".join(
        lines
    ).strip()


def _source_id(
    record: dict[str, Any],
) -> str:
    value = record.get(
        "source_id"
    )

    if value is None:
        raise ValueError(
            "Record is missing source_id"
        )

    value = str(value).strip()

    if not value:
        raise ValueError(
            "Record has empty source_id"
        )

    return value
