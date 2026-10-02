from __future__ import annotations

import re

from membership_rag.chunking.ids import stable_id
from membership_rag.chunking.models import (
    MarkdownSection,
    NormalizedDocument,
)

HEADING_RE = re.compile(
    r"^(#{1,6})\s+(.+?)\s*$"
)


def parse_markdown_sections(
    document: NormalizedDocument,
) -> list[MarkdownSection]:
    """
    Parse Markdown into hierarchical sections.

    Example:

    # Document
    ## Section
    ### Subsection

    Each section receives a deterministic section ID
    and points to its parent section.
    """

    raw_sections: list[
        tuple[int, str | None, str, int]
    ] = []

    current_level = 0
    current_heading: str | None = None
    body_lines: list[str] = []

    ordinal = 0

    def flush() -> None:
        nonlocal ordinal

        body = "\n".join(body_lines).strip()

        if current_heading is None and not body:
            return

        raw_sections.append(
            (
                current_level,
                current_heading,
                body,
                ordinal,
            )
        )

        ordinal += 1

    for line in document.content_markdown.splitlines():
        match = HEADING_RE.match(line)

        if match is None:
            body_lines.append(line)
            continue

        # Save previous section before starting
        # the new heading.
        flush()
        body_lines.clear()

        current_level = len(match.group(1))
        current_heading = match.group(2).strip()

    # Flush the final section.
    flush()

    result: list[MarkdownSection] = []

    # Each stack element:
    # (heading_level, section_id, heading_text)
    stack: list[tuple[int, str, str]] = []

    for (
        level,
        heading,
        body,
        section_ordinal,
    ) in raw_sections:

        while stack and stack[-1][0] >= level:
            stack.pop()

        parent_id = (
            stack[-1][1]
            if stack
            else document.document_id
        )

        heading_key = heading or "__root__"

        section_id = stable_id(
            "sec",
            document.document_id,
            parent_id,
            heading_key,
            str(section_ordinal),
        )

        heading_path = tuple(
            item[2]
            for item in stack
        )

        if heading is not None:
            heading_path += (heading,)

        result.append(
            MarkdownSection(
                section_id=section_id,
                parent_id=parent_id,
                level=level,
                heading=heading,
                heading_path=heading_path,
                body=body,
                ordinal=section_ordinal,
            )
        )

        if heading is not None:
            stack.append(
                (
                    level,
                    section_id,
                    heading,
                )
            )

    return result