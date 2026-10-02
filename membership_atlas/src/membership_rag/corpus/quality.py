from __future__ import annotations

import re

from membership_rag.chunking.models import (
    NormalizedDocument,
)
from membership_rag.privacy import is_personal_account_page

WORD_RE = re.compile(r"\b[\w'-]+\b")

SHORTCODE_RE = re.compile(
    r"\[[A-Za-z0-9_-]+(?:\s[^\]]*)?\]"
)


def should_index_document(
    document: NormalizedDocument,
) -> bool:
    """
    Decide whether a normalized document contains
    knowledge worth placing in the retrieval index.

    Filtering is deliberately source-aware. We do not
    apply a global minimum length because some short
    records, such as membership plans, are legitimate
    retrieval targets.
    """

    if document.content_type == "page" and is_personal_account_page(document.url):
        return False

    if document.content_type != "page":
        return True

    return _is_useful_page(
        document.content_markdown
    )


def _is_useful_page(
    markdown: str,
) -> bool:
    body = _remove_title(markdown)

    if not body.strip():
        return False

    # Placeholder pages such as:
    #
    # # Courses
    #
    # …
    cleaned = body.replace(
        "…",
        "",
    ).strip()

    if not cleaned:
        return False

    # Remove WordPress/plugin shortcodes before
    # estimating whether meaningful prose exists.
    without_shortcodes = SHORTCODE_RE.sub(
        "",
        cleaned,
    ).strip()

    words = WORD_RE.findall(
        without_shortcodes
    )

    # This threshold is only for generic WordPress
    # pages, not for all corpus types.
    return len(words) >= 20


def _remove_title(
    markdown: str,
) -> str:
    lines = markdown.splitlines()

    if (
        lines
        and lines[0].startswith("# ")
    ):
        lines = lines[1:]

    return "\n".join(lines).strip()
