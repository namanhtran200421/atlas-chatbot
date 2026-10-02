from __future__ import annotations

from typing import Any

from membership_rag.access_policy import is_public_catalogue_record
from membership_rag.chunking.models import NormalizedDocument


def record_to_document(
    record: dict[str, Any],
) -> NormalizedDocument | None:
    content_format = str(
        record.get("content_format") or ""
    ).strip()

    if content_format != "text/markdown":
        raise ValueError(
            "Unsupported content_format: "
            f"{content_format!r}"
        )

    content_markdown = str(
        record.get("content_markdown") or ""
    ).strip()

    if not content_markdown:
        return None

    return NormalizedDocument(
        document_id=_required_string(
            record,
            "document_id",
        ),
        source_id=_required_string(
            record,
            "source_id",
        ),
        source_type=_required_string(
            record,
            "source",
        ),
        content_type=_required_string(
            record,
            "content_type",
        ),
        access_class=_access_class(record),
        title=_required_string(
            record,
            "title",
        ),
        content_markdown=content_markdown,
        url=_optional_string(
            record.get("url")
        ),
        modified_at=_optional_string(
            record.get("modified_at")
        ),
    )


def _access_class(record: dict[str, Any]) -> str:
    """Apply the product catalogue's public visibility rule.

    Membership plan descriptions and prices are public catalogue information.
    This intentionally does not alter access for content unlocked by a plan.
    """

    source_type = _required_string(record, "source")
    content_type = _required_string(record, "content_type")
    if is_public_catalogue_record(source_type, content_type):
        return "public"
    return _required_string(record, "access_class")


def _required_string(
    record: dict[str, Any],
    field: str,
) -> str:
    value = record.get(field)

    if value is None:
        raise ValueError(
            f"Missing required field: {field}"
        )

    value = str(value).strip()

    if not value:
        raise ValueError(
            f"Required field is empty: {field}"
        )

    return value


def _optional_string(
    value: Any,
) -> str | None:
    if value is None:
        return None

    value = str(value).strip()

    return value or None
