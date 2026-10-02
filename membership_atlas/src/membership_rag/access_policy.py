"""Product-level access rules shared by ingestion and retrieval."""

from __future__ import annotations

PUBLIC_CATALOGUE_IDENTITIES = frozenset(
    {
        ("membership_plans", "membership_plan"),
    }
)


def is_public_catalogue_record(
    source_type: object,
    content_type: object,
) -> bool:
    """Return whether a record describes a publicly visible product plan."""

    return (source_type, content_type) in PUBLIC_CATALOGUE_IDENTITIES
