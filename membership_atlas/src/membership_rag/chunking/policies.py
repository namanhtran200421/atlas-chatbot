from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from membership_rag.config.models import LocalChunkingConfig

ChunkMode = Literal[
    "sectional",
    "section_atomic",
    "whole_document",
]


@dataclass(frozen=True, slots=True)
class ChunkPolicy:
    name: str
    mode: ChunkMode
    max_tokens: int
    overlap_units: int


POLICY_BY_CONTENT_TYPE: dict[str, str] = {
    "info_hub": "article",
    "anthropedia": "article",
    "page": "article",

    "research": "research",

    "faq": "faq",

    "cign_event": "event",
    "calendar_event_definition": "event",

    "membership_plan": "atomic",
}


MODE_BY_POLICY: dict[str, ChunkMode] = {
    "article": "sectional",
    "research": "sectional",
    "faq": "section_atomic",
    "event": "whole_document",
    "atomic": "whole_document",
}


def resolve_policy(
    content_type: str,
    config: LocalChunkingConfig,
) -> ChunkPolicy:
    policy_name = POLICY_BY_CONTENT_TYPE.get(
        content_type
    )

    if policy_name is None:
        raise ValueError(
            "No chunking policy configured for "
            f"content_type={content_type!r}"
        )

    policy_config = getattr(
        config.policies,
        policy_name,
    )

    return ChunkPolicy(
        name=policy_name,
        mode=MODE_BY_POLICY[policy_name],
        max_tokens=policy_config.max_tokens,
        overlap_units=policy_config.overlap_units,
    )
