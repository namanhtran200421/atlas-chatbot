from __future__ import annotations

import difflib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.config import Config

from membership_rag.access_policy import is_public_catalogue_record
from membership_rag.conversation import ConversationTurn, contextual_query
from membership_rag.guardrails import (
    enforce_required_metadata,
    enforce_result_access,
    inspect_query,
    validate_access_classes,
)
from membership_rag.privacy import is_personal_account_page

_MEC_EVENTS = re.compile(r"\bmec\s+events?\b", re.IGNORECASE)
_GENERAL_MEC_EVENTS = re.compile(
    r"^(?:(?:what|which)\s+(?:is|are)\s+|tell me about\s+)?"
    r"(?:the\s+)?mec\s+events?\s*\??$",
    re.IGNORECASE,
)
# Naming a research document is itself the signal. "article" is deliberately
# absent: info hub and worldview entries are articles too, and filtering them
# out would hide the right answer.
_RESEARCH_DOCUMENT_REQUEST = re.compile(
    r"\bresearch\b"
    r"|\bpapers?\b"
    r"|\bpublications?\b"
    r"|\b(?:published\s+)?studies\b"
    r"|\b(?:scholarly|academic)\s+articles?\b",
    re.IGNORECASE,
)
# "I'm doing research on X" asks about X, not about the research collection.
_RESEARCH_AS_ACTIVITY = re.compile(
    r"\b(?:doing|do|did|conducting|conduct)\s+research\b"
    r"|\b(?:my|our|their)\s+research\b",
    re.IGNORECASE,
)
# "reserch papers pls" should still reach the research collection. Spelling is
# corrected only to classify the request; the original text is sent to Bedrock.
_RESEARCH_VOCABULARY = (
    "research",
    "publications",
    "publication",
    "studies",
    "scholarly",
    "academic",
    "articles",
    "article",
    "papers",
    "paper",
)
# Ordinary words that are close to the vocabulary but mean something else.
_NEVER_CORRECTED = frozenset(
    {
        "search",
        "searches",
        "searching",
        "searched",
        "studio",
        "studios",
        "student",
        "students",
        "scholar",
        "scholars",
        "academy",
        "academies",
        "resource",
        "resources",
        "service",
        "services",
    }
)
_ALPHABETIC_WORD = re.compile(r"[A-Za-z]+")


def _correct_spelling(query: str) -> str:
    """Repair near-miss spellings of the research vocabulary."""

    def correct(match: re.Match[str]) -> str:
        word = match.group(0)
        lowered = word.casefold()
        if len(lowered) < 5 or lowered in _NEVER_CORRECTED:
            return word
        closest = difflib.get_close_matches(
            lowered, _RESEARCH_VOCABULARY, n=1, cutoff=0.84
        )
        return closest[0] if closest else word

    return _ALPHABETIC_WORD.sub(correct, query)


def is_research_list_request(query: str) -> bool:
    """Recognise requests for examples of papers in the indexed research collection."""

    if _RESEARCH_AS_ACTIVITY.search(query):
        return False
    return bool(_RESEARCH_DOCUMENT_REQUEST.search(_correct_spelling(query)))


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    text: str
    score: float | None
    metadata: dict[str, Any]
    document_id: str | None
    source_uri: str | None


class BedrockRetriever:
    """Production retrieval boundary for an Amazon Bedrock knowledge base.

    The backend owns the user's access classes and must pass them on every
    request. Query guardrails run before AWS, while result ACL checks run after
    AWS. The second check deliberately fails closed if metadata is malformed.
    """

    def __init__(
        self,
        knowledge_base_id: str,
        region_name: str,
        client: Any | None = None,
        *,
        use_guardrails: bool = True,
        strict_result_access: bool = True,
        local_reranking: bool = True,
        include_public_catalogue: bool = False,
        candidate_pool_size: int = 20,
    ) -> None:
        if not knowledge_base_id.strip():
            raise ValueError("knowledge_base_id must not be empty")
        self.knowledge_base_id = knowledge_base_id
        self.use_guardrails = use_guardrails
        self.strict_result_access = strict_result_access
        self.local_reranking = local_reranking
        self.include_public_catalogue = include_public_catalogue
        if not 1 <= candidate_pool_size <= 100:
            raise ValueError("candidate_pool_size must be between 1 and 100")
        self.candidate_pool_size = candidate_pool_size

        self.client = client or boto3.client(
            "bedrock-agent-runtime",
            region_name=region_name,
            config=Config(
                connect_timeout=5,
                read_timeout=20,
                retries={
                    "max_attempts": 3,
                    "mode": "standard",
                },
                user_agent_extra="atlas-membership-rag/0.1",
            ),
        )

    def retrieve(
        self,
        query: str,
        *,
        allowed_access_classes: Sequence[str],
        number_of_results: int = 5,
        history: Sequence[ConversationTurn] = (),
    ) -> list[RetrievedChunk]:
        query = query.strip()

        if not query:
            raise ValueError("query must not be empty")

        allowed_access_classes = validate_access_classes(allowed_access_classes)

        if not 1 <= number_of_results <= 100:
            raise ValueError("number_of_results must be between 1 and 100")

        if self.use_guardrails and not inspect_query(query).allowed:
            return []

        # "How much is it?" cannot be searched on its own, so a follow-up is
        # searched together with the question it refers to. Guardrails have
        # already passed on the message as the person typed it, and routing
        # below reads this text so "any others?" stays on the same collection.
        search_text = contextual_query(query, history)

        access_filter = _build_access_filter(
            allowed_access_classes,
            include_public_catalogue=self.include_public_catalogue,
        )
        retrieval_query = search_text
        if _MEC_EVENTS.search(search_text):
            # The public catalogue exception concerns membership plans only.
            # Excluding it keeps the MEC filter within Bedrock's nesting limit.
            event_access_filter = _build_access_filter(allowed_access_classes)
            access_filter = {
                "andAll": [
                    event_access_filter,
                    {"equals": {"key": "source_type", "value": "mec_event_definitions"}},
                ]
            }
            # Only the bare question is broadened; what the person actually
            # typed decides that, not the question it follows on from.
            if _GENERAL_MEC_EVENTS.fullmatch(query.strip()):
                retrieval_query = (
                    "Cultural Infusion membership calendar events, cultural "
                    "observances and event suggestions"
                )
        elif is_research_list_request(search_text):
            # As for MEC events, the public catalogue exception concerns
            # membership plans only. Excluding it keeps this filter inside
            # Bedrock's two-level nesting limit.
            research_access_filter = _build_access_filter(allowed_access_classes)
            access_filter = {
                "andAll": [
                    research_access_filter,
                    {"equals": {"key": "content_type", "value": "research"}},
                ]
            }
            # The filter selects the research collection. Preserve the full
            # question so specific paper descriptions remain searchable.

        # Retrieve more candidates than the final response needs. Bedrock can
        # return several chunks from one document; fetching only the requested
        # final count can leave the local deduplicator with a single source.
        candidate_count = (
            max(number_of_results, self.candidate_pool_size)
            if self.local_reranking
            else number_of_results
        )

        response = self.client.retrieve(
            knowledgeBaseId=self.knowledge_base_id,
            retrievalQuery={
                "text": retrieval_query,
            },
            retrievalConfiguration={
                "managedSearchConfiguration": {
                    "numberOfResults": candidate_count,
                    "filter": access_filter,
                }
            },
        )

        chunks = [
            _parse_result(result)
            for result in response.get("retrievalResults", [])
        ]

        if self.strict_result_access:
            for chunk in chunks:
                enforce_required_metadata(chunk.metadata)
                access_class = chunk.metadata.get("access_class")
                if (
                    self.include_public_catalogue
                    and "public" in allowed_access_classes
                    and is_public_catalogue_record(
                        chunk.metadata.get("source_type"),
                        chunk.metadata.get("content_type"),
                    )
                ):
                    access_class = "public"
                enforce_result_access(
                    access_class,
                    allowed_access_classes,
                )

        # Existing knowledge bases can still hold pages captured while a
        # crawler was signed in. Never pass those results to generation.
        chunks = [
            chunk for chunk in chunks
            if not is_personal_account_page(chunk.metadata.get("url"))
        ]

        if not self.local_reranking:
            return chunks[:number_of_results]

        # Local import keeps the simple RetrievedChunk model in this module
        # without creating a module-import cycle.
        from membership_rag.bedrock.ranking import deduplicate_and_rank

        return deduplicate_and_rank(chunks, search_text, limit=number_of_results)


def _build_access_filter(
    allowed_access_classes: Sequence[str],
    *,
    include_public_catalogue: bool = False,
) -> dict[str, Any]:
    values = list(dict.fromkeys(allowed_access_classes))

    if len(values) == 1:
        access_filter: dict[str, Any] = {
            "equals": {
                "key": "access_class",
                "value": values[0],
            }
        }
    else:
        access_filter = {
            "in": {
                "key": "access_class",
                "value": values,
            }
        }

    if not include_public_catalogue or "public" not in values:
        return access_filter

    return {
        "orAll": [
            access_filter,
            {
                "andAll": [
                    {
                        "equals": {
                            "key": "source_type",
                            "value": "membership_plans",
                        }
                    },
                    {
                        "equals": {
                            "key": "content_type",
                            "value": "membership_plan",
                        }
                    },
                ]
            },
        ]
    }


def _parse_result(result: dict[str, Any]) -> RetrievedChunk:
    content = result.get("content", {})
    metadata = result.get("metadata", {})
    location = result.get("location", {})
    s3_location = location.get("s3Location", {})

    return RetrievedChunk(
        text=content.get("text", ""),
        score=result.get("score"),
        metadata=metadata,
        document_id=metadata.get("document_id"),
        source_uri=s3_location.get("uri"),
    )
