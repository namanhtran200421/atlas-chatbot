"""Small, deterministic guardrails for the retrieval boundary.

These checks do not replace an identity provider or an application firewall.
They provide defence in depth: suspicious requests are stopped before they are
sent to the knowledge base, and access classes are validated again after AWS
returns results.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

KNOWN_ACCESS_CLASSES = frozenset(
    {
        "public",
        "member_restricted",
        "configuration",
    }
)
MAX_QUERY_CHARACTERS = 4_000


@dataclass(frozen=True, slots=True)
class GuardrailDecision:
    allowed: bool
    reason: str | None = None


_SECRET_REQUEST = re.compile(
    r"\b(?:secret|private|internal)\b.{0,80}"
    r"\b(?:password|credential|api[ -]?key|token|payroll)\b"
    r"|\b(?:password|credential|api[ -]?key|token)\b.{0,80}"
    r"\b(?:secret|private|internal)\b",
    re.IGNORECASE,
)
_PRIVATE_CONTACT_REQUEST = re.compile(
    r"\b(?:personal|private)\b.{0,80}"
    r"\b(?:phone|mobile|email|address|profile|contact details?)\b",
    re.IGNORECASE,
)
#: Product and brand names read as two capitalised words, so "Membership Atlas
#: is a membership programme" would otherwise look like an account enquiry.
_NON_PERSON_SUBJECT = (
    r"(?!(?:Cultural\s+Infusion|Membership\s+Atlas|Diversity\s+Atlas|"
    r"Cultural\s+Atlas|Free\s+Tour|Org\s+Admin|Org\s+Member|"
    r"Individual\s+Plan)\b)"
)
_NAMED_MEMBER_REQUEST = re.compile(
    r"\b(?:is|was|does|did|has)\s+"
    r"[A-Z][a-z'-]+\s+[A-Z][a-z'-]+\s+"
    r"(?:a\s+|an\s+)?(?:member\b|have\s+(?:an?\s+)?(?:account|membership)\b)",
    re.IGNORECASE,
)
# The pattern above only reads the inverted question form. "Confirm whether
# Sarah Nguyen has an active membership" states the name first and slipped
# through to retrieval, where the reader got the grounding fallback instead of
# the privacy redirect. Case matters here, so this runs on the original text.
_NAMED_MEMBER_STATEMENT = re.compile(
    r"\b" + _NON_PERSON_SUBJECT + r"[A-Z][a-z'’-]+\s+[A-Z][a-z'’-]+\b\s+"
    r"(?:is|was|has|have|had)\s+(?:still\s+)?(?:an?\s+)?"
    r"(?:active\s+|current\s+|valid\s+|paid\s+|lapsed\s+|expired\s+)?"
    r"(?:member\b|membership\b|account\b|subscription\b)"
)
# "What is the email address of member John Smith?" names no "personal" or
# "private" keyword, so the pattern above it never fired. Asking for a contact
# detail *of a named person* is the request being refused, and the name has to
# be matched case-sensitively so "the email address for support" stays fine.
_NAMED_CONTACT_REQUEST = re.compile(
    r"(?i:\b(?:e-?mail(?:\s+address)?|phone(?:\s+number)?|mobile(?:\s+number)?|"
    r"contact\s+details?|contact\s+number|postal\s+address|home\s+address)\b"
    r"\s*(?:of|for|belonging\s+to)\s+(?:the\s+|a\s+|an\s+)?"
    r"(?:member|user|customer|subscriber|employee|staff\s+member|person)?\s*)"
    + _NON_PERSON_SUBJECT + r"[A-Z][a-z'’-]+"
    r"|\b" + _NON_PERSON_SUBJECT + r"[A-Z][a-z'’-]+(?:\s+[A-Z][a-z'’-]+)?['’]s\s+"
    r"(?i:(?:e-?mail|phone|mobile|contact\s+details?|address|account))"
)
_PERMISSION_BYPASS = re.compile(
    r"\b(?:ignore|bypass|override|disable)\b.{0,80}"
    r"\b(?:permissions?|access|authori[sz]ation|security)\b"
    r"|\b(?:not signed in|without (?:a )?membership|without access)\b"
    r".{0,100}\b(?:anyway|full|internal|private|restricted|retrieve|show)\b",
    re.IGNORECASE,
)
_MODEL_INTERNALS_REQUEST = re.compile(
    r"\b(?:system|developer|hidden|internal)\s+"
    r"(?:prompts?|messages?|instructions?|rules?)\b"
    r"|\b(?:chain[- ]of[- ]thought|reasoning trace|model configuration)\b"
    r"|\b(?:raw\s+)?(?:source\s+)?metadata\b"
    r"|\b(?:s3[ _-]?uris?|chunk[ _-]?ids?|document[ _-]?ids?|"
    r"access[ _-]?class(?:es)?)\b",
    re.IGNORECASE,
)
_PROMPT_OVERRIDE = re.compile(
    r"\b(?:ignore|disregard|forget|override)\b.{0,80}"
    r"\b(?:previous|prior|above|system|developer|hidden|every)\b.{0,40}"
    r"\b(?:instructions?|prompts?|rules?)\b",
    re.IGNORECASE,
)
_DIRECT_PROMPT_OVERRIDE = re.compile(
    r"\b(?:ignore|disregard|forget|override)\s+"
    r"(?:(?:all|the|any)\s+)?(?:instructions?|prompts?|rules?)\b",
    re.IGNORECASE,
)
_ROLE_INJECTION = re.compile(
    r"(?:<|\[|\{){0,2}\s*(?:role\s*[:=]\s*)?"
    r"(?:system|developer)\s*(?:>|\]|\})"
    r"|[\"']?(?:role|message)[\"']?\s*[:=]\s*"
    r"[\"']?(?:system|developer)\b"
    r"|<\|\s*(?:system|developer)\s*\|>"
    r"|\[(?:/)?inst\]"
    r"|\b(?:developer mode|unrestricted assistant|do anything now|jailbreak)\b"
    r"|\b(?:print|repeat|return|show)\b.{0,60}"
    r"\b(?:everything|text|messages?)\b.{0,40}"
    r"\b(?:before|above|prior to)\b"
    r"|\bbase64\b.{0,60}\b(?:decode|follow|execute|instructions?)\b",
    re.IGNORECASE,
)
_RESPONSE_HIJACK = re.compile(
    r"\b(?:when|while|before|after)\s+(?:you\s+)?"
    r"(?:answer(?:ing)?|respond(?:ing)?|reply(?:ing)?)\b"
    r".{0,120}\b(?:say|write|output|print|return)\b"
    r"|\b(?:for|in)\s+(?:your\s+)?(?:next|future)\s+(?:answer|response|reply)\b"
    r".{0,120}\b(?:say|write|output|print|return)\b",
    re.IGNORECASE,
)
_COMPACT_PROMPT_OVERRIDE = re.compile(
    r"(?:ignore|disregard|forget|override)(?:all|every|the)?"
    r"(?:previous|prior|above|system|developer|hidden)"
    r".{0,40}(?:instruction|prompt|rule)"
)
_COMPACT_INTERNAL_MARKERS = (
    "systemprompt",
    "developerprompt",
    "hiddenprompt",
    "internalprompt",
    "systeminstruction",
    "developerinstruction",
    "hiddeninstruction",
    "internalinstruction",
    "chainofthought",
    "reasoningtrace",
    "modelconfiguration",
    "rawmetadata",
    "s3uri",
    "chunkid",
    "documentid",
    "accessclass",
)
_LEET_TRANSLATION = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
    }
)
_CONFUSABLE_TRANSLATION = str.maketrans(
    {
        # Common Cyrillic and Greek homoglyphs used to conceal English prompt
        # injection phrases. This view is security-only; it never changes the
        # multilingual query sent to retrieval.
        "а": "a",
        "е": "e",
        "і": "i",
        "ј": "j",
        "о": "o",
        "р": "p",
        "с": "c",
        "у": "y",
        "х": "x",
        "α": "a",
        "ε": "e",
        "ι": "i",
        "κ": "k",
        "ο": "o",
        "ρ": "p",
        "τ": "t",
        "υ": "y",
        "χ": "x",
    }
)
_REPEATED_CHARACTER = re.compile(r"(.)\1{63,}", re.DOTALL)


def _inspection_views(value: str) -> tuple[str, str]:
    """Return normalized and compact views used only for security inspection."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    # Ignore invisible formatting characters while inspecting so zero-width
    # separators cannot conceal prompt-injection phrases. The original query is
    # retained for retrieval and multilingual text remains otherwise unchanged.
    visible = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Cf"
    )
    inspection_text = visible.translate(_CONFUSABLE_TRANSLATION)
    compact = re.sub(
        r"[^a-z0-9]+",
        "",
        inspection_text.translate(_LEET_TRANSLATION),
    )
    return inspection_text, compact


def _has_unsafe_control_character(value: str) -> bool:
    return any(
        unicodedata.category(character) == "Cc"
        and character not in {"\t", "\n", "\r"}
        for character in value
    )


def _looks_like_spam(value: str) -> bool:
    if _REPEATED_CHARACTER.search(value):
        return True

    words = re.findall(r"\b\w+\b", value.casefold(), flags=re.UNICODE)
    if len(words) < 12:
        return False
    most_common_count = Counter(words).most_common(1)[0][1]
    return most_common_count >= 12 and most_common_count / len(words) >= 0.5


def inspect_query(query: str) -> GuardrailDecision:
    """Return a deterministic decision before retrieval is attempted."""

    cleaned = query.strip()
    if not cleaned:
        return GuardrailDecision(False, "empty_query")
    if len(cleaned) > MAX_QUERY_CHARACTERS:
        return GuardrailDecision(False, "query_too_long")
    if _has_unsafe_control_character(cleaned):
        return GuardrailDecision(False, "unsafe_control_character")
    normalized, compact = _inspection_views(cleaned)
    if _looks_like_spam(normalized):
        return GuardrailDecision(False, "spam_query")
    if _PERMISSION_BYPASS.search(normalized):
        return GuardrailDecision(False, "permission_bypass_request")
    if _SECRET_REQUEST.search(normalized):
        return GuardrailDecision(False, "secret_request")
    if _PRIVATE_CONTACT_REQUEST.search(normalized):
        return GuardrailDecision(False, "private_contact_request")
    if (
        _NAMED_MEMBER_REQUEST.search(cleaned)
        or _NAMED_MEMBER_STATEMENT.search(cleaned)
        or _NAMED_CONTACT_REQUEST.search(cleaned)
    ):
        return GuardrailDecision(False, "private_membership_request")
    if (
        _MODEL_INTERNALS_REQUEST.search(normalized)
        or _PROMPT_OVERRIDE.search(normalized)
        or _DIRECT_PROMPT_OVERRIDE.search(normalized)
        or _ROLE_INJECTION.search(normalized)
        or _RESPONSE_HIJACK.search(normalized)
        or _COMPACT_PROMPT_OVERRIDE.search(compact)
        or any(marker in compact for marker in _COMPACT_INTERNAL_MARKERS)
    ):
        return GuardrailDecision(False, "model_internals_request")
    return GuardrailDecision(True)


def validate_access_classes(access_classes: Sequence[str]) -> tuple[str, ...]:
    """Validate and deduplicate access classes supplied by the backend."""

    values = tuple(dict.fromkeys(access_classes))
    if not values:
        raise ValueError("allowed_access_classes must not be empty")

    unknown = sorted(set(values) - KNOWN_ACCESS_CLASSES)
    if unknown:
        raise ValueError(f"unknown access classes: {', '.join(unknown)}")
    return values


class AccessControlError(RuntimeError):
    """Raised when AWS returns a result that violates the caller's ACL."""


class ResultMetadataError(RuntimeError):
    """Raised when a retrieved result cannot be safely identified or routed."""


def enforce_required_metadata(metadata: object) -> None:
    if not isinstance(metadata, dict):
        raise ResultMetadataError("retrieved result metadata is not an object")
    for field in ("document_id", "access_class", "content_type"):
        if not metadata.get(field):
            raise ResultMetadataError(f"retrieved result is missing {field}")


def enforce_result_access(
    result_access_class: object,
    allowed_access_classes: Sequence[str],
) -> None:
    """Fail closed for missing or unauthorised result metadata."""

    if not isinstance(result_access_class, str) or not result_access_class:
        raise AccessControlError("retrieved result is missing access_class")
    if result_access_class not in set(allowed_access_classes):
        raise AccessControlError(
            f"retrieved result has unauthorised access_class={result_access_class!r}"
        )
