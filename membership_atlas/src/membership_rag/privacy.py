"""Keep personalised account pages out of the knowledge path."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

_ACCOUNT_PATH = re.compile(
    r"^/(?:profile|login|membership-account|upgrade-account-form)(?:/|$)",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PERSONAL_MEMBERSHIP_CLAIM = re.compile(
    r"\b(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+"
    r"(?:is|was|has|had)\s+(?:a\s+|an\s+)?"
    r"(?:member\b|membership\b|(?:membership\s+)?account\b)"
)
# Brands and plan names are not people. Without this, describing the product
# ("Cultural Infusion is a membership organisation") was suppressed as though
# it disclosed someone's account.
_NON_PERSON_SUBJECTS = frozenset(
    {
        "cultural infusion",
        "membership atlas",
        "diversity atlas",
        "cultural atlas",
        "atlas",
        "oriana",
        "free tour",
        "individual",
        "enterprise",
    }
)


def is_personal_account_page(url: object) -> bool:
    if not isinstance(url, str):
        return False
    return bool(_ACCOUNT_PATH.match(urlsplit(url).path))


def contains_email_address(text: str) -> bool:
    return bool(_EMAIL.search(text))


def contains_personal_membership_claim(text: str) -> bool:
    """Return whether the text asserts that a named person holds a membership."""

    for match in _PERSONAL_MEMBERSHIP_CLAIM.finditer(text):
        name = " ".join(match.group("name").split()).casefold()
        if any(
            name == subject or name.endswith(f" {subject}")
            for subject in _NON_PERSON_SUBJECTS
        ):
            continue
        return True
    return False
