"""Resolve a reference back to something the assistant has already listed.

A reader shown five papers replies "let's go with the fourth one", "the LGBTQ
one", "number 3" or "what does that paper say". None of those name a
document. Searching the words themselves finds whatever happens to match:
"the fourth one" returned a paper on the Fourth Industrial Revolution.

The titles are already present in the answer that listed them, so the
reference is resolved against that list first and the item it picks out
becomes the search. One mechanism covers the ordinal, positional and
descriptive ways of pointing at a list, rather than a pattern per phrasing.

Everything here reads the assistant's own earlier turn, which arrives from
the browser and may have been altered. It only ever steers retrieval inside
the access-controlled corpus; it never becomes grounding or authorization.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from membership_rag.scope import content_tokens

#: Words for positions a reader can name. "Last" is resolved against the end.
_ORDINAL_WORDS = {
    "first": 1,
    "1st": 1,
    "second": 2,
    "2nd": 2,
    "third": 3,
    "3rd": 3,
    "fourth": 4,
    "4th": 4,
    "fifth": 5,
    "5th": 5,
    "sixth": 6,
    "6th": 6,
    "seventh": 7,
    "7th": 7,
    "eighth": 8,
    "8th": 8,
    "ninth": 9,
    "9th": 9,
    "tenth": 10,
    "10th": 10,
}
_LAST_WORDS = frozenset({"last", "final", "bottom"})

# "the fourth one", "number 3", "option 2", "#4", "the last one".
_POSITION = re.compile(
    r"\b(?:the\s+|that\s+|this\s+)?"
    r"(?P<word>first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|sixth|6th|"
    r"seventh|7th|eighth|8th|ninth|9th|tenth|10th|last|final|bottom)\b"
    # "#2" has no word boundary before the hash, so it needs its own branch.
    r"|(?:\b(?:number|option|item|no\.?)\s*|#\s*)(?P<digit>\d{1,2})\b",
    re.IGNORECASE,
)

# A numbered or bulleted line, with the title kept and any trailing gloss
# ("- Explores the reasons ...") dropped.
_LIST_LINE = re.compile(
    r"^\s*(?:\d{1,2}\s*[.)\]]|[-*•–])\s*(?P<item>.+?)\s*$",
    re.MULTILINE,
)
# Quote marks must be paired by kind. Accepting a lone apostrophe as an
# opening quote made "If you're interested in ..." yield the fragment
# "re interested in workplace challenges," as though it were a title.
_QUOTED = re.compile(
    r"\"(?P<double>[^\"\n]{6,200})\""
    r"|“(?P<curly>[^”\n]{6,200})”"
    r"|‘(?P<single>[^’\n]{6,200})’"
)
_TRAILING_GLOSS = re.compile(r"\s+[-–—:]\s+.*$")
_LEADING_LABEL = re.compile(r"^(?:title|paper|article|study)\s*:\s*", re.IGNORECASE)
_MARKDOWN_EMPHASIS = re.compile(r"\*{1,3}|_{2,}|`")

#: Words that do the pointing rather than the describing. Left in, "let's read
#: the paper about X" would score every item that happens to say "paper".
_POINTING_WORDS = frozenset(
    {
        "let", "lets", "go", "pick", "choose", "take", "try", "read", "open",
        "tell", "explain", "describe", "show", "more", "about", "please",
        "one", "paper", "article", "study", "report", "publication",
        "document", "research", "piece", "item", "option", "me", "i", "want",
        "like", "know", "say", "said", "talk", "discus", "detail",
    }
)

#: A descriptive reference has to account for this much of what the reader
#: named before it is trusted, so an offhand word does not pick at random.
_DESCRIPTIVE_OVERLAP = 0.5


def presented_items(answer: str) -> tuple[str, ...]:
    """Return the items the assistant listed, in the order it listed them.

    Numbered and bulleted lines are preferred because they carry the position
    a reader counts. Quoted titles are the fallback for a prose sentence that
    names several works.
    """

    if not answer:
        return ()

    items = [
        cleaned
        for match in _LIST_LINE.finditer(answer)
        if (cleaned := _clean_item(match.group("item")))
    ]
    if len(items) >= 2:
        return tuple(items)

    quoted = [
        cleaned
        for match in _QUOTED.finditer(answer)
        if (raw := match.group("double") or match.group("curly") or match.group("single"))
        if (cleaned := _clean_item(raw))
    ]
    if len(quoted) >= 2:
        return tuple(quoted)
    return tuple(items or quoted)


def _clean_item(raw: str) -> str:
    # The model often bolds a listed title. Left in, "**Diversity...**" was
    # kept verbatim and the item never matched the document it named.
    item = _MARKDOWN_EMPHASIS.sub("", raw.strip())
    item = _LEADING_LABEL.sub("", item)
    # The gloss goes first: stripping the quotes before it leaves the closing
    # quote stranded in the middle of the line.
    item = _TRAILING_GLOSS.sub("", item).strip()
    item = item.strip("\"“”‘’'").strip()
    # A bare number or a stray fragment is not a title.
    return item if len(item) >= 6 and content_tokens(item) else ""


def resolve_reference(query: str, items: Sequence[str]) -> str | None:
    """Return the listed item the message points at, or None if unclear.

    Position is tried first because it is unambiguous when the reader gives
    one. A descriptive reference falls back to the item it overlaps most, and
    anything that matches nothing in particular resolves to nothing rather
    than to a guess.
    """

    if not items:
        return None

    positioned = _by_position(query, items)
    if positioned is not None:
        return positioned

    descriptive = _by_description(query, items)
    if descriptive is not None:
        return descriptive

    # "What does that paper say" after a single suggestion has only one
    # candidate, so there is nothing to disambiguate.
    return items[0] if len(items) == 1 else None


def _by_position(query: str, items: Sequence[str]) -> str | None:
    for match in _POSITION.finditer(query):
        word = match.group("word")
        if word:
            lowered = word.casefold()
            if lowered in _LAST_WORDS:
                return items[-1]
            index = _ORDINAL_WORDS.get(lowered)
        else:
            index = int(match.group("digit"))
        if index is not None and 1 <= index <= len(items):
            return items[index - 1]
    return None


def _by_description(query: str, items: Sequence[str]) -> str | None:
    """Return the item that accounts for what the reader actually named.

    The score is the share of the reader's own words the item covers, not the
    share of the item's words they used. "The LGBTQ one" is three words
    against a fifteen-word title: measured against the title it looks like a
    poor match, measured against the request it is an exact one.
    """

    asked = set(content_tokens(query)) - _POINTING_WORDS
    if not asked:
        return None

    best: str | None = None
    best_score = 0.0
    runner_up = 0.0
    for item in items:
        tokens = set(content_tokens(item))
        if not tokens:
            continue
        score = len(asked & tokens) / len(asked)
        if score > best_score:
            best_score, runner_up, best = score, best_score, item
        elif score > runner_up:
            runner_up = score

    if best is None or best_score < _DESCRIPTIVE_OVERLAP or best_score <= runner_up:
        return None
    return best
