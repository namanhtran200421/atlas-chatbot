"""Small, fail-closed scope checks around the grounded answer generator.

Scope is decided by grounding, not by query wording. Retrieval is already
restricted to the Membership Atlas knowledge base and the caller's access
classes, and the generator refuses any factual answer the model cannot tie to a
retrieved source. A question is therefore redirected only when it asks for
something the corpus should never answer, whatever it happens to retrieve.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from membership_rag.bedrock.retrieval import RetrievedChunk

    # ``conversation`` imports this module, so the turn type is only ever a
    # name here; the checks below read ``role`` and ``content`` structurally.

_BIOGRAPHY_REQUEST = re.compile(
    r"\bwho\s+(?:is|was)\s+"
    r"(?!(?:Cultural Infusion|Membership Atlas|Atlas|Oriana)\b)",
    re.IGNORECASE,
)
_PERSON_DETAIL_REQUEST = re.compile(
    r"\b(?:explain|describe|summarise|summarize)\s+"
    r"[A-Z][a-z]+(?:['’]s)?\s+(?:role|life|story|biography)\b"
    r"|\b(?:role|life|story|biography)\s+of\s+[A-Z][a-z]+\b"
)
_POSSESSIVE_ROLE = re.compile(
    r"\b(?!(?:Atlas|Infusion)\b)[A-Z][a-z]+(?:['’]s)\s+"
    r"(?:role|life|story|biography)\b"
)
_LIVE_EXTERNAL_FACT = re.compile(
    r"\b(?:weather|forecast|temperature|air quality|current time|local time|"
    r"what time is it|stock price|share price|exchange rate|sports? score)\b",
    re.IGNORECASE,
)
_GREETING = re.compile(
    r"(?:hi+|hey+|hello+|yo|howdy|greetings|"
    r"good\s+(?:morning|afternoon|evening|day)|"
    r"how\s+(?:are|were|r)\s+(?:you|u|ya|things)"
    r"(?:\s+(?:doing|going|today|feeling))?|"
    r"how(?:'s|s|\s+is)\s+(?:it\s+going|everything|your\s+day)|"
    r"what(?:'s|s|\s+is)\s+up|sup|"
    r"thanks(?:\s+(?:a\s+lot|so\s+much|heaps))?|"
    r"thank\s+you(?:\s+(?:very\s+much|so\s+much))?|thankyou|cheers|"
    r"(?:good)?bye|see\s+(?:you|ya)(?:\s+later)?|"
    r"have\s+a\s+(?:good|great|nice)\s+(?:day|one|evening)|"
    r"nice\s+to\s+meet\s+you)"
    r"[!.?\s]*",
    re.IGNORECASE,
)
# "Hey Oriana!", "thanks mate" — a direct address does not change the intent.
_DIRECT_ADDRESS = re.compile(
    r"\s*,?\s*(?:oriana|mate|there|friend|team)\b",
    re.IGNORECASE,
)
_IDENTITY = re.compile(
    r"(?:who\s+are\s+you|who(?:'s|s|\s+is)\s+Oriana|"
    r"what(?:'s|s|\s+is)\s+your\s+name|"
    # "My name is Nam, what is yours" - the second half is still the same
    # question, and without it the introduction reads as a topic and anchors
    # every later search on the person's name.
    r"what(?:'s|s|\s+is)\s+yours|"
    r"what\s+(?:is|about)\s+yours|"
    r"what(?:'s|s|\s+is)\s+(?:the\s+name\s+of\s+)?(?:your|the)\s+company|"
    r"who\s+(?:made|created|built)\s+you|"
    r"are\s+you\s+(?:a\s+)?(?:robot|bot|human|real|an?\s+ai))[?.!\s]*",
    re.IGNORECASE,
)
# "what can you do?", "help" and friends describe the assistant, not the
# corpus. Retrieving for them returned whichever articles happened to match,
# so "help" was answered with a list of LGBTQI+ allyship resources.
_CAPABILITY = re.compile(
    r"(?:please\s+|pls\s+)?"
    r"(?:what(?:'s|s|\s+is)?\s+(?:this|that)(?:\s+(?:about|for))?|"
    r"what\s+(?:can|do)\s+(?:you|u)\s+(?:do|help\s+(?:me\s+)?with)|"
    r"what\s+are\s+you(?:\s+for)?|"
    r"how\s+(?:can|do)\s+(?:you|u)\s+help(?:\s+me)?|"
    r"(?:can|could|will)\s+(?:you|u)\s+help(?:\s+me)?|"
    r"help(?:\s+me)?|"
    r"what\s+(?:topics|things|questions)\s+can\s+(?:i|you)\s+ask"
    r"(?:\s+(?:you\s+)?about)?|"
    r"what\s+(?:info|information|data|content)\s+do\s+you\s+have)"
    r"[?.!\s]*",
    re.IGNORECASE,
)
_SOCIAL_CHAT = re.compile(
    r"(?:nice\s+to\s+meet\s+you(?:\s+too)?|"
    r"(?:that(?:'s|\s+is|\s+was)|sounds?)\s+"
    r"(?:nice|great|good|cool|interesting|helpful|amazing|lovely)|"
    r"i\s+(?:like|love|appreciate)\s+(?:that|this|your\s+answer)|"
    r"i(?:'m|\s+am)\s+(?:excited|curious|happy|glad|just\s+browsing)|"
    r"tell\s+me\s+a\s+joke|make\s+me\s+laugh)"
    r"[!.?\s]*",
    re.IGNORECASE,
)
# A bare acknowledgement is a whole turn on its own, and they arrive in runs
# separated by nothing but a space: "cool thanks", "haha nice", "perfect
# thanks heaps". Splitting on punctuation alone missed those, so the turn was
# searched, and it then dragged the small talk into the next question too.
_ACKNOWLEDGEMENT_WORD = (
    r"(?:ha)+h?a*|lol|hehe|nice(?:\s+one)?|cool|sweet|perfect|awesome|"
    r"brilliant|lovely|excellent|wonderful|amazing|fair\s+enough|"
    r"no\s+worries|all\s+good|sounds?\s+good|great\s+stuff|got\s+it|"
    r"makes\s+sense|good\s+to\s+know|very\s+helpful|"
    r"thanks(?:\s+(?:a\s+lot|so\s+much|heaps|mate))?|thank\s+you|thx|ty|"
    r"cheers|ok(?:ay)?|alright|great|good|yep|yeah|sure"
)
_ACKNOWLEDGEMENT_RUN = re.compile(
    rf"(?:(?:{_ACKNOWLEDGEMENT_WORD})[\s,!.]*){{1,4}}",
    re.IGNORECASE,
)
# "great!, you?" is two social halves, and the compound rule below tests each
# half on its own, so both need a pattern. Without them the turn fell through
# to retrieval and a "how are you?" was answered with research papers.
_WELLBEING_REPLY = re.compile(
    r"(?:i(?:'m|\s+am)\s+|i\s+feel\s+)?"
    r"(?:doing\s+|pretty\s+|really\s+|very\s+|all\s+)?"
    r"(?:great|good|well|fine|ok(?:ay)?|alright|awesome|fantastic|excellent|"
    r"wonderful|lovely|not\s+bad|can't\s+complain|so\s+far\s+so\s+good)"
    r"(?:\s*,?\s*(?:thanks|thank\s+you|thx|ty))?"
    r"[!.?\s]*",
    re.IGNORECASE,
)
_RECIPROCAL_QUESTION = re.compile(
    r"(?:and\s+|so\s+|but\s+)?"
    r"(?:how\s+(?:about|are)\s+|what\s+about\s+|what(?:'s|s|\s+is)\s+)?"
    r"(?:you|u|yourself|urself|yours)(?:\s+too)?"
    r"[?!.\s]*",
    re.IGNORECASE,
)
# An introduction is social context, not a claim about the Atlas corpus.
# Match the whole message so an attached factual request still needs sources.
_NAME_WORD = r"[^\W\d_][\w'’-]{0,39}"
_GREETING_PREFIX = r"(?:(?i:hi+|hey+|hello+|yo)\s*[,!.]?\s+)?"
#: "im Nam btw", "I'm Nam by the way" - an aside tacked on the end.
_ASIDE_SUFFIX = r"(?:\s*,?\s*(?i:btw|by\s+the\s+way|anyway))?"
_EXPLICIT_INTRODUCTION = re.compile(
    rf"{_GREETING_PREFIX}"
    rf"(?i:my\s+name\s+is|(?:you\s+can\s+|please\s+)?call\s+me)"
    rf"\s+(?P<name>{_NAME_WORD}(?:\s+{_NAME_WORD})?)"
    rf"{_ASIDE_SUFFIX}"
    r"[!.?\s]*",
)
_INLINE_NAME = re.compile(
    r"(?i:\bmy\s+name\s+is\s+)"
    r"(?P<name>[A-Z][\w'’-]{0,39}(?:\s+[A-Z][\w'’-]{0,39})?)\b",
)
_SELF_INTRODUCTION = re.compile(
    # "im Nam btw" arrives without the apostrophe more often than with it.
    rf"{_GREETING_PREFIX}(?i:i(?:'m|’m|m|\s+am))\s+"
    r"(?P<name>[A-Z][\w'’-]{0,39}(?:\s+[A-Z][\w'’-]{0,39})?)"
    rf"{_ASIDE_SUFFIX}"
    r"[!.?\s]*",
)
_NAME_RECALL = re.compile(
    r"(?:what(?:'s|’s|s|\s+is)\s+my\s+name|"
    r"do\s+you\s+remember\s+my\s+name|"
    r"what\s+did\s+i\s+(?:tell|ask)\s+you\s+to\s+call\s+me)"
    r"(?:\s+again)?"
    r"[?.!\s]*",
    re.IGNORECASE,
)
_ARITHMETIC = re.compile(
    r"(?:(?:what is|what's|calculate|solve)\s+)?"
    r"\d+(?:\.\d+)?(?:\s*[+*/-]\s*\d+(?:\.\d+)?)+[?.!\s]*",
    re.IGNORECASE,
)
# A question about the conversation itself, not about Atlas. Searching the
# corpus for "what have we talked about?" returns whichever articles happen to
# match and the answer then fails grounding, so the reader is asked to clarify
# a question that was already perfectly clear.
_CONVERSATION_META = re.compile(
    r"(?:(?:so\s+)?what\s+(?:have|did)\s+(?:we|i|you)\s+"
    r"(?:talk(?:ed)?|chat(?:ted)?|discuss(?:ed)?|say|said|ask(?:ed)?|tell|told)"
    r"(?:\s+(?:about|me|you))*|"
    r"what\s+(?:was|were)\s+(?:my|your)\s+(?:last|previous|first)\s+"
    r"(?:question|message|answer)|"
    r"(?:can\s+you\s+)?(?:summari[sz]e|recap)\s+(?:our|this|the)\s+"
    r"(?:conversation|chat|discussion)|"
    r"do\s+you\s+remember\s+what\s+(?:we|i)\s+"
    r"(?:talked\s+about|discussed|said)|"
    r"remind\s+me\s+(?:what|of\s+what)\s+(?:we|i|you)\s+"
    r"(?:talked\s+about|discussed|said|covered)|"
    # "What did I say my org size was?" asks after the reader's own words.
    r"what\s+did\s+i\s+(?:say|tell\s+you)\b[^?]*)"
    r"(?:\s+(?:so\s+far|today|earlier|yet|before))?"
    r"[?.!\s]*",
    re.IGNORECASE,
)
_QUESTION_PREFIX = re.compile(
    r"^(?:(?:what|when|where)\s+(?:is|are|was|were)|"
    r"tell me about|explain|describe)\s+(?:the\s+|a\s+|an\s+)?",
    re.IGNORECASE,
)
_EVENT_SUFFIX = frozenset({"begins", "ends", "day", "week", "eve", "festival"})


# Chat shorthand, expanded only while deciding whether a message is small talk.
# The query sent to retrieval is never rewritten.
_CHAT_SHORTHAND = {
    "r": "are",
    "u": "you",
    "ur": "your",
    "yr": "your",
    "abt": "about",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "ty": "thanks",
    "tks": "thanks",
    "wat": "what",
    "wot": "what",
    "wut": "what",
    "whos": "who is",
    "hru": "how are you",
    "hbu": "how about you",
    "wyd": "what are you doing",
    "sup": "what is up",
    "gm": "good morning",
    "ga": "good afternoon",
    "ge": "good evening",
    "cya": "see you",
    "bye": "bye",
    "yo": "yo",
}
_SHORTHAND_WORD = re.compile(r"[A-Za-z']+")


def introduced_name(message: str) -> str | None:
    """Read a name explicitly supplied in a standalone user introduction."""

    cleaned = message.strip()
    for pattern in (_EXPLICIT_INTRODUCTION, _SELF_INTRODUCTION):
        match = pattern.fullmatch(cleaned)
        if match:
            return match.group("name")
    match = _INLINE_NAME.search(cleaned)
    if match:
        return match.group("name")
    return None


def is_name_recall(query: str) -> bool:
    """Identify a direct request to recall the user's own stated name."""

    return bool(_NAME_RECALL.fullmatch(query.strip()))


def is_social_chat(query: str) -> bool:
    """Identify a standalone social turn that needs no factual answer."""

    cleaned = query.strip()
    addressed = _DIRECT_ADDRESS.sub("", cleaned).strip(" ,.!?")
    candidates = (cleaned, addressed, _expand_chat_shorthand(cleaned), _expand_chat_shorthand(addressed))
    patterns = (
        _GREETING,
        _EXPLICIT_INTRODUCTION,
        _SELF_INTRODUCTION,
        _SOCIAL_CHAT,
        _WELLBEING_REPLY,
        _RECIPROCAL_QUESTION,
        _ACKNOWLEDGEMENT_RUN,
    )
    if any(
        pattern.fullmatch(candidate)
        for candidate in candidates
        for pattern in patterns
    ):
        return True
    # A compound acknowledgement such as "Thanks, that's interesting!" is
    # still social, but every part must be social so a real question survives.
    parts = [part.strip() for part in re.split(r"[,;.!?]+", cleaned) if part.strip()]
    return len(parts) > 1 and all(
        any(pattern.fullmatch(part) for pattern in patterns)
        for part in parts
    )


def _expand_chat_shorthand(text: str) -> str:
    return _SHORTHAND_WORD.sub(
        lambda match: _CHAT_SHORTHAND.get(match.group(0).casefold(), match.group(0)),
        text,
    )


def is_simple_utility(query: str) -> bool:
    """Return whether a message can be answered without any retrieved source.

    Matching is whole-message on purpose: "What is 1+1? Also tell me about
    Mary." is not a utility. Trailing punctuation and a direct address are
    ignored so ordinary small talk does not fall through to the grounding
    rules, where it would get a request for clarification.
    """

    cleaned = query.strip()
    addressed = _DIRECT_ADDRESS.sub("", cleaned).strip(" ,.!?")
    candidates = (
        cleaned,
        addressed,
        _expand_chat_shorthand(cleaned),
        _expand_chat_shorthand(addressed),
    )
    if is_social_chat(query) or any(
        pattern.fullmatch(candidate)
        for candidate in candidates
        for pattern in (
            _IDENTITY,
            _NAME_RECALL,
            _ARITHMETIC,
            _CAPABILITY,
            _CONVERSATION_META,
        )
    ):
        return True

    # "thanks! what have we discussed?" is two utilities in one message.
    # `is_social_chat` splits compounds too, but only against the social
    # patterns, so this half-social, half-meta message fell through to
    # retrieval and was answered with whichever events happened to match.
    # Every part must still be a utility, so "What is 1+1? Tell me about
    # Mary." keeps needing sources.
    parts = [part.strip() for part in re.split(r"[,;.!?]+", cleaned) if part.strip()]
    return len(parts) > 1 and all(is_simple_utility(part) for part in parts)


_SQUEEZE_REPEATS = re.compile(r"([A-Za-z])\1+")
_VAGUE_CONTINUATION = re.compile(
    r"(?:(?:ok(?:ay)?|sure|yes|yeah|yep|alright|cool|nice|great)\s*[,!.]*\s*)*"
    r"(?:(?:let'?s|lets)\s+(?:explore|dive\s+into|look\s+at|see|do|discuss)"
    r"\s+(?:that|this|it|them)|"
    r"(?:tell\s+me\s+)?more|go\s+on|carry\s+on|continue|keep\s+going|"
    r"explore\s+(?:that|this|it)|sounds?\s+good|i'?m\s+interested)"
    r"[!.?\s]*",
    re.IGNORECASE,
)


def is_vague_continuation(query: str) -> bool:
    """Return whether a message asks to go on without saying what about.

    "Sure, let's explore that" carries no subject of its own. When nothing
    earlier supplies one, searching the corpus for the words themselves
    returns an arbitrary document, and the conversation then follows that
    document instead of the reader.
    """

    cleaned = query.strip()
    addressed = _DIRECT_ADDRESS.sub("", cleaned).strip(" ,.!?")
    candidates = (
        cleaned,
        addressed,
        _expand_chat_shorthand(cleaned),
        _expand_chat_shorthand(addressed),
    )
    # "SUre lets expllore that" is how the message actually arrives. Squeezing
    # repeated letters only ever adds a candidate - a correctly spelled
    # "discuss" still matches as typed - so a slip does not cost the reader a
    # sensible reply.
    return any(
        _VAGUE_CONTINUATION.fullmatch(candidate)
        for candidate in (*candidates, *(_SQUEEZE_REPEATS.sub(r"\1", c) for c in candidates))
    )


def is_out_of_scope(query: str, chunks: Sequence[RetrievedChunk]) -> bool:
    """Return whether a question must be redirected before generation.

    General biographies and live external facts are refused here. Atlas
    articles and calendar events can mention them without supporting the
    requested answer. Other questions reach grounding checks.
    """

    if _LIVE_EXTERNAL_FACT.search(query):
        return True
    biography = bool(
        _BIOGRAPHY_REQUEST.search(query)
        or _PERSON_DETAIL_REQUEST.search(query)
        or _POSSESSIVE_ROLE.search(query)
    )
    # Naming a retrieved document makes the question about that document,
    # unless it is squarely a request for a person's life story: "Assumption
    # of Mary" is an event in the calendar, not a biography of Mary.
    if matches_source_title(query, chunks) and not biography:
        return False
    return biography


#: Function words carry no signal about which document is meant, and counting
#: them let a rival title score almost as well as the right one purely on
#: "in the ... and ... for a".
_TITLE_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "for", "in", "on", "to", "with",
        "your", "our", "its", "it", "is", "are", "was", "were", "be", "from",
        "by", "at", "as", "that", "this", "these", "those", "what", "do",
        "does", "how", "we", "you", "s",
    }
)

#: How much of a document's title must appear in the assistant's last answer
#: before "this paper" is taken to mean that document. With function words
#: removed the separation is wide: the paper under discussion scores about
#: three quarters, the nearest rival on the same theme under a half.
_REFERENCE_TITLE_OVERLAP = 0.6


def _fold(token: str) -> str:
    """Fold a plural onto its singular so "strategies" matches "strategy"."""

    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith("sses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_tokens(text: str) -> list[str]:
    """Return a title's meaningful words, folded for plurals."""

    return [
        folded
        for token in _tokens(text)
        if token not in _TITLE_STOPWORDS
        if (folded := _fold(token))
    ]


def narrow_to_referenced_document(
    chunks: Sequence[RetrievedChunk],
    anchor: str | None,
) -> list[RetrievedChunk]:
    """Keep only the document the assistant's last answer was describing.

    Searching with that answer's wording is not enough on its own: the prose
    around a title ("hybrid work and its impact on diversity, equity and
    inclusion") embeds close to every other paper on the same theme, so
    "explain this paper" drifted to a different one mid-conversation. The
    title is the reliable signal, so prefer the retrieved document whose title
    the answer actually covers, and fall back to the full results when nothing
    stands out.
    """

    if not anchor or not chunks:
        return list(chunks)

    anchor_tokens = set(content_tokens(anchor))
    if not anchor_tokens:
        return list(chunks)

    best_title: str | None = None
    best_overlap = 0.0
    for chunk in chunks:
        title = chunk.metadata.get("title")
        if not isinstance(title, str):
            continue
        title_tokens = content_tokens(title)
        if not title_tokens:
            continue
        overlap = sum(token in anchor_tokens for token in title_tokens) / len(
            title_tokens
        )
        if overlap > best_overlap:
            best_overlap, best_title = overlap, title

    if best_title is None or best_overlap < _REFERENCE_TITLE_OVERLAP:
        return list(chunks)
    # Put the document in hand first and keep the rest behind it. Dropping the
    # others outright left a single excerpt, which was too thin to answer
    # "who wrote it?" and failed the grounding check instead.
    matched = [c for c in chunks if c.metadata.get("title") == best_title]
    return matched + [c for c in chunks if c.metadata.get("title") != best_title]


def matches_source_title(query: str, chunks: Sequence[RetrievedChunk]) -> bool:
    """Return whether the query names the title of a retrieved Atlas source."""

    query_tokens = _tokens(query)
    query_text = " ".join(query_tokens)

    subject = _tokens(_QUESTION_PREFIX.sub("", query.strip()).strip(" ?!."))
    for chunk in chunks:
        title = chunk.metadata.get("title")
        if not isinstance(title, str):
            continue
        title_tokens = _tokens(title)
        if len(title_tokens) >= 2 and " ".join(title_tokens) in query_text:
            return True
        if not subject:
            continue
        if title_tokens[: len(subject)] != subject:
            continue
        if len(subject) >= 2 or len(title_tokens) == 1:
            return True
        if len(title_tokens) > 1 and title_tokens[1] in _EVENT_SUFFIX:
            return True
    return False


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[\w]+", normalized, flags=re.UNICODE)
