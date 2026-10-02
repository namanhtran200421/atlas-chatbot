"""Session-scoped conversation memory for the grounded-answer pipeline.

Nothing is stored server-side. Each request carries the recent turns the
browser still holds in memory, so a reload — or "New chat" — starts an empty
conversation again.

History is therefore untrusted client input. It is capped in turns and
characters, stripped of control characters, screened by the same guardrails as
a live question, and used to resolve follow-ups and remember details the user
shared about themselves. It is never grounding for Atlas or external facts.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from membership_rag.guardrails import inspect_query
from membership_rag.reference import presented_items, resolve_reference
from membership_rag.scope import content_tokens, is_simple_utility

#: The browser can retain fifty exchanges; generation uses a compact view.
MAX_HISTORY_TURNS = 100

#: Per-turn character cap. A listing answer carries five titles and their
#: descriptions, and truncating it lost the items a reader then referred to
#: by position, so this leaves room for the whole list.
MAX_TURN_CHARACTERS = 1_400

#: How many of the assistant's own turns to look back through for the list a
#: positional reference counts against. Once it has answered about one paper
#: at a time, several turns can separate the reader from the list they are
#: counting, so this counts answers rather than raw turns.
_REFERENCE_LOOKBACK = 10

#: How much of the assistant's last answer anchors a reference such as "this
#: paper": enough to carry the subject and its title, short enough that the
#: question the person actually asked still dominates the search.
MAX_ANCHOR_CHARACTERS = 300

_ROLES = ("user", "assistant")

# A message that cannot be searched on its own: it points back at something
# already said. Retrieval for "how much is it?" only works once "it" is known.
_REFERRING_LANGUAGE = re.compile(
    r"\b(?:it|its|it's|that|this|those|these|they|them|their|theirs|"
    r"the\s+(?:same|other|others|first|second|last|one)|"
    r"one\s+of\s+(?:them|those))\b",
    re.IGNORECASE,
)
# "there" used to be in that list, but almost every use of it here is the
# existential one: "what plans are there?" is a standalone question, and
# treating it as a follow-up prepended the previous message to the search,
# which then matched research papers and produced invented plan names. A
# genuine "what's there?" is short enough for the word-count rule below.
_FOLLOW_UP_OPENER = re.compile(
    r"^(?:and|but|so|also|then|what\s+about|how\s+about|what\s+else|"
    r"any\s+(?:others?|more)|anything\s+else|more|tell\s+me\s+more|"
    r"go\s+on|why|why\s+not|which\s+one|who\s+else|when|where|how\s+much|"
    r"how\s+many|ok(?:ay)?|yes|yeah|no|sure|please\s+do)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[\w']+", re.UNICODE)
# "What does this paper say" names a document without naming which one. Only
# the assistant's turn holds the title, so these are the references worth
# anchoring on it. A vague "let's explore that" is deliberately excluded:
# after a social turn there is nothing to anchor, and borrowing one answered
# "how are you?" with whatever the small talk happened to mention.
_DOCUMENT_REFERENCE = re.compile(
    r"\b(?:this|that|the|these|those)\s+"
    r"(?:paper|article|study|report|publication|document|research|"
    r"piece|event|plan|one)\b"
    r"|\bwhat\s+does\s+it\s+say\b"
    r"|\b(?:tell|say)\s+me\s+more\s+about\s+(?:it|that|this)\b"
    # "What were the main findings?" is about the paper under discussion just
    # as much as "what does this paper say", but names no document, so it was
    # searched on its own and matched nothing in particular.
    r"|\b(?:the|its|their)\s+(?:main\s+|key\s+|overall\s+)?"
    r"(?:findings?|results?|conclusions?|outcomes?|methodology|methods?|"
    r"abstract|authors?|argument|recommendations?|takeaways?)\b"
    r"|\bwho\s+wrote\s+(?:it|that|this)\b"
    r"|\bwhen\s+was\s+(?:it|that|this)\s+(?:published|written)\b",
    re.IGNORECASE,
)
# "Let's go with the fourth one", "number 3", "the last one". A reader picking
# from a list the assistant just gave is making the same kind of reference,
# and searching the words themselves found a paper about the Fourth
# Industrial Revolution.
_POSITIONAL_REFERENCE = re.compile(
    r"\b(?:the\s+|that\s+|this\s+)?"
    r"(?:first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|sixth|6th|"
    r"seventh|7th|eighth|8th|ninth|9th|tenth|10th|last|final)\s+one\b"
    r"|\b(?:go\s+with|pick|choose|take|do|try|read|open)\s+"
    r"(?:the\s+|that\s+|this\s+)?"
    r"(?:first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|sixth|6th|"
    r"seventh|7th|eighth|8th|ninth|9th|tenth|10th|last|final)\b"
    r"|\b(?:number|option|item|no\.)\s*\d{1,2}\b"
    r"|#\s*\d{1,2}\b"
    # "the LGBTQ one", "the glass door one": a reader picks an item by what it
    # was about as readily as by where it sat in the list.
    r"|\b(?:the|that|this)\s+(?:[\w+-]+\s+){1,3}?one\b",
    re.IGNORECASE,
)


#: Words that carry a request along without naming what it is about: "go
#: deeper", "tell me more", "can you elaborate". A message built only from
#: these has no subject of its own, so it must take one from the answer it is
#: replying to. Without this, "go deep into it" was searched together with the
#: reader's earlier "recommend me a paper" and found a different paper.
_CARRIER_WORDS = frozenset(
    {
        "go", "going", "goes", "gone", "deep", "deeper", "dive", "diving",
        "into", "expand", "elaborate", "further", "detail", "unpack",
        "explore", "continue", "keep", "break", "down", "more", "tell",
        "explain", "describe", "show", "say", "said", "talk", "discus",
        "me", "my", "i", "please", "can", "could", "would", "will", "let",
        "us", "sure", "ok", "okay", "yes", "yeah", "great", "good", "nice",
        "cool", "thanks", "thank", "like", "love", "want", "interested",
        "sound", "much", "many", "else", "other", "another", "anything",
        "any", "one", "read", "open", "pick", "choose", "take", "try",
        "about", "give", "bit", "little", "again", "now", "then", "also",
        "well", "hey", "hi", "some", "which", "few", "couple", "several",
        "example", "number", "our", "back", "actually", "just",
        "so", "right", "alright", "anyway", "still",
    }
)

#: "We have about 400 employees", "actually we're closer to 2000 people". A
#: reader supplies their situation a piece at a time, and each piece belongs
#: to the thing they are asking about. Searched on its own, "we have about 400
#: employees" matches whatever mentions headcount instead of their question.
_PERSONAL_STATEMENT = re.compile(
    r"^(?:(?:actually|btw|oh|also|and|so|well)\s*,?\s*)*"
    r"(?:we|i|our|my)\b(?:'|’)?",
    re.IGNORECASE,
)


#: "Show me a few", "any others?", "more examples" - a continuation asking for
#: breadth rather than depth on one item.
_ASKS_FOR_MORE_ITEMS = re.compile(
    r"\b(?:a\s+few|a\s+couple|some\s+more|more\s+(?:examples?|papers?|options?)|"
    r"(?:any\s+)?others?|anything\s+else|what\s+else|show\s+me\s+(?:a\s+few|some|more)|"
    r"list\s+(?:them|some|a\s+few)|give\s+me\s+(?:a\s+few|some|more))\b",
    re.IGNORECASE,
)


def _states_their_situation(message: str) -> bool:
    """Return whether the message tells the assistant about the reader."""

    return "?" not in message and bool(_PERSONAL_STATEMENT.match(message.strip()))


def _starts_a_topic(message: str) -> bool:
    """Return whether the message raises a subject rather than continuing one."""

    return (
        not is_simple_utility(message)
        and not _points_backwards(message)
        and not _states_their_situation(message)
        and not _is_follow_up(message)
    )


def _points_backwards(message: str) -> bool:
    """Return whether the message only makes sense against what came before."""

    return (
        bool(_DOCUMENT_REFERENCE.search(message))
        or bool(_POSITIONAL_REFERENCE.search(message))
        or not _names_a_subject(message)
    )


def _names_a_subject(message: str) -> bool:
    """Return whether the message says what it is about.

    "How much is membership?" does and is searched as asked. "How much is it?"
    does not and has to borrow one.
    """

    return bool(set(content_tokens(message)) - _CARRIER_WORDS)


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """One earlier message in the current browser session."""

    role: str
    content: str


def normalise_history(
    raw: object,
    *,
    maximum_turns: int = MAX_HISTORY_TURNS,
) -> tuple[ConversationTurn, ...]:
    """Return the trustworthy part of a client-supplied history.

    Unusable entries are skipped rather than rejected: a single malformed turn
    should cost the reader their memory, not their answer.
    """

    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    if maximum_turns <= 0:
        return ()

    turns: list[ConversationTurn] = []
    for entry in raw[-maximum_turns:]:
        turn = _turn_from(entry)
        if turn is not None:
            turns.append(turn)
    return tuple(turns[-maximum_turns:])


def _turn_from(entry: object) -> ConversationTurn | None:
    if not isinstance(entry, dict):
        return None
    role = entry.get("role")
    content = entry.get("content") or entry.get("text")
    if role not in _ROLES or not isinstance(content, str):
        return None

    cleaned = _sanitise(content)
    if not cleaned:
        return None
    # Client-supplied assistant turns are untrusted too: a forged earlier
    # answer must not re-enter the pipeline as an instruction.
    if not inspect_query(cleaned).allowed:
        return None
    return ConversationTurn(role=role, content=cleaned)


def _sanitise(content: str) -> str:
    stripped = "".join(
        character
        for character in content
        if unicodedata.category(character) != "Cc" or character in {"\t", "\n"}
    )
    collapsed = re.sub(r"[ \t]{2,}", " ", stripped).strip()
    return _truncate(collapsed, MAX_TURN_CHARACTERS)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    clipped = text[:limit].rstrip()
    boundary = clipped.rfind(" ")
    if boundary > limit // 2:
        clipped = clipped[:boundary]
    return f"{clipped.rstrip()}…"


def contextual_query(
    query: str,
    history: Sequence[ConversationTurn] = (),
) -> str:
    """Return the text to search the knowledge base with.

    A self-contained question is searched as asked. A follow-up includes the
    most recent substantive user question and the intervening follow-ups.
    Small talk does not replace the topic.
    """

    message = query.strip()
    if not history or not message or is_simple_utility(message):
        # Greetings and thanks never borrow the topic: doing so would search
        # the corpus for whatever the conversation last happened to be about.
        return message

    # Naming a document is explicit: it points at what the assistant chose,
    # which may not be what the reader last asked for, so it takes precedence
    # over their earlier questions. "What were the main findings?" belongs
    # here too - no referring word, long enough to look self-contained, but a
    # reference all the same.
    explicit = bool(
        _DOCUMENT_REFERENCE.search(message)
        or _POSITIONAL_REFERENCE.search(message)
    )
    # Merely having no subject is weaker evidence: "how much is it?" still
    # means the thing the reader asked about, so the answer is only borrowed
    # when nothing they said names a subject either.
    subjectless = not _names_a_subject(message)
    # A statement about the reader's own situation belongs to whatever they
    # are asking about, so it carries the topic with it.
    situational = _states_their_situation(message)
    if (
        not explicit
        and not subjectless
        and not situational
        and not _is_follow_up(message)
    ):
        return message

    previous = _recent_topic(
        history,
        query=message,
        anchor=explicit or subjectless,
        explicit=explicit,
    )
    if not previous:
        return message
    return " ".join((*previous, message))


def _is_follow_up(message: str) -> bool:
    if is_simple_utility(message):
        # Greetings and identity questions never reach retrieval anyway, and
        # borrowing the previous question would only distort them.
        return False
    if _REFERRING_LANGUAGE.search(message):
        return True
    if _FOLLOW_UP_OPENER.match(message):
        # "How much is it?" cannot stand alone. "How much is membership?"
        # opens the same way and can, and dragging the previous question onto
        # it sent the search somewhere else entirely.
        return not _names_a_subject(message)
    # A compound reply carries its request in the last clause: "sounds good,
    # tell me more" is the same turn as "tell me more".
    parts = [part.strip() for part in re.split(r"[,;.!?]+", message) if part.strip()]
    if len(parts) > 1:
        return _is_follow_up(parts[-1])
    # "cheaper options?" carries no referring word but cannot stand alone.
    return len(_WORD.findall(message)) <= 3


def _resolved_item(
    query: str,
    history: Sequence[ConversationTurn],
) -> str | None:
    """Return the listed item the message points at, if it picks one out."""

    # "Number 1" counts against the list the assistant gave, which may be
    # several turns back: once it has answered about one paper, its latest
    # turn holds a single title, and resolving "number 1" against that
    # returned the paper already under discussion instead of the first of the
    # list. So keep walking until a real list turns up, remembering the most
    # recent single suggestion in case nothing better exists.
    lone_suggestion: str | None = None
    inspected = 0
    for turn in reversed(history):
        if turn.role == "user" and _starts_a_topic(turn.content):
            # The reader asked something new here, so anything listed before
            # it belongs to the topic they left. Without this boundary, "so
            # which one now?" - asked after returning to membership - resolved
            # to a research paper from fourteen turns earlier.
            break
        if turn.role != "assistant":
            continue
        inspected += 1
        if inspected > _REFERENCE_LOOKBACK:
            break
        items = presented_items(turn.content)
        if not items:
            continue
        if len(items) == 1:
            if lone_suggestion is None:
                lone_suggestion = items[0]
            continue
        return resolve_reference(query, items) or lone_suggestion
    return lone_suggestion


def _recent_topic(
    history: Sequence[ConversationTurn],
    *,
    query: str = "",
    anchor: bool = False,
    explicit: bool = False,
) -> tuple[str, ...]:
    if anchor:
        # "This paper" names whatever the assistant just described, not
        # whatever the person asked earlier. After "pick one and let's talk
        # about it", the subject exists only in the answer that picked it, so
        # that answer is the whole topic. Mixing in the earlier questions
        # buried the chosen paper under general talk about Atlas and the
        # search returned a different paper entirely.
        #
        # This steers retrieval only. The text never reaches the model as
        # context, and retrieval stays inside the access-controlled corpus, so
        # a forged earlier answer can shift which Atlas records are searched
        # but cannot add claims to an answer or reach anything unauthorised.
        resolved = _resolved_item(query, history) if query else None
        if resolved:
            return (resolved,)
        if explicit:
            last_answer = _last_answer(history)
            if last_answer:
                return (last_answer,)

    topic: list[str] = []
    named_the_subject = False
    for turn in reversed(history):
        if turn.role != "user" or is_simple_utility(turn.content):
            continue
        topic.append(turn.content)
        # A turn that is itself a reference names nothing: "ok what about
        # number 1" pointed at a list, and stopping there left the search with
        # a position and no paper. A turn that only supplies the reader's
        # situation - "actually we're closer to 2000 people" - qualifies the
        # topic rather than setting it, so it does not end the walk either.
        if (
            not _is_follow_up(turn.content)
            and not _points_backwards(turn.content)
            and not _states_their_situation(turn.content)
        ):
            named_the_subject = True
            break
        if len(topic) >= 3:
            break

    if anchor and not named_the_subject:
        # Nothing the reader said names a subject either, so the answer they
        # are replying to is all there is to go on.
        last_answer = _last_answer(history)
        if last_answer:
            topic.append(last_answer)
    return tuple(reversed(topic))


def has_topic(history: Sequence[ConversationTurn]) -> bool:
    """Return whether anything has been raised that a reply could continue.

    After nothing but small talk there is no subject to go deeper into, and
    anchoring on the small talk answered "let's explore that" with whatever
    the chit-chat happened to mention.
    """

    return bool(_recent_topic(history))


def referenced_item(query: str, history: Sequence[ConversationTurn]) -> str | None:
    """Return the previously listed item the message picks out, if any."""

    return _resolved_item(query, history)


def continues_previous_subject(query: str) -> bool:
    """Return whether the turn is about whatever was already being discussed.

    Covers both naming a document without saying which ("this paper", "the
    fourth one") and carrying on with no subject at all ("go deep into it").
    Retrieval finds the right document either way, but the words the reader
    used can still pull the answer elsewhere: "go deep into it" led to a paper
    called "A Deep Dive into DEI Practitioners" that happened to be retrieved
    alongside. Narrowing to the document in hand removes the temptation.
    """

    message = query.strip()
    if not message or is_simple_utility(message):
        return False
    if _ASKS_FOR_MORE_ITEMS.search(message):
        # "Show me a few", "any others?" continue the topic but want breadth,
        # so pinning them to one document leaves nothing to show.
        return False
    return references_a_document(message) or not _names_a_subject(message)


def references_a_document(query: str) -> bool:
    """Return whether the message points at a document without naming it."""

    return bool(
        _DOCUMENT_REFERENCE.search(query) or _POSITIONAL_REFERENCE.search(query)
    )


def last_answer(history: Sequence[ConversationTurn]) -> str | None:
    """Return the assistant's most recent turn, trimmed to an anchor."""

    return _last_answer(history)


def _last_answer(history: Sequence[ConversationTurn]) -> str | None:
    """Return the assistant's most recent turn, trimmed to an anchor."""

    for turn in reversed(history):
        if turn.role == "assistant":
            anchor = turn.content.strip()
            return _truncate(anchor, MAX_ANCHOR_CHARACTERS) if anchor else None
    return None


def alternating_turns(
    history: Sequence[ConversationTurn],
) -> tuple[ConversationTurn, ...]:
    """Return history shaped the way a Bedrock ``converse`` call requires.

    The turns must start with the person, alternate, and leave the next message
    to the person. Runs of one role are merged, a leading answer is dropped,
    and a trailing question — a turn whose answer never arrived — is dropped
    with it.
    """

    turns: list[ConversationTurn] = []
    for turn in history:
        if not turns:
            if turn.role != "user":
                continue
            turns.append(turn)
        elif turn.role == turns[-1].role:
            merged = _truncate(
                f"{turns[-1].content}\n{turn.content}", MAX_TURN_CHARACTERS
            )
            turns[-1] = ConversationTurn(role=turn.role, content=merged)
        else:
            turns.append(turn)

    if turns and turns[-1].role == "user":
        turns.pop()
    return tuple(turns)
