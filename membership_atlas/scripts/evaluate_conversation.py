#!/usr/bin/env python3
"""Evaluate how the pipeline routes a turn, across the ways people actually talk.

Retrieval quality is measured by the held-out gate. This measures the step
before it: given what the assistant just said, does a reply get understood as
the kind of turn it is? Every failure found by hand so far has been here - a
reply searched as though it were a fresh question, matching whatever happened
to share its words.

Nothing here calls AWS. It reports counts per category so a change can be
judged on the whole space rather than the one phrasing that prompted it.

    python scripts/evaluate_conversation.py [--verbose]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field

sys.path.insert(0, "src")

from membership_rag.conversation import (
    ConversationTurn,
    contextual_query,
    has_topic,
    normalise_history,
    referenced_item,
    references_a_document,
)
from membership_rag.scope import (
    is_simple_utility,
    is_vague_continuation,
)

LISTING = '''Here are a few research papers from our Membership Atlas:

1. "Sexual Identity in the Workplace: Reasons for (not) Coming out" - Explores disclosure.
2. "Gender, workplace preferences and firm performance: Looking through the glass door" - Gender differences.
3. "The multicultural workplace: interactive acculturation and intergroup relations" - Examines discordance.
4. "Between the corporation and the closet: Ethically researching LGBTQ+ identities" - Discusses ethics.
5. "Future office layouts for large organisations: workplace specialist perspective" - Design insights.'''

SINGLE = (
    'If you\'re interested in workplace challenges, "Challenges, experiences, and '
    'potential supports for East and Southeast Asian mothers in the workforce: a '
    'systematic review" might be a great fit. It explores the obstacles faced by '
    "Asian working mothers."
)
SOCIAL = "Hey there! Just enjoying a sunny day. How about you?"

#: What a turn should be routed to. "Attached" covers both resolving to one
#: named item and searching against the previous answer: what matters is that
#: the turn is understood as continuing, not searched as a new question.
CHAT = "chat"  # answered without retrieval
ATTACHED = "attached"  # understood as continuing the previous answer
FRESH = "fresh"  # searched as its own question
ASK = "ask"  # nothing to continue; ask what they mean


@dataclass
class Case:
    message: str
    expect: str
    context: str = LISTING
    item: int | None = None
    #: A word from the subject the turn must actually be searched against.
    #: Attaching is not enough: "go deep into it" attached to the reader's
    #: earlier "recommend me a paper" and searched without the paper at all.
    subject: str | None = None


@dataclass
class Result:
    passed: int = 0
    failures: list[str] = field(default_factory=list)


def _history(answer: str) -> tuple[ConversationTurn, ...]:
    return normalise_history(
        [
            {"role": "user", "content": "recommend me a paper"},
            {"role": "assistant", "content": answer},
        ]
    )


def _social_history(answer: str) -> tuple[ConversationTurn, ...]:
    """A conversation where nothing has been offered yet."""

    return normalise_history(
        [
            {"role": "user", "content": "great !, you ?"},
            {"role": "assistant", "content": answer},
        ]
    )


def classify(message: str, history: Sequence[ConversationTurn]) -> str:
    """Reproduce the routing the handler and generator perform.

    The gates match production: an item is only resolved for a message that
    reads as a reference, so a fresh question is never quietly attached to
    whatever the assistant said last.
    """

    if is_simple_utility(message):
        return CHAT
    search = contextual_query(message, history)
    if is_vague_continuation(message) and not has_topic(history):
        return ASK
    if references_a_document(message) and referenced_item(message, history):
        return ATTACHED
    if search != message:
        return ATTACHED
    return FRESH


def resolved(message: str, history: Sequence[ConversationTurn]) -> str | None:
    if not references_a_document(message):
        return None
    return referenced_item(message, history)


CASES: list[Case] = []

# Picking an item out of a list, by position.
for text, index in [
    ("the first one", 1),
    ("Let's go with the fourth one", 4),
    ("the second one", 2),
    ("number 3", 3),
    ("#2", 2),
    ("option 5", 5),
    ("the last one", 5),
    ("lets do the third one", 3),
    ("read the fifth one", 5),
]:
    CASES.append(Case(text, ATTACHED, item=index))

# Picking an item out of a list, by what it was about.
for text, index in [
    ("the LGBTQ one", 4),
    ("the glass door one", 2),
    ("the multicultural one", 3),
    ("the office layout one", 5),
    ("tell me about the gender one", 2),
]:
    CASES.append(Case(text, ATTACHED, item=index))

# Asking to go further into the single thing just offered. These carry no
# subject at all, so they must attach to the previous answer.
CONTINUATIONS = [
    "go deep into it",
    "i like that, go deep into it",
    "go deeper",
    "dive into it",
    "tell me more",
    "tell me more about it",
    "expand on that",
    "can you elaborate?",
    "elaborate please",
    "explain further",
    "more detail please",
    "break it down for me",
    "unpack that",
    "keep going",
    "sounds good, tell me more",
    "great, expand on it",
    "i like that, explain it",
    "explore that further",
]
for text in CONTINUATIONS:
    CASES.append(Case(text, ATTACHED, context=SINGLE, subject="mothers"))

# Asking about an aspect of the thing under discussion.
ASPECTS = [
    "what were the main findings?",
    "what are the key results?",
    "what does this paper say",
    "who wrote it?",
    "what was the methodology?",
    "what were its conclusions?",
    "when was it published?",
]
for text in ASPECTS:
    CASES.append(Case(text, ATTACHED, context=SINGLE, subject="mothers"))

# Questions that stand on their own must not be dragged onto the last answer.
FRESH_QUESTIONS = [
    "how much is membership?",
    "what plans are there?",
    "What research does Atlas have on AI?",
    "What cultural events are coming up?",
    "How do I join?",
    "recommend me a paper",
    "tell me about the Individual plan",
    "what is the Enterprise plan?",
    "do you have anything on hybrid work?",
]
for text in FRESH_QUESTIONS:
    CASES.append(Case(text, FRESH, context=SINGLE))

# Small talk and questions about the conversation need no corpus at all.
UTILITIES = [
    "hi",
    "hey Oriana!",
    "great !, you ?",
    "good, you?",
    "thanks!",
    "thanks, that's helpful",
    "what have we talked about?",
    "thanks! what have we discussed?",
    "what is your name?",
    "My name is Nam, what is yours",
    "what is 2+2?",
    "what can you do?",
]
for text in UTILITIES:
    CASES.append(Case(text, CHAT, context=SINGLE))

# A continuation with nothing yet to continue.
for text in ["lets explore that", "SUre lets expllore that", "tell me more", "go on"]:
    CASES.append(Case(text, ASK, context=SOCIAL))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()

    by_expected: dict[str, Result] = {}
    items = None
    for case in CASES:
        history = (
            _social_history(case.context)
            if case.expect == ASK
            else _history(case.context)
        )
        actual = classify(case.message, history)
        result = by_expected.setdefault(case.expect, Result())

        ok = actual == case.expect
        if ok and case.item is not None:
            from membership_rag.reference import presented_items

            items = presented_items(case.context)
            ok = resolved(case.message, history) == items[case.item - 1]
        if ok and case.subject is not None:
            searched = resolved(case.message, history) or contextual_query(
                case.message, history
            )
            ok = case.subject.casefold() in searched.casefold()
            if not ok:
                result.failures.append(
                    f"{case.message!r} searched without {case.subject!r}: "
                    f"{searched[:70]!r}"
                )
                continue

        if ok:
            result.passed += 1
        else:
            result.failures.append(f"{case.message!r} -> {actual} (want {case.expect})")

    total = sum(r.passed for r in by_expected.values())
    count = len(CASES)
    print(f"{'category':10s} {'pass':>7s}")
    for expected, result in sorted(by_expected.items()):
        size = result.passed + len(result.failures)
        print(f"{expected:10s} {result.passed:3d}/{size:<3d}")
        if result.failures and arguments.verbose:
            for failure in result.failures:
                print(f"    FAIL {failure}")
    print(f"{'TOTAL':10s} {total:3d}/{count:<3d}  ({total / count:.0%})")

    if not arguments.verbose and total != count:
        print("\nre-run with --verbose to list failures")
    return 0 if total == count else 1


if __name__ == "__main__":
    raise SystemExit(main())
