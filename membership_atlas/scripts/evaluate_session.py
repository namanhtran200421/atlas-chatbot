#!/usr/bin/env python3
"""Replay one long session and check what each turn is understood to be about.

The isolated cases in ``evaluate_conversation.py`` each start from a two-turn
history. Real sessions do not: the reader drifts between membership and
research, breaks for small talk, comes back to something raised fifteen turns
earlier, corrects themselves, and refers to things by position, by description
and by pronoun. Memory failures only show up at that length, so this replays a
scripted session turn by turn, accumulating history exactly as the browser
does, and asserts for every turn both how it is routed and what subject the
search ends up carrying.

The assistant's replies are scripted rather than generated, so this makes no
AWS calls and costs nothing to run.

    python scripts/evaluate_session.py [--verbose]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

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
    introduced_name,
    is_name_recall,
    is_simple_utility,
    is_vague_continuation,
)

CHAT = "chat"
ATTACHED = "attached"
FRESH = "fresh"
ASK = "ask"


@dataclass
class Turn:
    user: str
    assistant: str
    expect: str
    #: A word the search must carry. For an attached turn this is the subject
    #: it borrowed; for a fresh one, a word of its own.
    subject: str | None = None
    #: Something the reader said earlier that the reply has to be able to use.
    recalls: str | None = None


def classify(message: str, history: tuple[ConversationTurn, ...]) -> str:
    if is_simple_utility(message):
        return CHAT
    if is_vague_continuation(message) and not has_topic(history):
        return ASK
    if references_a_document(message) and referenced_item(message, history):
        return ATTACHED
    return ATTACHED if contextual_query(message, history) != message else FRESH


def searched(message: str, history: tuple[ConversationTurn, ...]) -> str:
    if references_a_document(message):
        item = referenced_item(message, history)
        if item:
            return item
    return contextual_query(message, history)


PLANS = (
    "Our organisation plans are Small (up to 300 employees) at $2,500 a year, "
    "Medium (301 to 1,500) at $5,500, Large (1,501 to 5,000) at $8,000 and "
    "Enterprise at $9,950."
)
PAPERS = (
    "Here are some of our research papers:\n"
    '1. "Hybrid work: Making it fit with your diversity, equity, and inclusion strategy" - on hybrid models.\n'
    '2. "Measuring Diversity of Artificial Intelligence Conferences" - on AI events.\n'
    '3. "LGBTQ+ Workplace Inclusion and the Great Resignation" - on retention.\n'
    '4. "The multicultural workplace: interactive acculturation and intergroup relations" - on acculturation.\n'
    '5. "Future office layouts for large organisations: workplace specialist perspective" - on office design.'
)

LGBTQ_REPLY = (
    'The paper "LGBTQ+ Workplace Inclusion and the Great Resignation" looks at '
    "how inclusion affects retention among LGBTQ+ employees."
)
HYBRID_REPLY = (
    'The paper "Hybrid work: Making it fit with your diversity, equity, and '
    'inclusion strategy" finds most employees prefer hybrid models.'
)
AI_REPLY = (
    'The paper "Measuring Diversity of Artificial Intelligence Conferences" '
    "examines representation at AI events."
)
OFFICE_REPLY = (
    'The paper "Future office layouts for large organisations: workplace '
    'specialist perspective" covers how office design is decided.'
)

SESSION: list[Turn] = [
    # --- opening small talk and an introduction -------------------------
    Turn("hiii", "Hey there! What's up?", CHAT),
    Turn("im Nam btw", "Lovely to meet you, Nam!", CHAT),
    Turn("hbu? how r u", "Doing well, thanks for asking!", CHAT),
    # --- a real need, stated in pieces ----------------------------------
    Turn(
        "im looking at membership for my organisation",
        "Happy to help! Our organisation plans scale by headcount.",
        FRESH,
        subject="membership",
    ),
    Turn(
        "we have about 400 employees",
        "With 400 employees you'd be on the Medium plan.",
        ATTACHED,
        subject="membership",
    ),
    # Names its own subject, so it is searched as asked; the 400 employees
    # still reach the model through the recent questions it is given.
    Turn("so which plan fits us?", PLANS, FRESH, subject="plan"),
    Turn("how much is that one?", PLANS, ATTACHED, subject="plan"),
    Turn("whats included?", PLANS, ATTACHED, subject="plan"),
    Turn("and the Large one?", PLANS, ATTACHED, subject="plan"),
    # --- small talk in the middle must not lose the thread --------------
    Turn("cool thanks!", "Anytime!", CHAT),
    Turn("haha nice", "Glad that helps!", CHAT),
    Turn(
        "does it include courses?",
        "Each plan includes online courses for nominated employees.",
        ATTACHED,
        subject="Large",
    ),
    # --- a deliberate topic switch --------------------------------------
    Turn(
        "different question - do you have research on diversity?",
        PAPERS,
        FRESH,
        subject="diversity",
    ),
    Turn("show me a few", PAPERS, ATTACHED, subject="research"),
    # --- picking out of the list, several ways --------------------------
    Turn("the third one", LGBTQ_REPLY, ATTACHED, subject="LGBTQ"),
    Turn("who wrote it?", LGBTQ_REPLY, ATTACHED, subject="LGBTQ"),
    Turn("what were the main findings?", LGBTQ_REPLY, ATTACHED, subject="LGBTQ"),
    Turn("ok what about number 1", HYBRID_REPLY, ATTACHED, subject="Hybrid work"),
    Turn("go deep into it", HYBRID_REPLY, ATTACHED, subject="Hybrid work"),
    Turn("the AI conferences one", AI_REPLY, ATTACHED, subject="Artificial Intelligence"),
    Turn("the office layout one please", OFFICE_REPLY, ATTACHED, subject="office layout"),
    # --- vague continuations mid-topic still continue --------------------
    Turn("tell me more", OFFICE_REPLY, ATTACHED, subject="office"),
    Turn("can you elaborate?", OFFICE_REPLY, ATTACHED, subject="office"),
    Turn("i like that, go deeper", OFFICE_REPLY, ATTACHED, subject="office"),
    # --- recall of what the reader themselves said ----------------------
    Turn("whats my name again?", "You're Nam!", CHAT, recalls="Nam"),
    Turn(
        "what did i say my org size was?",
        "You mentioned around 400 employees.",
        CHAT,
        recalls="400",
    ),
    Turn("what have we talked about?", "Membership plans and research.", CHAT),
    Turn("remind me what we discussed earlier", "Plans, then research.", CHAT),
    # --- returning to the earlier topic ---------------------------------
    Turn(
        "back to membership - how do i sign up?",
        "You can join from the membership page.",
        FRESH,
        subject="membership",
    ),
    Turn(
        "and what does the Enterprise plan cost?",
        PLANS,
        FRESH,
        subject="Enterprise",
    ),
    Turn("is that billed yearly?", PLANS, ATTACHED, subject="Enterprise"),
    # --- a correction ----------------------------------------------------
    Turn(
        "actually we're closer to 2000 people",
        "Then the Large plan would suit you.",
        ATTACHED,
        subject="plan",
    ),
    Turn("so which one now?", PLANS, ATTACHED, subject="plan"),
    # --- closing ---------------------------------------------------------
    Turn("perfect, thanks heaps!", "You're very welcome, Nam!", CHAT),
    Turn("what is 12 * 3?", "36!", CHAT),
    Turn("bye!", "See you, Nam!", CHAT),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()

    raw: list[dict[str, str]] = []
    failures: list[str] = []
    passed = 0

    for index, turn in enumerate(SESSION, start=1):
        history = normalise_history(raw)
        actual = classify(turn.user, history)
        search = searched(turn.user, history)

        problems = []
        if actual != turn.expect:
            problems.append(f"routed {actual}, want {turn.expect}")
        if turn.subject and turn.subject.casefold() not in search.casefold():
            problems.append(f"search lacks {turn.subject!r}")
        if turn.recalls:
            remembered = (
                introduced_name(
                    next(
                        (t.content for t in reversed(history) if t.role == "user"
                         and introduced_name(t.content)),
                        "",
                    )
                )
                if is_name_recall(turn.user)
                else None
            )
            available = "\n".join(t.content for t in history if t.role == "user")
            if turn.recalls not in available and turn.recalls != remembered:
                problems.append(f"cannot recall {turn.recalls!r}")

        if problems:
            failures.append(f"  {index:2d}. {turn.user!r}: " + "; ".join(problems))
            if arguments.verbose:
                failures.append(f"      search: {search[:96]!r}")
        else:
            passed += 1

        raw.append({"role": "user", "content": turn.user})
        raw.append({"role": "assistant", "content": turn.assistant})

    total = len(SESSION)
    print(f"session turns: {passed}/{total} ({passed / total:.0%})")
    for failure in failures:
        print(failure)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
