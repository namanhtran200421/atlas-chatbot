"""The long scripted session, run as a regression test.

``scripts/evaluate_session.py`` is the readable form and prints a per-turn
report. This keeps it honest: a change that breaks how a turn is understood
mid-conversation fails the suite rather than waiting to be noticed in a chat.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import evaluate_session

from membership_rag.conversation import ConversationTurn, normalise_history


def _history_before(index: int) -> tuple[ConversationTurn, ...]:
    raw: list[dict[str, str]] = []
    for turn in evaluate_session.SESSION[:index]:
        raw.append({"role": "user", "content": turn.user})
        raw.append({"role": "assistant", "content": turn.assistant})
    return normalise_history(raw)


@pytest.mark.parametrize(
    ("index", "turn"),
    list(enumerate(evaluate_session.SESSION)),
    ids=[f"{i:02d}-{t.user[:32]}" for i, t in enumerate(evaluate_session.SESSION)],
)
def test_every_turn_of_a_long_session_is_understood(
    index: int, turn: evaluate_session.Turn
) -> None:
    history = _history_before(index)

    assert evaluate_session.classify(turn.user, history) == turn.expect

    if turn.subject:
        search = evaluate_session.searched(turn.user, history)
        assert turn.subject.casefold() in search.casefold(), search

    if turn.recalls:
        said = "\n".join(t.content for t in history if t.role == "user")
        assert turn.recalls in said
