"""Resolving a reference back to something the assistant already listed."""

import pytest

from membership_rag.conversation import (
    normalise_history,
    referenced_item,
    references_a_document,
)
from membership_rag.reference import presented_items, resolve_reference

LISTING = '''Here are a few more research papers from our Membership Atlas:

1. "Sexual Identity in the Workplace: Reasons for (not) Coming out" - Explores disclosure.
2. "Gender, workplace preferences and firm performance: Looking through the glass door" - Investigates gender.
3. "The multicultural workplace: interactive acculturation and intergroup relations" - Examines discordance.
4. "Between the corporation and the closet: Ethically researching LGBTQ+ identities in the workplace" - Discusses ethics.
5. "Future office layouts for large organisations: workplace specialist and design firms' perspective" - Provides insights.'''

ITEMS = presented_items(LISTING)


def _history():
    return normalise_history(
        [
            {"role": "user", "content": "any more research?"},
            {"role": "assistant", "content": LISTING},
        ]
    )


def test_titles_are_extracted_without_numbering_quotes_or_gloss() -> None:
    assert len(ITEMS) == 5
    assert ITEMS[0] == "Sexual Identity in the Workplace: Reasons for (not) Coming out"
    assert ITEMS[3].startswith("Between the corporation and the closet")
    assert all(not item.startswith(('"', "1", "-")) for item in ITEMS)
    assert all("Explores" not in item for item in ITEMS)


@pytest.mark.parametrize(
    ("query", "position"),
    [
        ("Let's go with the fourth one", 4),
        ("the second one", 2),
        ("tell me about number 3", 3),
        ("the last one", 5),
        ("the first one", 1),
        ("#2", 2),
        ("option 5", 5),
        # Named by what it was about rather than where it sat.
        ("the LGBTQ one", 4),
        ("lets do the glass door one", 2),
        ("the multicultural one", 3),
        ("the office layout one", 5),
        ("tell me about the gender one", 2),
    ],
)
def test_a_reader_can_pick_an_item_any_of_the_usual_ways(
    query: str, position: int
) -> None:
    """"The fourth one" used to retrieve a Fourth Industrial Revolution paper."""

    assert references_a_document(query) is True
    assert referenced_item(query, _history()) == ITEMS[position - 1]


@pytest.mark.parametrize(
    "query",
    [
        "how much is membership?",
        "What events are coming up?",
        "What research does Atlas have on AI?",
        "what plans are there?",
        "tell me about the Individual plan",
    ],
)
def test_a_question_of_its_own_is_not_a_pick(query: str) -> None:
    assert references_a_document(query) is False
    assert referenced_item(query, _history()) is None


def test_an_ambiguous_reference_resolves_to_nothing_rather_than_a_guess() -> None:
    """Five candidates and no way to choose: better to fall back than guess."""

    assert referenced_item("what does this paper say", _history()) is None


def test_a_single_suggestion_needs_no_disambiguation() -> None:
    assert resolve_reference("what does that paper say", ("Hybrid work",)) == (
        "Hybrid work"
    )


def test_a_position_past_the_end_is_not_invented() -> None:
    assert resolve_reference("the ninth one", ITEMS) is None


def test_quoted_titles_in_prose_are_found_when_there_is_no_list() -> None:
    prose = (
        'We hold "Hybrid work: Making it fit" and '
        '"Measuring Diversity of Artificial Intelligence Conferences" among others.'
    )

    assert len(presented_items(prose)) == 2


def test_a_long_listing_survives_the_history_cap() -> None:
    """Truncating the answer lost the items a reader then picked by position."""

    history = _history()

    assert not history[-1].content.endswith("…")
    assert referenced_item("the last one", history) == ITEMS[-1]
