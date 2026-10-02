from membership_rag.corpus.events import (
    merge_event_occurrences,
)


def test_existing_occurrences_heading_is_reused() -> None:
    events = [
        {
            "document_id": "event:42379",
            "source_id": 42379,
            "content_markdown": ("# Yule\n\nYule event information."),
        }
    ]

    occurrences = [
        {
            "document_id": "occurrence:42379",
            "source_id": 42379,
            "content_markdown": ("# Yule\n\n- 21 December 2026"),
        }
    ]

    merged = merge_event_occurrences(
        events,
        occurrences,
    )

    assert len(merged) == 1

    markdown = merged[0]["content_markdown"]

    assert "Yule event information." in markdown
    assert "## Calendar occurrences" in markdown
    assert "21 December 2026" in markdown


def test_occurrence_is_joined_to_correct_event() -> None:
    events = [
        {
            "document_id": "event:1",
            "source_id": 1,
            "content_markdown": "# Event One\n\nOne.",
        },
        {
            "document_id": "event:2",
            "source_id": 2,
            "content_markdown": "# Event Two\n\nTwo.",
        },
    ]

    occurrences = [
        {
            "source_id": 2,
            "content_markdown": ("# Event Two\n\n- 10 September 2026"),
        }
    ]

    merged = merge_event_occurrences(
        events,
        occurrences,
    )

    assert "10 September 2026" not in merged[0]["content_markdown"]

    assert "10 September 2026" in merged[1]["content_markdown"]


def test_event_without_occurrences_is_preserved() -> None:
    events = [
        {
            "document_id": "event:1",
            "source_id": 1,
            "content_markdown": ("# Event\n\nDescription."),
        }
    ]

    merged = merge_event_occurrences(
        events,
        [],
    )

    assert merged[0]["content_markdown"] == "# Event\n\nDescription."


def test_duplicate_occurrences_are_removed() -> None:
    events = [
        {
            "document_id": "event:1",
            "source_id": 1,
            "content_markdown": "# Event\n\nDescription.",
        }
    ]

    occurrence = {
        "source_id": 1,
        "content_markdown": ("# Event\n\n- 21 December 2026"),
    }

    merged = merge_event_occurrences(
        events,
        [
            occurrence,
            occurrence,
        ],
    )

    assert merged[0]["content_markdown"].count("21 December 2026") == 1


def test_occurrences_are_joined_by_source_id() -> None:
    events = [
        {
            "document_id": "event:42379",
            "source_id": 42379,
            "content_markdown": ("# Yule\n\nYule event information."),
        }
    ]

    occurrences = [
        {
            "document_id": "occurrence:42379",
            "source_id": 42379,
            "content_markdown": ("# Yule\n\n## Occurrences\n\n- 21 December 2026"),
        }
    ]

    merged = merge_event_occurrences(
        events,
        occurrences,
    )

    assert len(merged) == 1

    markdown = merged[0]["content_markdown"]

    assert "Yule event information." in markdown
    assert "## Occurrences" in markdown
    assert "21 December 2026" in markdown

    # We must not introduce the redundant wrapper heading.
    assert "## Calendar occurrences" not in markdown
