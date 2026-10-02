from pathlib import Path

from membership_rag.chunking import (
    NormalizedDocument,
    SourceAwareChunker,
)
from membership_rag.chunking.policies import resolve_policy
from membership_rag.config import (
    load_rag_config,
)


def make_chunker() -> SourceAwareChunker:
    config = load_rag_config(Path("configs/membership_v2.yaml"))

    return SourceAwareChunker.from_rag_config(config)


def make_document(
    markdown: str,
    *,
    content_type: str = "info_hub",
    document_id: str = "info_hub:1",
) -> NormalizedDocument:
    return NormalizedDocument(
        document_id=document_id,
        source_id="1",
        source_type="info_hub",
        content_type=content_type,
        access_class="member_restricted",
        title="Test Document",
        content_markdown=markdown,
    )


def test_empty_document_returns_no_chunks() -> None:
    chunker = make_chunker()

    document = make_document("")

    assert chunker.chunk_document(document) == []


def test_heading_hierarchy_is_preserved() -> None:
    document = make_document(
        """
# Accessibility

## Strategies

### Physical accessibility

Businesses should provide accessible facilities.
"""
    )

    chunker = make_chunker()

    chunks = chunker.chunk_document(document)

    assert chunks

    physical_chunk = next(
        chunk for chunk in chunks if chunk.heading == "Physical accessibility"
    )

    assert "# Accessibility" in physical_chunk.text
    assert "## Strategies" in physical_chunk.text
    assert "### Physical accessibility" in physical_chunk.text


def test_long_article_is_split() -> None:
    markdown = "# Test\n\n" + " ".join(
        f"Sentence {i} contains useful information." for i in range(400)
    )

    chunker = make_chunker()

    chunks = chunker.chunk_document(make_document(markdown))

    assert len(chunks) > 1

    assert all(chunker.packer.token_count(chunk.text) <= 512 for chunk in chunks)


def test_same_input_produces_same_ids() -> None:
    document = make_document("# Test\n\nSome useful content.")

    chunker = make_chunker()

    first = chunker.chunk_document(document)

    second = chunker.chunk_document(document)

    assert first == second


def test_atomic_document_remains_whole() -> None:
    document = NormalizedDocument(
        document_id="membership_plans:1",
        source_id="1",
        source_type="membership_plans",
        content_type="membership_plan",
        access_class="configuration",
        title="Free Tour",
        content_markdown="""
# Free Tour

Initial payment: 0

Billing amount: 0

Expiration period: Week
""",
    )

    chunker = make_chunker()

    chunks = chunker.chunk_document(document)

    assert len(chunks) == 1


def test_research_policy_allows_larger_chunks() -> None:
    document = make_document(
        """
# Paper

## 1. INTRODUCTION

This is an introduction.

## 2. LITERATURE REVIEW

This is the literature review.
""",
        content_type="research",
        document_id="research:1",
    )

    chunker = make_chunker()

    chunks = chunker.chunk_document(document)

    assert chunks

    assert all(chunker.packer.token_count(chunk.text) <= 768 for chunk in chunks)


def test_public_and_restricted_duplicates_are_not_collapsed() -> None:
    content = "# Test\n\nSame information."

    public = NormalizedDocument(
        document_id="faq:1",
        source_id="1",
        source_type="faq",
        content_type="faq",
        access_class="public",
        title="Test",
        content_markdown=content,
    )

    restricted = NormalizedDocument(
        document_id="info_hub:1",
        source_id="1",
        source_type="info_hub",
        content_type="info_hub",
        access_class="member_restricted",
        title="Test",
        content_markdown=content,
    )

    chunker = make_chunker()

    chunks = chunker.chunk_documents([public, restricted])

    assert len(chunks) == 2


def test_exact_duplicate_chunks_are_suppressed() -> None:
    first = make_document(
        "# Test\n\nSame content.",
        document_id="info_hub:1",
    )

    second = make_document(
        "# Test\n\nSame content.",
        document_id="info_hub:2",
    )

    chunker = make_chunker()

    chunks = chunker.chunk_documents([first, second])

    assert len(chunks) == 1


def test_chunker_uses_configured_policy_values() -> None:
    config = load_rag_config(Path("configs/membership_v2.yaml"))

    assert config.preprocessing is not None

    chunker = SourceAwareChunker.from_rag_config(config)

    article_policy = resolve_policy(
        "info_hub",
        chunker.config,
    )

    configured_article = config.preprocessing.chunking.policies.article

    assert article_policy.max_tokens == configured_article.max_tokens

    assert article_policy.overlap_units == configured_article.overlap_units


def test_heading_without_body_does_not_create_chunk() -> None:
    document = make_document(
        """# Accessibility

## Strategies

### Physical accessibility

Useful physical accessibility information.
"""
    )

    chunker = make_chunker()

    chunks = chunker.chunk_document(document)

    texts = [chunk.text for chunk in chunks]

    assert not any(text.strip() == "# Accessibility" for text in texts)

    assert not any(text.strip().endswith("## Strategies") for text in texts)

    assert any(
        "Physical accessibility" in text
        and "Useful physical accessibility information" in text
        for text in texts
    )


def test_event_remains_whole_when_under_limit() -> None:
    document = NormalizedDocument(
        document_id="mec:1",
        source_id="1",
        source_type="mec_event_definitions",
        content_type="calendar_event_definition",
        access_class="member_restricted",
        title="Yule",
        content_markdown="""# Yule

Yule celebrates the Winter Solstice.

## Occurrences

- 21 December 2027
- 21 December 2028
""",
    )

    chunker = make_chunker()

    chunks = chunker.chunk_document(document)

    assert len(chunks) == 1

    assert "Winter Solstice" in chunks[0].text
    assert "21 December 2027" in chunks[0].text
