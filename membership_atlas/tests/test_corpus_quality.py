from dataclasses import replace

from membership_rag.chunking.models import (
    NormalizedDocument,
)
from membership_rag.corpus.quality import (
    should_index_document,
)


def make_page(
    markdown: str,
) -> NormalizedDocument:
    return NormalizedDocument(
        document_id="page:1",
        source_id="1",
        source_type="pages",
        content_type="page",
        access_class="member_restricted",
        title="Test",
        content_markdown=markdown,
    )


def test_placeholder_page_is_rejected() -> None:
    document = make_page("# Courses\n\n…")

    assert not should_index_document(document)


def test_shortcode_only_page_is_rejected() -> None:
    document = make_page("# Membership Invoice\n\n[pmpro_invoice]")

    assert not should_index_document(document)


def test_real_content_page_is_kept() -> None:
    document = make_page(
        """# Accessibility

Cultural accessibility is important for
organisations serving diverse communities.
Businesses should consider communication,
physical access, language requirements,
inclusive policies, employee participation,
community consultation, and accessible
digital services when designing programs.
"""
    )

    assert should_index_document(document)


def test_signed_in_account_page_is_not_indexed() -> None:
    document = make_page(
        "# Profile\n\nWelcome, Example Person. Your account is active and "
        "contains membership details that should not be searchable."
    )
    document = replace(document, url="https://membership.example.com/profile/")
    assert not should_index_document(document)


def test_short_non_page_document_is_not_filtered() -> None:
    document = NormalizedDocument(
        document_id="membership_plan:1",
        source_id="1",
        source_type="membership_plans",
        content_type="membership_plan",
        access_class="member_restricted",
        title="Free Tour",
        content_markdown=("# Free Tour\n\nFree membership plan."),
    )

    assert should_index_document(document)
