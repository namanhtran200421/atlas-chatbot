from membership_rag.bedrock.ranking import deduplicate_and_rank
from membership_rag.bedrock.retrieval import RetrievedChunk


def make_chunk(
    document_id: str,
    score: float,
    *,
    title: str,
    content_type: str,
) -> RetrievedChunk:
    return RetrievedChunk(
        text="Example",
        score=score,
        metadata={
            "document_id": document_id,
            "chunk_id": f"chunk-{document_id}",
            "access_class": "public",
            "content_type": content_type,
            "title": title,
        },
        document_id=document_id,
        source_uri=None,
    )


def test_duplicate_document_chunks_are_removed() -> None:
    chunks = [
        make_chunk("doc-1", 0.9, title="First", content_type="page"),
        make_chunk("doc-1", 0.8, title="First", content_type="page"),
    ]
    ranked = deduplicate_and_rank(chunks, "first", limit=20)
    assert [chunk.document_id for chunk in ranked] == ["doc-1"]


def test_title_match_receives_small_boost() -> None:
    chunks = [
        make_chunk("generic", 0.91, title="Overview", content_type="page"),
        make_chunk("target", 0.89, title="Accessibility", content_type="page"),
    ]
    ranked = deduplicate_and_rank(chunks, "Tell me about Accessibility", limit=20)
    assert ranked[0].document_id == "target"


def test_soft_routing_does_not_remove_other_content_types() -> None:
    chunks = [
        make_chunk(
            "plan", 0.9, title="Individual plan", content_type="membership_plan"
        ),
        make_chunk("page", 0.8, title="Membership levels", content_type="page"),
    ]
    ranked = deduplicate_and_rank(chunks, "membership plan options", limit=20)
    assert {chunk.document_id for chunk in ranked} == {"plan", "page"}
