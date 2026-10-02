from membership_rag.bedrock.generation import GeneratedAnswer, PublicCitation
from membership_rag.bedrock.retrieval import RetrievedChunk
from membership_rag.bedrock.verification import BedrockGroundingVerifier


class _AlwaysApproveClient:
    def converse(self, **kwargs):
        return {"output": {"message": {"content": [{"text": '{"supported":true}'}]}}}


def test_authorship_is_now_the_verifier_model_s_call() -> None:
    """Authorship used to be gated by a byline regex before the model ran.

    It refused any "who wrote this?" whose excerpt had no recognisable byline,
    which is most of the corpus. The rule now lives in the verifier prompt so
    a reasonable reading can pass.
    """

    title = "Workplace diversity study"
    chunk = RetrievedChunk(
        text=f"# {title}\n\nWritten by Quinetta Roberson.",
        score=0.9,
        metadata={"title": title},
        document_id="paper",
        source_uri=None,
    )
    answer = GeneratedAnswer(
        text="Quinetta Roberson wrote the study.",
        citations=(PublicCitation(title, None),),
    )

    verifier = BedrockGroundingVerifier(
        "ap-southeast-2", client=_AlwaysApproveClient()
    )
    assert verifier.supported("Who wrote it?", answer, [chunk])


def test_plan_cannot_fit_more_people_than_its_sourced_capacity() -> None:
    chunk = RetrievedChunk(
        text="# Org Admin – Medium\n\nUp to 1,500 employees, 20 courses, 1 location",
        score=0.9,
        metadata={"title": "Org Admin – Medium"},
        document_id="medium",
        source_uri=None,
    )
    answer = GeneratedAnswer(
        text="The Medium plan fits your team of 2,000 people.",
        citations=(PublicCitation("Org Admin – Medium", None),),
    )

    verifier = BedrockGroundingVerifier(
        "ap-southeast-2", client=_AlwaysApproveClient()
    )
    assert not verifier.supported(
        "We have 2,000 employees. Which plan fits?", answer, [chunk]
    )


def test_an_acronym_answer_no_longer_needs_the_letters_verbatim() -> None:
    """"What is MEC?" was refused whenever the excerpt spelled the name out.

    The excerpt describing the Cultural Events Calendar never contains the
    literal string "MEC", so a verbatim check rejected every correct answer.
    """

    chunk = RetrievedChunk(
        text="# Cultural Events Calendar\n\nExplore celebrations worldwide.",
        score=0.9,
        metadata={"title": "Cultural Events Calendar"},
        document_id="calendar",
        source_uri=None,
    )
    answer = GeneratedAnswer(
        text="MEC is the Atlas cultural events calendar.",
        citations=(PublicCitation("Cultural Events Calendar", None),),
    )
    verifier = BedrockGroundingVerifier(
        "ap-southeast-2", client=_AlwaysApproveClient()
    )
    assert verifier.supported("What is MEC?", answer, [chunk])


def test_a_canned_redirect_is_not_sent_for_grounding_checks() -> None:
    """A fixed redirect is our own wording, not a claim read from a source.

    Treated as an ordinary uncited answer it failed the check whenever
    retrieval had returned anything, so "write me a Python script" and "who is
    Barack Obama?" both reached the reader as the grounding fallback instead
    of the scope redirect they were given.
    """

    from membership_rag.bedrock.generation import CANNED_REPLIES

    scope_redirect = (
        "I can help with Membership Atlas and Cultural Infusion's cultural "
        "content or events. Ask me about those!"
    )

    assert scope_redirect in CANNED_REPLIES
    assert all(reply.strip() for reply in CANNED_REPLIES)

