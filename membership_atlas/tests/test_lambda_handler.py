import json

import pytest

from membership_rag.bedrock.generation import GeneratedAnswer, PublicCitation
from membership_rag.bedrock.retrieval import RetrievedChunk
from membership_rag.bedrock.routing import TurnRoute
from membership_rag.conversation import contextual_query
from membership_rag.lambda_handler import _authenticated_corpus_access, handler
from membership_rag.scope import is_simple_utility
from membership_rag.session_state import (
    InvalidConversationState,
    sign_state,
    verify_state,
)


@pytest.fixture(autouse=True)
def fake_semantic_router(monkeypatch) -> None:
    from membership_rag import lambda_handler

    class FakeRouter:
        def route(self, query, history=(), *, trusted_history=False):
            if is_simple_utility(query):
                return TurnRoute("chat", "")
            return TurnRoute("atlas", contextual_query(query, history))

    monkeypatch.setattr(lambda_handler, "_get_router", lambda: FakeRouter())


def test_access_classes_follow_verified_gateway_group_claims() -> None:
    member = {
        "requestContext": {
            "authorizer": {
                "jwt": {"claims": {"sub": "member-1", "cognito:groups": "[members]"}}
            }
        }
    }
    assert _authenticated_corpus_access({}, "/chat") == ("public",)
    assert _authenticated_corpus_access(member, "/public-chat") == ("public",)
    assert _authenticated_corpus_access(member, "/chat") == (
        "public", "member_restricted"
    )
    assert _authenticated_corpus_access(
        {"body": json.dumps({"cognito:groups": "members"})}, "/chat"
    ) == ("public",)


def test_member_conversation_state_cannot_be_replayed_as_public() -> None:
    secret = b"s" * 32
    state = sign_state((), secret, scope="member:member-1", now=100)
    with pytest.raises(InvalidConversationState):
        verify_state(state, secret, scope="public", now=100)


def test_named_membership_request_is_refused_before_retrieval() -> None:
    response = handler(
        {
            "httpMethod": "POST",
            "path": "/chat",
            "body": json.dumps(
                {"query": "Is Example Person a member of Membership Atlas?"}
            ),
        },
        None,
    )

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert "can't verify or share" in body["answer"]
    assert body["citations"] == []


def test_simple_math_skips_retrieval(monkeypatch) -> None:
    from membership_rag import lambda_handler

    class FakeGenerator:
        def generate(self, query, chunks, history=(), **kwargs):
            assert query == "What is 1+1?"
            assert chunks == []
            return GeneratedAnswer(text="2", citations=())

    monkeypatch.setattr(lambda_handler, "_get_generator", lambda: FakeGenerator())
    monkeypatch.setattr(
        lambda_handler,
        "_get_retriever",
        lambda: (_ for _ in ()).throw(AssertionError("retrieval should be skipped")),
    )
    response = handler(
        {"httpMethod": "POST", "path": "/chat", "body": json.dumps({"query": "What is 1+1?"})},
        None,
    )

    assert response["statusCode"] == 200
    assert json.loads(response["body"])["answer"] == "2"


def test_introduction_skips_retrieval_and_gets_a_social_reply(monkeypatch) -> None:
    from membership_rag import lambda_handler

    class FakeGenerator:
        def generate(self, query, chunks, history=(), **kwargs):
            assert query == "hi my name is Nam"
            assert chunks == []
            return GeneratedAnswer(text="Hi Nam! Lovely to meet you!", citations=())

    monkeypatch.setattr(lambda_handler, "_get_generator", lambda: FakeGenerator())
    monkeypatch.setattr(
        lambda_handler,
        "_get_retriever",
        lambda: (_ for _ in ()).throw(AssertionError("retrieval should be skipped")),
    )

    response = handler(
        {
            "httpMethod": "POST",
            "path": "/chat",
            "body": json.dumps({"query": "hi my name is Nam"}),
        },
        None,
    )

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["answer"] == "Hi Nam! Lovely to meet you!"
    assert body["citations"] == []


def test_history_reaches_retrieval_and_generation(monkeypatch) -> None:
    from membership_rag import lambda_handler

    seen: dict[str, object] = {}

    class FakeRetriever:
        def retrieve(self, query, *, allowed_access_classes, number_of_results, history=()):
            seen.setdefault("retrieval_queries", []).append(query)
            seen["retrieval_history"] = history
            return []

    class FakeGenerator:
        def generate(self, query, chunks, history=(), **kwargs):
            seen["generation_history"] = history
            return GeneratedAnswer(text="Four plans.", citations=())

    monkeypatch.setattr(lambda_handler, "_get_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(lambda_handler, "_get_generator", lambda: FakeGenerator())

    response = handler(
        {
            "httpMethod": "POST",
            "path": "/chat",
            "body": json.dumps(
                {
                    "query": "How much is it?",
                    "history": [
                        {"role": "user", "content": "What membership plans are available?"},
                        {"role": "assistant", "content": "There are four plans."},
                    ],
                }
            ),
        },
        None,
    )

    assert response["statusCode"] == 200
    assert [turn.content for turn in seen["generation_history"]] == [
        "What membership plans are available?",
        "There are four plans.",
    ]
    assert seen["retrieval_history"] == ()
    # The router passes a standalone follow-up query to retrieval.
    assert "What membership plans are available? How much is it?" in seen[
        "retrieval_queries"
    ]


def test_history_of_the_wrong_shape_is_rejected() -> None:
    response = handler(
        {
            "httpMethod": "POST",
            "path": "/chat",
            "body": json.dumps({"query": "How much is it?", "history": "earlier turns"}),
        },
        None,
    )

    assert response["statusCode"] == 400
    assert json.loads(response["body"])["error"] == "invalid_history"


def test_history_can_be_turned_off_by_configuration(monkeypatch) -> None:
    from membership_rag import lambda_handler

    monkeypatch.setenv("MEMBERSHIP_RAG_HISTORY_TURNS", "0")
    seen: dict[str, object] = {}

    class FakeRetriever:
        def retrieve(self, query, *, allowed_access_classes, number_of_results, history=()):
            seen["history"] = history
            return []

    class FakeGenerator:
        def generate(self, query, chunks, history=(), **kwargs):
            return GeneratedAnswer(text="Four plans.", citations=())

    monkeypatch.setattr(lambda_handler, "_get_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(lambda_handler, "_get_generator", lambda: FakeGenerator())

    handler(
        {
            "httpMethod": "POST",
            "path": "/chat",
            "body": json.dumps(
                {
                    "query": "How much is it?",
                    "history": [{"role": "user", "content": "What plans are available?"}],
                }
            ),
        },
        None,
    )

    assert seen["history"] == ()


def test_uncited_atlas_claim_is_retried_before_display(monkeypatch) -> None:
    from membership_rag import lambda_handler

    chunk = RetrievedChunk(
        text="The Medium plan costs $5,500 per year.",
        score=0.9,
        metadata={"title": "Membership plans"},
        document_id="plan",
        source_uri=None,
    )
    calls: list[bool] = []

    class FakeRouter:
        def route(self, query, history=(), *, trusted_history=False):
            return TurnRoute("atlas", query)

    class FakeRetriever:
        def retrieve(self, query, *, allowed_access_classes, number_of_results):
            return [chunk]

    class FakeGenerator:
        def generate(self, query, chunks, history=(), **kwargs):
            retry = kwargs.get("grounding_retry", False)
            calls.append(retry)
            return (
                GeneratedAnswer(
                    text="The Medium plan costs $5,500 per year.",
                    citations=(PublicCitation("Membership plans", None),),
                )
                if retry
                else GeneratedAnswer(text="The Medium plan is free.", citations=())
            )

    class FakeVerifier:
        def safe_uncited(self, query, answer):
            return False

        def supported(self, query, answer, chunks):
            return True

    monkeypatch.setattr(lambda_handler, "_get_router", lambda: FakeRouter())
    monkeypatch.setattr(lambda_handler, "_get_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(lambda_handler, "_get_generator", lambda: FakeGenerator())
    monkeypatch.setattr(lambda_handler, "_get_verifier", lambda: FakeVerifier())

    response = handler(
        {
            "httpMethod": "POST",
            "path": "/chat",
            "body": json.dumps({"query": "What does the Medium plan cost?"}),
        },
        None,
    )

    body = json.loads(response["body"])
    assert response["statusCode"] == 200
    assert body["answer"] == "The Medium plan costs $5,500 per year."
    assert body["citations"] == [{"title": "Membership plans", "url": None}]
    assert calls == [False, True]
