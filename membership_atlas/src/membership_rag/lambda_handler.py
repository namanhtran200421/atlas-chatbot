"""AWS Lambda HTTP adapter for the Membership Atlas grounded-answer backend.

Lambda handler: ``membership_rag.lambda_handler.handler``

Required environment variables:

* ``MEMBERSHIP_RAG_KB_ID``: Amazon Bedrock Knowledge Base ID.
* ``AWS_REGION`` or ``MEMBERSHIP_RAG_REGION``: AWS region for Bedrock.

Authorization is enforced before Lambda by the API Gateway ``/chat`` route.
Every authenticated Membership Atlas user can search the complete corpus; the
legacy access-class metadata is retained only for compatibility with the
already-ingested knowledge base.

Optional runtime variables:

* ``MEMBERSHIP_RAG_ALLOWED_ORIGINS``: comma-separated CORS origin allowlist.
* ``MEMBERSHIP_RAG_MAX_RESULTS``: API result limit (default 20, maximum 100).
* ``MEMBERSHIP_RAG_CANDIDATE_RESULTS``: retrieval candidates before local
  deduplication/reranking (default 20, maximum 100).
* ``MEMBERSHIP_RAG_MODEL_ID``: answer model (default Amazon Nova Pro).
* ``MEMBERSHIP_RAG_MAX_ANSWER_TOKENS``: generated answer limit (default 600).
* ``MEMBERSHIP_RAG_CONTEXT_CHARACTERS``: grounding limit (default 12000).
* ``MEMBERSHIP_RAG_HISTORY_TURNS``: earlier turns accepted from the client for
  session memory (default 100, 0 disables it).
* ``MEMBERSHIP_RAG_SESSION_SECRET``: signing key for browser-held conversation
  state (at least 32 characters).

Conversation memory lives in the caller's page, not here: the optional
``history`` field carries the recent turns it still holds, so a browser reload
starts a new conversation. It is untrusted input and is never authorization or
grounding.

The request body is intentionally not an authorization source.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

from membership_rag.bedrock.generation import (
    CANNED_REPLIES,
    AnswerGenerationError,
    BedrockAnswerGenerator,
    GeneratedAnswer,
    PublicCitation,
)
from membership_rag.bedrock.retrieval import BedrockRetriever, RetrievedChunk
from membership_rag.bedrock.routing import BedrockConversationRouter, TurnRoute
from membership_rag.bedrock.verification import BedrockGroundingVerifier
from membership_rag.conversation import (
    MAX_HISTORY_TURNS,
    contextual_query,
    continues_previous_subject,
    last_answer,
    normalise_history,
    referenced_item,
)
from membership_rag.guardrails import (
    AccessControlError,
    ResultMetadataError,
    inspect_query,
    validate_access_classes,
)
from membership_rag.scope import is_simple_utility, narrow_to_referenced_document
from membership_rag.session_state import (
    InvalidConversationState,
    sign_state,
    verify_state,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

_ACCESS_CLASS_ORDER = ("public", "member_restricted", "configuration")
_SAFE_SCOPE_REDIRECT = (
    "I'm Oriana!!! I'd LOVE to help you explore Membership Atlas or our "
    "cultural events! What sparks your curiosity?"
)
_SAFE_PRIVACY_REDIRECT = (
    "I can't verify or share someone's membership account or contact details here. "
    "Please use the official account or support channel."
)
_RETRIEVER: BedrockRetriever | None = None
_RETRIEVER_CONFIG: tuple[str, str, int] | None = None
_GENERATOR: BedrockAnswerGenerator | None = None
_GENERATOR_CONFIG: tuple[str, str, int, int] | None = None
_VERIFIER: BedrockGroundingVerifier | None = None
_VERIFIER_CONFIG: tuple[str, str] | None = None
_ROUTER: BedrockConversationRouter | None = None
_ROUTER_CONFIG: tuple[str, str] | None = None
# Shown when an answer failed its grounding check. The question was usually
# perfectly clear, so asking what they meant reads as though they were at
# fault. Say it is about accuracy and offer a way to narrow the search.
_GROUNDING_CLARIFICATION = (
    "I'm not finding enough in Atlas to answer that properly. Could you tell me a "
    "bit more about what you're after, or try naming the plan, paper or event?"
)


class ServiceConfigurationError(RuntimeError):
    """Raised when required Lambda environment configuration is missing."""


def _broad_membership_request(query: str) -> bool:
    return bool(
        re.search(r"\b(?:plans?|membership\s+options)\b", query, re.IGNORECASE)
        and re.search(
            r"\b(?:available|options|types|compare|difference|prices)\b",
            query,
            re.IGNORECASE,
        )
        and not re.search(r"\b(?:my|our|me|us|it|that|this)\b", query, re.IGNORECASE)
    )


def _prioritize_eligible_plans(
    chunks: list[RetrievedChunk], employees: int
) -> list[RetrievedChunk]:
    def rank(chunk: RetrievedChunk) -> tuple[int, int]:
        title = str(chunk.metadata.get("title", ""))
        if re.search(r"\bOrg Admin\s*[–-]", title, re.IGNORECASE):
            capacity = re.search(
                r"\bup\s+to\s+(\d[\d,]*)\s+employees\b",
                chunk.text[:500],
                re.IGNORECASE,
            )
            if capacity:
                maximum = int(capacity.group(1).replace(",", ""))
                return (0, maximum) if maximum >= employees else (2, maximum)
        if title == "Landing page":
            return (1, 0)
        return (3, 0)

    return sorted(chunks, key=rank)


def _get_router() -> BedrockConversationRouter:
    global _ROUTER, _ROUTER_CONFIG
    region = (
        os.environ.get("MEMBERSHIP_RAG_REGION", "").strip()
        or os.environ.get("AWS_REGION", "").strip()
    )
    model_id = os.environ.get(
        "MEMBERSHIP_RAG_ROUTER_MODEL_ID", "amazon.nova-lite-v1:0"
    ).strip()
    if not region or not model_id:
        raise ServiceConfigurationError("conversation routing is not configured")
    config = (region, model_id)
    if _ROUTER is None or _ROUTER_CONFIG != config:
        _ROUTER = BedrockConversationRouter(region, model_id=model_id)
        _ROUTER_CONFIG = config
    return _ROUTER


def _csv_environment(name: str) -> tuple[str, ...]:
    return tuple(
        value.strip()
        for value in os.environ.get(name, "").split(",")
        if value.strip()
    )


def _maximum_results() -> int:
    raw = os.environ.get("MEMBERSHIP_RAG_MAX_RESULTS", "20")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ServiceConfigurationError(
            "MEMBERSHIP_RAG_MAX_RESULTS must be an integer"
        ) from exc
    if not 1 <= value <= 100:
        raise ServiceConfigurationError(
            "MEMBERSHIP_RAG_MAX_RESULTS must be between 1 and 100"
        )
    return value


def _integer_environment(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ServiceConfigurationError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ServiceConfigurationError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _history_turn_limit() -> int:
    return _integer_environment(
        "MEMBERSHIP_RAG_HISTORY_TURNS",
        MAX_HISTORY_TURNS,
        minimum=0,
        maximum=100,
    )


def _session_secret() -> bytes | None:
    value = os.environ.get("MEMBERSHIP_RAG_SESSION_SECRET", "")
    if not value:
        return None
    secret = value.encode("utf-8")
    if len(secret) < 32:
        raise ServiceConfigurationError("MEMBERSHIP_RAG_SESSION_SECRET is too short")
    return secret


def _get_verifier() -> BedrockGroundingVerifier:
    global _VERIFIER, _VERIFIER_CONFIG

    region = (
        os.environ.get("MEMBERSHIP_RAG_REGION", "").strip()
        or os.environ.get("AWS_REGION", "").strip()
    )
    model_id = os.environ.get(
        "MEMBERSHIP_RAG_VERIFIER_MODEL_ID", "amazon.nova-pro-v1:0"
    ).strip()
    if not region or not model_id:
        raise ServiceConfigurationError("grounding verification is not configured")
    current_config = (region, model_id)
    if _VERIFIER is None or _VERIFIER_CONFIG != current_config:
        _VERIFIER = BedrockGroundingVerifier(region, model_id=model_id)
        _VERIFIER_CONFIG = current_config
    return _VERIFIER


def _get_retriever() -> BedrockRetriever:
    global _RETRIEVER, _RETRIEVER_CONFIG

    knowledge_base_id = os.environ.get("MEMBERSHIP_RAG_KB_ID", "").strip()
    region = (
        os.environ.get("MEMBERSHIP_RAG_REGION", "").strip()
        or os.environ.get("AWS_REGION", "").strip()
    )
    if not knowledge_base_id:
        raise ServiceConfigurationError("MEMBERSHIP_RAG_KB_ID is not configured")
    if not region:
        raise ServiceConfigurationError("AWS region is not configured")

    candidate_pool_size = _integer_environment(
        "MEMBERSHIP_RAG_CANDIDATE_RESULTS",
        20,
        minimum=1,
        maximum=100,
    )

    current_config = (knowledge_base_id, region, candidate_pool_size)
    if _RETRIEVER is None or _RETRIEVER_CONFIG != current_config:
        _RETRIEVER = BedrockRetriever(
            knowledge_base_id=knowledge_base_id,
            region_name=region,
            include_public_catalogue=True,
            candidate_pool_size=candidate_pool_size,
        )
        _RETRIEVER_CONFIG = current_config
    return _RETRIEVER


def _get_generator() -> BedrockAnswerGenerator:
    global _GENERATOR, _GENERATOR_CONFIG

    region = (
        os.environ.get("MEMBERSHIP_RAG_REGION", "").strip()
        or os.environ.get("AWS_REGION", "").strip()
    )
    if not region:
        raise ServiceConfigurationError("AWS region is not configured")
    model_id = os.environ.get(
        "MEMBERSHIP_RAG_MODEL_ID", "amazon.nova-pro-v1:0"
    ).strip()
    if not model_id:
        raise ServiceConfigurationError("MEMBERSHIP_RAG_MODEL_ID is empty")
    max_output_tokens = _integer_environment(
        "MEMBERSHIP_RAG_MAX_ANSWER_TOKENS",
        600,
        minimum=1,
        maximum=1_000,
    )
    max_context_characters = _integer_environment(
        "MEMBERSHIP_RAG_CONTEXT_CHARACTERS",
        12_000,
        minimum=1_000,
        maximum=50_000,
    )

    current_config = (
        region,
        model_id,
        max_output_tokens,
        max_context_characters,
    )
    if _GENERATOR is None or _GENERATOR_CONFIG != current_config:
        _GENERATOR = BedrockAnswerGenerator(
            region_name=region,
            model_id=model_id,
            max_output_tokens=max_output_tokens,
            max_context_characters=max_context_characters,
        )
        _GENERATOR_CONFIG = current_config
    return _GENERATOR


def _request_context(event: Mapping[str, Any]) -> Mapping[str, Any]:
    context = event.get("requestContext")
    return context if isinstance(context, Mapping) else {}


def _request_id(event: Mapping[str, Any], context: Any) -> str:
    aws_request_id = getattr(context, "aws_request_id", None)
    if isinstance(aws_request_id, str) and aws_request_id:
        return aws_request_id
    value = _request_context(event).get("requestId")
    return value if isinstance(value, str) and value else "unknown"


def _http_method(event: Mapping[str, Any]) -> str:
    request_context = _request_context(event)
    http = request_context.get("http")
    if isinstance(http, Mapping):
        method = http.get("method")
        if isinstance(method, str) and method:
            return method.upper()
    method = event.get("httpMethod")
    if isinstance(method, str) and method:
        return method.upper()
    return "POST"


def _http_path(event: Mapping[str, Any]) -> str:
    path = event.get("rawPath") or event.get("path") or "/retrieve"
    if not isinstance(path, str):
        return "/retrieve"
    return path.rstrip("/") or "/"


def _request_origin(event: Mapping[str, Any]) -> str | None:
    headers = event.get("headers")
    if not isinstance(headers, Mapping):
        return None
    for key, value in headers.items():
        if str(key).lower() == "origin" and isinstance(value, str):
            return value
    return None


def _response_headers(event: Mapping[str, Any]) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    }
    origin = _request_origin(event)
    allowed_origins = _csv_environment("MEMBERSHIP_RAG_ALLOWED_ORIGINS")
    if origin and ("*" in allowed_origins or origin in allowed_origins):
        headers["Access-Control-Allow-Origin"] = (
            "*" if "*" in allowed_origins else origin
        )
        headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        headers["Vary"] = "Origin"
    return headers


def _response(
    event: Mapping[str, Any],
    status_code: int,
    body: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": _response_headers(event),
        "body": "" if body is None else json.dumps(body, separators=(",", ":")),
    }


def _error_response(
    event: Mapping[str, Any],
    status_code: int,
    code: str,
    message: str,
    request_id: str,
) -> dict[str, Any]:
    return _response(
        event,
        status_code,
        {
            "error": code,
            "message": message,
            "request_id": request_id,
        },
    )


def _parse_body(event: Mapping[str, Any]) -> Mapping[str, Any]:
    body = event.get("body")
    if body is None:
        # Support direct Lambda invocation with the same request fields.
        if "requestContext" not in event and "httpMethod" not in event:
            return event
        raise ValueError("request body is required")
    if isinstance(body, Mapping):
        return body
    if not isinstance(body, str):
        raise TypeError("request body must be a JSON object")
    if event.get("isBase64Encoded") is True:
        try:
            body = base64.b64decode(body, validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("request body is not valid base64-encoded UTF-8") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError("request body must contain valid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise TypeError("request body must be a JSON object")
    return parsed


def _authenticated_corpus_access() -> tuple[str, ...]:
    """Return all legacy classes for an authenticated Membership Atlas user.

    API Gateway owns authentication for ``/chat``. The corpus still contains
    the original access-class metadata, so include every known class until a
    future ingestion removes that obsolete distinction.
    """

    return validate_access_classes(_ACCESS_CLASS_ORDER)


def _serialize_citation(citation: PublicCitation) -> dict[str, Any]:
    return {
        "title": citation.title,
        "url": citation.url,
    }


def _log(
    *,
    request_id: str,
    status_code: int,
    elapsed_ms: float,
    result_count: int = 0,
    access_classes: Sequence[str] = (),
    error_category: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    history_turns: int = 0,
) -> None:
    record: dict[str, Any] = {
        "event": "membership_rag_retrieval",
        "request_id": request_id,
        "status_code": status_code,
        "elapsed_ms": round(elapsed_ms, 2),
        "result_count": result_count,
        "access_classes": list(access_classes),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "history_turns": history_turns,
    }
    if error_category:
        record["error_category"] = error_category
    LOGGER.info(json.dumps(record, separators=(",", ":")))


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Handle API Gateway HTTP API, REST API, Function URL, or direct invokes."""

    event = event or {}
    started = time.perf_counter()
    request_id = _request_id(event, context)
    access_classes: tuple[str, ...] = ()

    method = _http_method(event)
    path = _http_path(event)

    if method == "OPTIONS":
        return _response(event, 204, None)

    if method == "GET" and path in ("/", "/health"):
        response = _response(
            event,
            200,
            {
                "service": "membership-rag",
                "status": "ok",
                "request_id": request_id,
            },
        )
        _log(
            request_id=request_id,
            status_code=200,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
        return response

    if method != "POST":
        return _error_response(
            event, 405, "method_not_allowed", "Use POST /chat.", request_id
        )
    if path not in ("/", "/chat", "/retrieve"):
        return _error_response(
            event, 404, "not_found", "The requested endpoint does not exist.", request_id
        )

    try:
        body = _parse_body(event)
        query_value = body.get("query")
        if not isinstance(query_value, str) or not query_value.strip():
            return _error_response(
                event, 400, "invalid_query", "query must be a non-empty string.", request_id
            )
        query = query_value.strip()

        requested_results = body.get("number_of_results", 5)
        if isinstance(requested_results, bool) or not isinstance(requested_results, int):
            return _error_response(
                event,
                400,
                "invalid_number_of_results",
                "number_of_results must be an integer.",
                request_id,
            )
        maximum_results = _maximum_results()
        if not 1 <= requested_results <= maximum_results:
            return _error_response(
                event,
                422,
                "invalid_number_of_results",
                f"number_of_results must be between 1 and {maximum_results}.",
                request_id,
            )

        raw_history = body.get("history")
        if raw_history is not None and not isinstance(raw_history, list):
            return _error_response(
                event,
                400,
                "invalid_history",
                "history must be an array of {role, content} turns.",
                request_id,
            )
        secret = _session_secret()
        state_token = body.get("conversation_state")
        trusted_history = state_token is not None
        if trusted_history:
            if secret is None:
                raise ServiceConfigurationError("conversation state signing is not configured")
            try:
                history = verify_state(state_token, secret)
            except InvalidConversationState:
                return _error_response(
                    event,
                    400,
                    "invalid_conversation_state",
                    "Start a new chat and try again.",
                    request_id,
                )
        else:
            history = normalise_history(
                raw_history,
                maximum_turns=_history_turn_limit(),
            )

        decision = inspect_query(query)
        if not decision.allowed:
            if decision.reason in {"model_internals_request", "private_membership_request"}:
                response = _response(
                    event,
                    200,
                    {
                        "answer": (
                            _SAFE_PRIVACY_REDIRECT
                            if decision.reason == "private_membership_request"
                            else _SAFE_SCOPE_REDIRECT
                        ),
                        "citations": [],
                        "request_id": request_id,
                    },
                )
                _log(
                    request_id=request_id,
                    status_code=200,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    error_category=decision.reason,
                )
                return response
            if decision.reason == "unsafe_control_character":
                status_code = 400
            elif decision.reason in {"query_too_long", "spam_query"}:
                status_code = 422
            else:
                status_code = 403
            _log(
                request_id=request_id,
                status_code=status_code,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                error_category=decision.reason or "guardrail_block",
            )
            return _error_response(
                event,
                status_code,
                "request_not_allowed",
                "The request cannot be processed.",
                request_id,
            )

        access_classes = _authenticated_corpus_access()
        route = (
            TurnRoute("chat", "")
            if is_simple_utility(query)
            else TurnRoute("atlas", query)
            if _broad_membership_request(query)
            else _get_router().route(query, history, trusted_history=trusted_history)
        )
        results = []
        if route.mode == "atlas":
            search_query = route.search_query or contextual_query(query, history)
            retriever = _get_retriever()
            results = retriever.retrieve(
                search_query,
                allowed_access_classes=access_classes,
                number_of_results=requested_results,
            )
            if (
                re.search(r"\b(?:plans?|membership\s+options)\b", search_query, re.IGNORECASE)
                and re.search(
                    r"\b(?:available|options|types|compare|difference|which|what)\b",
                    search_query,
                    re.IGNORECASE,
                )
                and not re.search(r"\b(?:fit|suit|recommend)\b", search_query, re.IGNORECASE)
                and not re.search(r"\b\d[\d,]*\s+(?:employees|people|staff)\b", search_query, re.IGNORECASE)
            ):
                individual = retriever.retrieve(
                    "individual membership plan",
                    allowed_access_classes=access_classes,
                    number_of_results=min(2, requested_results),
                )
                organisations = retriever.retrieve(
                    "organisation membership plans pricing employee sizes",
                    allowed_access_classes=access_classes,
                    number_of_results=min(3, requested_results),
                )
                results = [*individual, *organisations, *results]
            if (
                re.search(r"\b(?:join|register|sign\s*up)\b", query, re.IGNORECASE)
                and "membership atlas" in search_query.casefold()
            ):
                registration = retriever.retrieve(
                    "Membership Atlas membership page join register",
                    allowed_access_classes=access_classes,
                    number_of_results=min(3, requested_results),
                )
                results = [*registration, *results]
            overview = []
            if (
                re.search(r"\b\d[\d,]*\s+(?:employees|people|staff)\b", search_query, re.IGNORECASE)
                and re.search(r"\b(?:plans?|membership)\b", search_query, re.IGNORECASE)
                and not re.search(r"\b(?:join|register|sign\s*up)\b", search_query, re.IGNORECASE)
            ):
                overview = retriever.retrieve(
                    "organisation membership plans pricing employee sizes",
                    allowed_access_classes=access_classes,
                    number_of_results=requested_results,
                )
                headcounts = re.findall(
                    r"\b(\d[\d,]*)\s+(?:employees|people|staff)\b",
                    search_query,
                    re.IGNORECASE,
                )
                if headcounts:
                    overview = _prioritize_eligible_plans(
                        overview, int(headcounts[-1].replace(",", ""))
                    )
                results = [*overview, *results]
            # A short entity search keeps a specific plan or document from
            # being buried by broad results for a recommendation question.
            entity_query = route.entity_query
            if len(entity_query.split()) < 2:
                entity_query = ""
            requested_numbers = {
                value.replace(",", "")
                for value in re.findall(r"\b\d[\d,]*\b", search_query)
            }
            entity_numbers = {
                value.replace(",", "")
                for value in re.findall(r"\b\d[\d,]*\b", entity_query)
            }
            if requested_numbers and not requested_numbers.issubset(entity_numbers):
                entity_query = ""
            if entity_query and entity_query.casefold() != search_query.casefold():
                focused = retriever.retrieve(
                    entity_query,
                    allowed_access_classes=access_classes,
                    number_of_results=min(3, requested_results),
                )
                results = ([*results, *focused] if overview else [*focused, *results])
            seen: set[tuple[str | None, str | None, str | None]] = set()
            unique = []
            for chunk in results:
                key = (
                    chunk.document_id,
                    chunk.source_uri,
                    chunk.metadata.get("title"),
                )
                if key not in seen:
                    seen.add(key)
                    unique.append(chunk)
            results = unique
        if results and continues_previous_subject(query):
            # "Explain this paper" means the one already under discussion.
            # Searching with the previous answer's wording alone drifted to
            # other papers on the same theme, so keep the document whose
            # title that answer actually covers.
            item = referenced_item(query, history)
            exact = (
                [
                    chunk for chunk in results
                    if str(chunk.metadata.get("title", "")).casefold() == item.casefold()
                ]
                if item else []
            )
            results = exact or narrow_to_referenced_document(
                results, item or last_answer(history)
            )
        generated = _get_generator().generate(
            query,
            results,
            history,
            trusted_history=trusted_history,
            conversation_mode=route.mode,
            resolved_question=route.search_query if route.mode == "atlas" else None,
        )
        grounding_question = (
            f"{query}\nResolved context: {route.search_query}"
            if route.mode == "atlas" and route.search_query != query
            else query
        )
        verified = (
            True
            if generated.text in CANNED_REPLIES
            else _get_verifier().supported(grounding_question, generated, results)
            if generated.citations
            else not (route.mode == "atlas" and results)
        )
        if not verified:
            revised = _get_generator().generate(
                query,
                results,
                history,
                trusted_history=trusted_history,
                conversation_mode=route.mode,
                grounding_retry=True,
                resolved_question=route.search_query if route.mode == "atlas" else None,
            )
            revised_verified = (
                _get_verifier().supported(grounding_question, revised, results)
                if revised.citations
                else _get_verifier().safe_uncited(grounding_question, revised)
            )
            generated = (
                revised if revised_verified else GeneratedAnswer(
                    text=_GROUNDING_CLARIFICATION,
                    citations=(),
                    input_tokens=generated.input_tokens + revised.input_tokens,
                    output_tokens=generated.output_tokens + revised.output_tokens,
                )
            )
        new_state = None
        if secret is not None:
            updated_history = normalise_history(
                [
                    *({"role": turn.role, "content": turn.content} for turn in history),
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": generated.text},
                ],
                maximum_turns=_history_turn_limit(),
            )
            new_state = sign_state(updated_history, secret)
        response = _response(
            event,
            200,
            {
                "answer": generated.text,
                "citations": [
                    _serialize_citation(citation)
                    for citation in generated.citations
                ],
                "request_id": request_id,
                **({"conversation_state": new_state} if new_state else {}),
            },
        )
        _log(
            request_id=request_id,
            status_code=200,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            result_count=len(results),
            access_classes=access_classes,
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
            history_turns=len(history),
        )
        return response
    except (TypeError, ValueError) as exc:
        _log(
            request_id=request_id,
            status_code=400,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            access_classes=access_classes,
            error_category="invalid_request",
        )
        return _error_response(event, 400, "invalid_request", str(exc), request_id)
    except ServiceConfigurationError:
        LOGGER.exception("Membership RAG Lambda is not configured")
        _log(
            request_id=request_id,
            status_code=500,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            access_classes=access_classes,
            error_category="service_configuration",
        )
        return _error_response(
            event,
            500,
            "service_unavailable",
            "The retrieval service is not configured.",
            request_id,
        )
    except (AccessControlError, ResultMetadataError):
        LOGGER.exception("Bedrock returned an unsafe retrieval result")
        _log(
            request_id=request_id,
            status_code=502,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            access_classes=access_classes,
            error_category="retrieval_safety",
        )
        return _error_response(
            event,
            502,
            "retrieval_safety_error",
            "The retrieval response could not be safely processed.",
            request_id,
        )
    except AnswerGenerationError:
        LOGGER.exception("Bedrock returned an unusable generation response")
        _log(
            request_id=request_id,
            status_code=502,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            access_classes=access_classes,
            error_category="generation_response",
        )
        return _error_response(
            event,
            502,
            "answer_unavailable",
            "The answer service returned an unusable response.",
            request_id,
        )
    except (BotoCoreError, ClientError):
        LOGGER.exception("Bedrock retrieval or generation failed")
        _log(
            request_id=request_id,
            status_code=502,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            access_classes=access_classes,
            error_category="bedrock",
        )
        return _error_response(
            event,
            502,
            "answer_unavailable",
            "The answer service is temporarily unavailable.",
            request_id,
        )
    except Exception:
        LOGGER.exception("Unexpected Membership RAG Lambda failure")
        _log(
            request_id=request_id,
            status_code=500,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            access_classes=access_classes,
            error_category="unexpected",
        )
        return _error_response(
            event,
            500,
            "internal_error",
            "An unexpected error occurred.",
            request_id,
        )


# Conventional alias for deployment systems that expect ``lambda_handler``.
lambda_handler = handler
