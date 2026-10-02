"""Authenticated, browser-held conversation state.

Lambda keeps no session database. The browser returns this signed state on the
next request, and the signature prevents a caller from forging Oriana's prior
answers. It never grants access to the knowledge base.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import time
from collections.abc import Sequence

from membership_rag.conversation import ConversationTurn, normalise_history

_VERSION = 2
_MAX_TOKEN_LENGTH = 160_000
_MAX_AGE_SECONDS = 24 * 60 * 60


class InvalidConversationState(ValueError):
    """The client supplied a malformed, expired, or altered state."""


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def sign_state(
    turns: Sequence[ConversationTurn],
    secret: bytes,
    *,
    scope: str = "public",
    now: int | None = None,
) -> str:
    """Sign recent dialogue for storage in the current browser page."""

    if len(secret) < 32:
        raise ValueError("conversation signing secret must be at least 32 bytes")
    payload = json.dumps(
        {
            "v": _VERSION,
            "scope": scope,
            "iat": int(time.time()) if now is None else now,
            "turns": [
                {"role": turn.role, "content": turn.content}
                for turn in turns[-100:]
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    signature = hmac.new(secret, payload, hashlib.sha256).digest()
    token = f"{_b64encode(payload)}.{_b64encode(signature)}"
    if len(token) > _MAX_TOKEN_LENGTH:
        raise ValueError("conversation state is too large")
    return token


def verify_state(
    token: object,
    secret: bytes,
    *,
    scope: str = "public",
    now: int | None = None,
) -> tuple[ConversationTurn, ...]:
    """Verify a state before any prior assistant text reaches the model."""

    if not isinstance(token, str) or not token or len(token) > _MAX_TOKEN_LENGTH:
        raise InvalidConversationState("invalid conversation state")
    try:
        encoded_payload, encoded_signature = token.split(".")
        payload = _b64decode(encoded_payload)
        signature = _b64decode(encoded_signature)
        expected = hmac.new(secret, payload, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            raise InvalidConversationState("invalid conversation state")
        data = json.loads(payload)
        timestamp = data["iat"]
        current_time = int(time.time()) if now is None else now
        if (
            data.get("v") != _VERSION
            or data.get("scope") != scope
            or not isinstance(timestamp, int)
            or isinstance(timestamp, bool)
            or timestamp > current_time + 60
            or current_time - timestamp > _MAX_AGE_SECONDS
            or not isinstance(data.get("turns"), list)
            or len(data["turns"]) > 100
        ):
            raise InvalidConversationState("invalid conversation state")
        turns = normalise_history(data["turns"])
        if len(turns) != len(data["turns"]):
            raise InvalidConversationState("invalid conversation state")
        return turns
    except (ValueError, KeyError, TypeError, UnicodeDecodeError, binascii.Error) as exc:
        raise InvalidConversationState("invalid conversation state") from exc
