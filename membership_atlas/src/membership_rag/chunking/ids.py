import hashlib
import re

_WHITESPACE_RE = re.compile(r"[ \t]+")


def canonicalize_text(text: str) -> str:
    """
    Normalize text deterministically before hashing.

    Preserves paragraph boundaries while removing
    unnecessary whitespace.
    """

    lines: list[str] = []

    for line in text.strip().splitlines():
        cleaned = _WHITESPACE_RE.sub(
            " ",
            line.strip(),
        )

        lines.append(cleaned)

    output: list[str] = []
    previous_blank = False

    for line in lines:
        is_blank = not line

        if is_blank and previous_blank:
            continue

        output.append(line)
        previous_blank = is_blank

    return "\n".join(output).strip()


def stable_hash(*parts: str) -> str:
    """
    Produce a deterministic SHA-256 hash.
    """

    canonical = "\x1f".join(parts)

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def stable_id(
    prefix: str,
    *parts: str,
) -> str:
    """
    Produce a deterministic readable identifier.
    """

    return (
        f"{prefix}_"
        f"{stable_hash(*parts)[:20]}"
    )