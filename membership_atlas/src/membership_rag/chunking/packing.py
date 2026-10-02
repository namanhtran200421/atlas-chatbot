from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9])"
)

LIST_ITEM_RE = re.compile(
    r"^\s*(?:[-*+]|\d+[.)])\s+"
)


@dataclass(frozen=True, slots=True)
class SemanticUnit:
    text: str
    separator: str


class TokenPacker:
    def __init__(
        self,
        *,
        tokenizer: str = "cl100k_base",
    ) -> None:
        self.encoding = tiktoken.get_encoding(
            tokenizer
        )

    def token_count(self, text: str) -> int:
        return len(
            self.encoding.encode(
                text,
                disallowed_special=(),
            )
        )

    def split_and_pack(
        self,
        *,
        prefix: str,
        body: str,
        max_tokens: int,
        overlap_units: int,
    ) -> list[str]:
        units = self._to_units(
            prefix=prefix,
            body=body,
            max_tokens=max_tokens,
        )

        if not units:
            text = prefix.strip()

            return [text] if text else []

        chunks: list[str] = []
        current: list[SemanticUnit] = []

        for unit in units:
            candidate = self._render(
                prefix,
                current + [unit],
            )

            if self.token_count(candidate) <= max_tokens:
                current.append(unit)
                continue

            if current:
                chunks.append(
                    self._render(
                        prefix,
                        current,
                    )
                )

            overlap = (
                current[-overlap_units:]
                if overlap_units
                else []
            )

            candidate = self._render(
                prefix,
                overlap + [unit],
            )

            if self.token_count(candidate) > max_tokens:
                overlap = []

            current = overlap + [unit]

        if current:
            chunks.append(
                self._render(
                    prefix,
                    current,
                )
            )

        return chunks

    def _to_units(
        self,
        *,
        prefix: str,
        body: str,
        max_tokens: int,
    ) -> list[SemanticUnit]:
        blocks = re.split(
            r"\n\s*\n",
            body.strip(),
        )

        units: list[SemanticUnit] = []

        for block in blocks:
            block = block.strip()

            if not block:
                continue

            candidate = (
                f"{prefix}\n\n{block}"
                if prefix
                else block
            )

            if self.token_count(candidate) <= max_tokens:
                units.append(
                    SemanticUnit(
                        text=block,
                        separator="\n\n",
                    )
                )
                continue

            lines = [
                line.strip()
                for line in block.splitlines()
                if line.strip()
            ]

            if (
                lines
                and all(
                    LIST_ITEM_RE.match(line)
                    for line in lines
                )
            ):
                for line in lines:
                    units.extend(
                        self._split_oversized_unit(
                            prefix=prefix,
                            text=line,
                            separator="\n",
                            max_tokens=max_tokens,
                        )
                    )

                continue

            sentences = [
                value.strip()
                for value in SENTENCE_RE.split(block)
                if value.strip()
            ]

            for sentence in sentences:
                units.extend(
                    self._split_oversized_unit(
                        prefix=prefix,
                        text=sentence,
                        separator=" ",
                        max_tokens=max_tokens,
                    )
                )

        return units

    def _split_oversized_unit(
        self,
        *,
        prefix: str,
        text: str,
        separator: str,
        max_tokens: int,
    ) -> list[SemanticUnit]:
        candidate = (
            f"{prefix}\n\n{text}"
            if prefix
            else text
        )

        if self.token_count(candidate) <= max_tokens:
            return [
                SemanticUnit(
                    text=text,
                    separator=separator,
                )
            ]

        prefix_tokens = self.token_count(prefix)

        available = (
            max_tokens
            - prefix_tokens
            - 8
        )

        if available <= 0:
            raise ValueError(
                "Heading context exceeds chunk token budget"
            )

        encoded = self.encoding.encode(
            text,
            disallowed_special=(),
        )

        units: list[SemanticUnit] = []

        for start in range(
            0,
            len(encoded),
            available,
        ):
            value = self.encoding.decode(
                encoded[
                    start:start + available
                ]
            ).strip()

            if value:
                units.append(
                    SemanticUnit(
                        text=value,
                        separator=separator,
                    )
                )

        return units

    @staticmethod
    def _render(
        prefix: str,
        units: list[SemanticUnit],
    ) -> str:
        if not units:
            return prefix.strip()

        body = units[0].text

        for unit in units[1:]:
            body += (
                unit.separator
                + unit.text
            )

        if not prefix:
            return body.strip()

        return (
            f"{prefix.strip()}\n\n"
            f"{body.strip()}"
        )