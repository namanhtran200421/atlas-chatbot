from __future__ import annotations

from membership_rag.chunking.ids import (
    canonicalize_text,
    stable_hash,
    stable_id,
)
from membership_rag.chunking.markdown import (
    parse_markdown_sections,
)
from membership_rag.chunking.models import (
    Chunk,
    NormalizedDocument,
)
from membership_rag.chunking.packing import (
    TokenPacker,
)
from membership_rag.chunking.policies import (
    ChunkPolicy,
    resolve_policy,
)
from membership_rag.config.models import (
    LocalChunkingConfig,
    RAGConfig,
)


class SourceAwareChunker:
    def __init__(
        self,
        config: LocalChunkingConfig,
    ) -> None:
        self.config = config

        self.packer = TokenPacker(
            tokenizer=config.tokenizer
        )

    @classmethod
    def from_rag_config(
        cls,
        config: RAGConfig,
    ) -> SourceAwareChunker:
        """
        Build the chunker directly from the complete
        immutable RAG configuration.
        """

        if config.preprocessing is None:
            raise ValueError(
                "RAG config does not define preprocessing"
            )

        if config.chunking.strategy != "NONE":
            raise ValueError(
                "Pre-chunked documents require "
                "Bedrock chunking strategy NONE"
            )

        return cls(
            config.preprocessing.chunking
        )

    def chunk_documents(
        self,
        documents: list[NormalizedDocument],
    ) -> list[Chunk]:
        chunks: list[Chunk] = []

        # Include access class in the deduplication key.
        #
        # Identical public and member-restricted chunks
        # must remain distinct so access provenance is
        # never lost.
        seen: set[tuple[str, str]] = set()

        for document in documents:
            document_chunks = self.chunk_document(
                document
            )

            for chunk in document_chunks:
                dedup_key = (
                    chunk.chunk_hash,
                    chunk.access_class,
                )

                if dedup_key in seen:
                    continue

                seen.add(dedup_key)
                chunks.append(chunk)

        return chunks

    def chunk_document(
        self,
        document: NormalizedDocument,
    ) -> list[Chunk]:
        if not document.content_markdown.strip():
            return []

        policy = resolve_policy(
            document.content_type,
            self.config,
        )

        if policy.mode == "whole_document":
            return self._chunk_whole_document(
                document,
                policy,
            )

        return self._chunk_sections(
            document,
            policy,
        )

    def _chunk_whole_document(
        self,
        document: NormalizedDocument,
        policy: ChunkPolicy,
    ) -> list[Chunk]:
        text = canonicalize_text(
            document.content_markdown
        )

        if (
            self.packer.token_count(text)
            <= policy.max_tokens
        ):
            return [
                self._build_chunk(
                    document=document,
                    section_id=document.document_id,
                    parent_id=document.document_id,
                    heading=document.title,
                    text=text,
                )
            ]

        # Safety fallback:
        # if a supposedly atomic document unexpectedly
        # exceeds its token budget, split it by sections.
        return self._chunk_sections(
            document,
            policy,
        )

    def _chunk_sections(
        self,
        document: NormalizedDocument,
        policy: ChunkPolicy,
    ) -> list[Chunk]:
        sections = parse_markdown_sections(
            document
        )

        chunks: list[Chunk] = []

        for section in sections:
            if not section.body.strip():
                continue
            
            prefix = self._heading_context(
                section.heading_path
            )

            texts = self.packer.split_and_pack(
                prefix=prefix,
                body=section.body,
                max_tokens=policy.max_tokens,
                overlap_units=policy.overlap_units,
            )

            for text in texts:
                chunks.append(
                    self._build_chunk(
                        document=document,
                        section_id=section.section_id,
                        parent_id=section.parent_id,
                        heading=section.heading,
                        text=text,
                    )
                )

        return chunks

    @staticmethod
    def _heading_context(
        heading_path: tuple[str, ...],
    ) -> str:
        lines: list[str] = []

        for index, heading in enumerate(
            heading_path,
            start=1,
        ):
            level = min(index, 6)

            lines.append(
                f"{'#' * level} {heading}"
            )

        return "\n".join(lines)

    @staticmethod
    def _build_chunk(
        *,
        document: NormalizedDocument,
        section_id: str,
        parent_id: str,
        heading: str | None,
        text: str,
    ) -> Chunk:
        canonical = canonicalize_text(
            text
        )

        chunk_hash = stable_hash(
            canonical
        )

        chunk_id = stable_id(
            "chk",
            document.document_id,
            section_id,
            chunk_hash,
        )

        return Chunk(
            document_id=document.document_id,
            section_id=section_id,
            parent_id=parent_id,

            chunk_id=chunk_id,
            chunk_hash=chunk_hash,

            source_id=document.source_id,
            source_type=document.source_type,
            content_type=document.content_type,
            access_class=document.access_class,

            title=document.title,
            heading=heading,
            text=canonical,

            url=document.url,
            modified_at=document.modified_at,
        )