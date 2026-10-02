from __future__ import annotations

import json
from pathlib import Path

from membership_rag.chunking.models import Chunk


def write_bedrock_chunks(
    chunks: list[Chunk],
    output_dir: Path,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for chunk in chunks:
        filename = f"{chunk.chunk_id}.md"

        content_path = output_dir / filename

        metadata_path = output_dir / (
            f"{filename}.metadata.json"
        )

        content_path.write_text(
            chunk.text.rstrip() + "\n",
            encoding="utf-8",
        )

        metadata = {
            "metadataAttributes": {
                "document_id": chunk.document_id,
                "section_id": chunk.section_id,
                "parent_id": chunk.parent_id,
                "chunk_id": chunk.chunk_id,
                "chunk_hash": chunk.chunk_hash,
                "source_id": chunk.source_id,
                "source_type": chunk.source_type,
                "content_type": chunk.content_type,
                "access_class": chunk.access_class,
                "title": chunk.title,
            }
        }

        if chunk.url is not None:
            metadata["metadataAttributes"][
                "url"
            ] = chunk.url

        if chunk.modified_at is not None:
            metadata["metadataAttributes"][
                "modified_at"
            ] = chunk.modified_at

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )