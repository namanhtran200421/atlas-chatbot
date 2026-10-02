from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from membership_rag.chunking.models import NormalizedDocument
from membership_rag.corpus.adapter import record_to_document
from membership_rag.corpus.events import (
    merge_event_occurrences,
)
from membership_rag.corpus.quality import (
    should_index_document,
)

EXCLUDED_FILES = {
    "calendar-occurrences.jsonl",
    "website-sitemap.jsonl",
}


def read_jsonl(
    path: Path,
) -> Iterator[dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise TypeError(
                    f"Invalid JSON in {path} "
                    f"at line {line_number}"
                ) from error

            if not isinstance(record, dict):
                raise TypeError(
                    f"Expected JSON object in "
                    f"{path}:{line_number}"
                )

            yield record


def load_documents(
    input_dir: Path,
) -> list[NormalizedDocument]:
    documents: list[NormalizedDocument] = []

    event_path = (
        input_dir
        / "mec-event-definitions.jsonl"
    )

    occurrence_path = (
        input_dir
        / "calendar-occurrences.jsonl"
    )

    handled_files = {
        "mec-event-definitions.jsonl",
        "calendar-occurrences.jsonl",
        "website-sitemap.jsonl",
    }

    for path in sorted(
        input_dir.glob("*.jsonl")
    ):
        if path.name in handled_files:
            continue

        for record in read_jsonl(path):
            document = record_to_document(
                record
            )

            if (
                document is not None
                and should_index_document(
                    document
                )
            ):
                documents.append(
                    document
                )
    if event_path.exists():
        event_records = list(
            read_jsonl(event_path)
        )

        occurrence_records = (
            list(
                read_jsonl(
                    occurrence_path
                )
            )
            if occurrence_path.exists()
            else []
        )

        merged_events = (
            merge_event_occurrences(
                event_records,
                occurrence_records,
            )
        )

        for record in merged_events:
            document = record_to_document(
                record
            )

            if (
                document is not None
                and should_index_document(
                    document
                )
            ):
                documents.append(
                    document
                )

    return documents
