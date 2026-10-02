"""Replay saved Bedrock responses without making AWS calls."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from membership_rag.bedrock.ranking import deduplicate_and_rank
from membership_rag.bedrock.retrieval import RetrievedChunk
from membership_rag.evaluation.evaluator import RetrievalEvaluator
from membership_rag.evaluation.models import EvaluationCase, QueryEvaluation
from membership_rag.guardrails import inspect_query


class _UnusedRetriever:
    def retrieve(
        self,
        query: str,
        *,
        allowed_access_classes: Sequence[str],
        number_of_results: int,
    ) -> list[RetrievedChunk]:
        raise RuntimeError("offline evaluation does not make retrieval calls")


def _read_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = row.get("query_id") or (row.get("case") or {}).get("query_id")
            if not isinstance(query_id, str) or not query_id:
                raise ValueError(f"{path}:{line_number}: missing query_id")
            if query_id in rows:
                raise ValueError(f"{path}:{line_number}: duplicate query_id {query_id!r}")
            rows[query_id] = row
    return rows


def _chunks_from_row(row: dict[str, Any]) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for raw_result in row.get("results", []):
        metadata = raw_result.get("metadata") or {}
        chunks.append(
            RetrievedChunk(
                text=raw_result.get("text") or "",
                score=raw_result.get("score"),
                metadata=metadata,
                document_id=raw_result.get("document_id")
                or metadata.get("document_id"),
                source_uri=raw_result.get("source_uri"),
            )
        )
    return chunks


def evaluate_saved_responses(
    cases: Sequence[EvaluationCase],
    response_path: Path,
) -> list[QueryEvaluation]:
    """Apply current guardrails/ranking to an immutable response fixture."""

    rows = _read_rows(response_path)
    evaluator = RetrievalEvaluator(_UnusedRetriever())
    results: list[QueryEvaluation] = []

    for case in cases:
        # A blocked query never reaches AWS, so it does not need a cached
        # response. This also keeps sensitive test prompts out of network logs.
        if not inspect_query(case.query).allowed:
            results.append(evaluator.evaluate_chunks(case, [], latency_ms=0.0))
            continue

        row = rows.get(case.query_id)
        if row is None:
            results.append(evaluator.error_result(case, 0.0, "missing cached response"))
            continue

        error = row.get("error")
        latency_ms = float(row.get("latency_ms") or 0.0)
        if error:
            results.append(evaluator.error_result(case, latency_ms, str(error)))
            continue

        chunks = deduplicate_and_rank(
            _chunks_from_row(row),
            case.query,
            limit=20,
        )
        results.append(evaluator.evaluate_chunks(case, chunks, latency_ms=latency_ms))

    return results
