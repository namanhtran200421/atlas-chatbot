
import json
from pathlib import Path

from pydantic import ValidationError

from membership_rag.evaluation.models import (
    EvaluationCase,
)


def load_evaluation_cases(
    path: Path,
) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []

    seen_query_ids: set[str] = set()

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
                payload = json.loads(
                    line
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: "
                    "invalid JSON"
                ) from exc

            try:
                case = (
                    EvaluationCase.model_validate(
                        payload
                    )
                )
            except ValidationError as exc:
                raise ValueError(
                    f"{path}:{line_number}: "
                    f"{exc}"
                ) from exc

            if (
                case.query_id
                in seen_query_ids
            ):
                raise ValueError(
                    f"{path}:{line_number}: "
                    "duplicate query_id "
                    f"{case.query_id!r}"
                )

            seen_query_ids.add(
                case.query_id
            )

            cases.append(
                case
            )

    if not cases:
        raise ValueError(
            "evaluation dataset is empty"
        )

    return cases