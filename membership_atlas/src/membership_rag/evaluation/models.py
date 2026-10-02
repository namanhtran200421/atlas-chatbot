from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class EvaluationCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra = "forbid")
    query_id: str = Field(min_length = 1)
    query: str = Field(min_length = 1)
    category: str = Field(min_length = 1)
    expected_document_ids: tuple[str, ...] = ()
    expected_source_types: tuple[str, ...] = ()
    expected_content_types: tuple[str, ...] = ()
    allowed_access_classes: tuple[str, ...] = Field(min_length=1)
    answerable: bool
    relevance_grades: dict[str, int] = Field(default_factory=dict)
    notes: str | None = None


    @model_validator(mode='after')
    def validate_score(self) -> Self:
        """
        grade convention:
        1 = partially relevant
        2 = strongly relevant
        3 = direct answer to query
        """
        if(self.answerable and not self.expected_document_ids):
            raise ValueError("answerable queries must have at least one expected document")

        if (not self.answerable and self.expected_document_ids):
            raise ValueError("unaswerable queries must not have expected document")

        if (not self.answerable and self.relevance_grades):
            raise ValueError("unanswerable queries must not have relevance grades")

        if self.relevance_grades: 
            expected = set(self.expected_document_ids)
            graded = set(self.relevance_grades)

            if graded != expected:
                raise ValueError("relevance grade must contain the expected document id")

            for grade in (self.relevance_grades.values()):
                if not 1 <= grade <= 3:
                    raise ValueError("relevance grade must be between 1 and 3")

        return self
    
class RankedRetrievalResult(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    rank: int = Field(
        ge=1,
    )

    document_id: str | None = None
    chunk_id: str | None = None
    score: float | None = None
    access_class: str | None = None
    source_type: str | None = None
    content_type: str | None = None
    title: str | None = None

class QueryEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True, extra = "forbid")
    query_id:str
    query:str
    category:str
    answerable:bool
    latency_ms: float = Field(ge=0)

    retrieved_results : tuple[
    RankedRetrievalResult,
    ...]
    hit_at_k: dict[int, float | None]
    recall_at_k: dict[int, float | None]
    reciprocal_rank: float | None

    ndcg_at_k: dict[
        int,
        float | None,
    ]

    unauthorized_result_count: int = Field(
        ge=0,
    )
    metadata_errors: tuple[str, ...]

    error: str | None = None






    

    