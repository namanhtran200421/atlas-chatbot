from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )


class AWSConfig(FrozenModel):
    region: str


class CorpusConfig(FrozenModel):
    version: str
    s3_uri: str


class FixedSizeChunkingConfig(FrozenModel):
    version: str
    strategy: Literal["FIXED_SIZE"]
    max_tokens: int = Field(ge=1, le=8192)
    overlap_percentage: int = Field(ge=1, le=99)


class SemanticChunkingConfig(FrozenModel):
    version: str
    strategy: Literal["SEMANTIC"]
    max_tokens: int = Field(ge=1, le=8192)
    buffer_size: int = Field(ge=0, le=1)
    breakpoint_percentile_threshold: int = Field(ge=50, le=99)


class HierarchicalChunkingConfig(FrozenModel):
    version: str
    strategy: Literal["HIERARCHICAL"]
    parent_max_tokens: int = Field(ge=1, le=8192)
    child_max_tokens: int = Field(ge=1, le=8192)
    overlap_tokens: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_levels(self) -> HierarchicalChunkingConfig:
        if self.child_max_tokens >= self.parent_max_tokens:
            raise ValueError(
                "child_max_tokens must be smaller than parent_max_tokens"
            )

        return self


class NoChunkingConfig(FrozenModel):
    version: str
    strategy: Literal["NONE"]


ChunkingConfig = Annotated[
    FixedSizeChunkingConfig
    | SemanticChunkingConfig
    | HierarchicalChunkingConfig
    | NoChunkingConfig,
    Field(discriminator="strategy"),
]

class EmbeddingConfig(FrozenModel):
    model_id: str
    dimensions: int = Field(ge=1, le=4096)
    data_type: Literal["FLOAT32", "BINARY"]


class RetrievalConfig(FrozenModel):
    search_type: Literal["SEMANTIC", "HYBRID"]
    candidate_k: int = Field(ge=1, le=100)


class RerankerConfig(FrozenModel):
    enabled: bool
    model_arn: str | None = None
    final_k: int | None = Field(default=None, ge=1, le=100)

    @model_validator(mode="after")
    def validate_reranker(self) -> RerankerConfig:
        if self.enabled:
            if self.model_arn is None:
                raise ValueError(
                    "model_arn is required when reranking is enabled"
                )

            if self.final_k is None:
                raise ValueError(
                    "final_k is required when reranking is enabled"
                )

        return self


class GenerationConfig(FrozenModel):
    enabled: bool
    model_arn: str | None = None
    context_token_budget: int | None = Field(default=None, gt=0)
    prompt_version: str | None = None

    @model_validator(mode="after")
    def validate_generation(self) -> GenerationConfig:
        if self.enabled:
            if self.model_arn is None:
                raise ValueError(
                    "model_arn is required when generation is enabled"
                )

            if self.context_token_budget is None:
                raise ValueError(
                    "context_token_budget is required when generation is enabled"
                )

            if self.prompt_version is None:
                raise ValueError(
                    "prompt_version is required when generation is enabled"
                )

        return self

class ChunkPolicyConfig(FrozenModel):
    max_tokens: int = Field(gt=0)
    overlap_units: int = Field(ge=0)


class ChunkPoliciesConfig(FrozenModel):
    article: ChunkPolicyConfig
    research: ChunkPolicyConfig
    faq: ChunkPolicyConfig
    event: ChunkPolicyConfig
    atomic: ChunkPolicyConfig


class LocalChunkingConfig(FrozenModel):
    version: str
    tokenizer: str
    policies: ChunkPoliciesConfig


class PreprocessingConfig(FrozenModel):
    chunking: LocalChunkingConfig

    
class RAGConfig(FrozenModel):
    config_version: str

    aws: AWSConfig
    corpus: CorpusConfig

    chunking: ChunkingConfig
    metadata_schema_version: str

    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    reranker: RerankerConfig
    generation: GenerationConfig

    preprocessing: PreprocessingConfig | None = None

    evaluation_dataset_version: str | None = None

    @model_validator(mode="after")
    def validate_pipeline(self) -> RAGConfig:
        if (
            self.reranker.enabled
            and self.reranker.final_k is not None
            and self.reranker.final_k > self.retrieval.candidate_k
        ):
            raise ValueError(
                "reranker.final_k cannot exceed retrieval.candidate_k"
            )

        return self