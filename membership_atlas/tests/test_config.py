from pathlib import Path

import pytest
from pydantic import ValidationError

from membership_rag.config import (
    create_config_id,
    create_index_id,
    load_rag_config,
)
from membership_rag.config.models import (
    EmbeddingConfig,
)

CONFIG_PATH = Path("configs/membership_v1.yaml")


def test_config_loads() -> None:
    config = load_rag_config(CONFIG_PATH)

    assert config.config_version == "rag-v1"
    assert config.chunking.strategy == "FIXED_SIZE"
    assert config.embedding.dimensions == 1024


def test_config_is_immutable() -> None:
    config = load_rag_config(CONFIG_PATH)

    with pytest.raises(ValidationError):
        config.config_version = "rag-v2"


def test_same_config_has_same_config_id() -> None:
    first = load_rag_config(CONFIG_PATH)
    second = load_rag_config(CONFIG_PATH)

    assert create_config_id(first) == create_config_id(second)


def test_embedding_change_changes_index_id() -> None:
    original = load_rag_config(CONFIG_PATH)

    modified = original.model_copy(
        update={
            "embedding": EmbeddingConfig(
                model_id="amazon.titan-embed-text-v2:0",
                dimensions=512,
                data_type="FLOAT32",
            )
        }
    )

    assert create_index_id(original) != create_index_id(modified)


def test_retrieval_k_does_not_change_index_id() -> None:
    original = load_rag_config(CONFIG_PATH)

    modified = original.model_copy(
        update={
            "retrieval": original.retrieval.model_copy(
                update={"candidate_k": 10}
            )
        }
    )

    assert create_config_id(original) != create_config_id(modified)

    assert create_index_id(original) == create_index_id(modified)

def test_config_can_be_serialized_into_run_report() -> None:
    config = load_rag_config(CONFIG_PATH)

    report = {
        "config_id": create_config_id(config),
        "index_id": create_index_id(config),
        "config": config.model_dump(mode="json"),
    }

    assert report["config_id"]
    assert report["index_id"]
    assert report["config"]["config_version"] == "rag-v1"
    assert report["config"]["embedding"]["dimensions"] == 1024


def test_preprocessing_chunk_change_changes_index_id() -> None:
    original = load_rag_config(
        Path("configs/membership_v2.yaml")
    )

    assert original.preprocessing is not None

    original_chunking = original.preprocessing.chunking
    original_policies = original_chunking.policies

    modified_article = (
        original_policies.article.model_copy(
            update={
                "max_tokens": 768,
            }
        )
    )

    modified_policies = (
        original_policies.model_copy(
            update={
                "article": modified_article,
            }
        )
    )

    modified_chunking = (
        original_chunking.model_copy(
            update={
                "policies": modified_policies,
            }
        )
    )

    modified_preprocessing = (
        original.preprocessing.model_copy(
            update={
                "chunking": modified_chunking,
            }
        )
    )

    modified = original.model_copy(
        update={
            "preprocessing": modified_preprocessing,
        }
    )

    assert (
        create_index_id(original)
        != create_index_id(modified)
    )

def test_prechunked_config_uses_bedrock_none() -> None:
    config = load_rag_config(
        Path("configs/membership_v2.yaml")
    )

    assert config.preprocessing is not None
    assert config.chunking.strategy == "NONE"