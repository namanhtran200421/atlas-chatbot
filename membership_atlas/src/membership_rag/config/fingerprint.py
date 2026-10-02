import hashlib
import json
from typing import Any

from membership_rag.config.models import RAGConfig


def _fingerprint(value: dict[str, Any]) -> str:
    canonical_json = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()[:16]


def create_config_id(config: RAGConfig) -> str:
    """
    Identifies the complete RAG configuration.

    Any RAG setting change should produce a different ID.
    """

    return _fingerprint(
        config.model_dump(mode="json")
    )


def create_index_id(config: RAGConfig) -> str:
    """
    Identifies configuration that affects the Bedrock
    knowledge-base index.

    Runtime retrieval changes such as K do not require
    rebuilding the index.
    """

    index_config = {
        "corpus": {
            "version": config.corpus.version,
            "s3_uri": config.corpus.s3_uri,
        },

        "preprocessing": (
            config.preprocessing.model_dump(mode='json')
            if config.preprocessing 
            else None
        ),
        "chunking": config.chunking.model_dump(mode="json"),
        "metadata_schema_version": config.metadata_schema_version,
        "embedding": config.embedding.model_dump(mode="json"),
    }

    return _fingerprint(index_config)