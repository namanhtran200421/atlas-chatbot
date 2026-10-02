from pathlib import Path

import yaml

from membership_rag.config.models import RAGConfig


def load_rag_config(path: str | Path) -> RAGConfig:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"RAG config does not exist: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    if not isinstance(raw_config, dict):
        raise TypeError(
            f"RAG config must contain a YAML object: {config_path}"
        )

    return RAGConfig.model_validate(raw_config)
