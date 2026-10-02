import json
from pathlib import Path
from typing import Any

import pytest

from membership_rag.ingestion import (
    _data_source_target,
    _download_run,
    _publish,
    earliest_cleaned_run,
)


class Paginator:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.pages = pages

    def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.pages


class S3:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.pages = pages
        self.objects: dict[str, bytes] = {}
        self.deleted: list[str] = []

    def get_paginator(self, name: str) -> Paginator:
        assert name == "list_objects_v2"
        return Paginator(self.pages)

    def download_file(self, bucket: str, key: str, path: str) -> None:
        Path(path).write_bytes(self.objects[key])

    def upload_file(self, path: str, bucket: str, key: str) -> None:
        self.objects[key] = Path(path).read_bytes()

    def delete_objects(self, *, Bucket: str, Delete: dict[str, Any]) -> dict[str, Any]:
        for entry in Delete["Objects"]:
            self.deleted.append(entry["Key"])
        return {}


def test_earliest_cleaned_run_ignores_non_run_folders() -> None:
    s3 = S3(
        [
            {"CommonPrefixes": [{"Prefix": "cleaned/2026-09-24_01-00-00_1_UTC/"}]},
            {
                "CommonPrefixes": [
                    {"Prefix": "cleaned/archive/"},
                    {"Prefix": "cleaned/2026-09-10_01-44-40_505837_UTC/"},
                ]
            },
        ]
    )
    assert earliest_cleaned_run(s3, "bucket", "cleaned/") == (
        "cleaned/2026-09-10_01-44-40_505837_UTC/"
    )


def test_download_requires_matching_manifest(tmp_path: Path) -> None:
    run = "cleaned/2026-09-10_01-44-40_505837_UTC/"
    s3 = S3(
        [
            {
                "Contents": [
                    {"Key": run + "research.jsonl"},
                    {"Key": run + "manifest.json"},
                ]
            }
        ]
    )
    s3.objects[run + "research.jsonl"] = b'{}\n'
    s3.objects[run + "manifest.json"] = json.dumps({"run_id": "other"}).encode()
    with pytest.raises(ValueError, match="Manifest run_id"):
        _download_run(s3, "bucket", run, tmp_path)


def test_publish_removes_only_stale_objects_within_rag_prefix(tmp_path: Path) -> None:
    (tmp_path / "chunk.md").write_text("content", encoding="utf-8")
    (tmp_path / "chunk.md.metadata.json").write_text("{}", encoding="utf-8")
    prefix = "rag/bedrock/index/"
    s3 = S3([{"Contents": [{"Key": prefix + "old.md"}, {"Key": prefix + "chunk.md"}]}])
    uploaded, removed = _publish(s3, tmp_path, "bucket", prefix)
    assert (uploaded, removed) == (2, 1)
    assert s3.deleted == [prefix + "old.md"]


def test_data_source_target_reads_live_inclusion_prefix() -> None:
    class Bedrock:
        def get_paginator(self, name: str) -> Paginator:
            return Paginator([{"dataSourceSummaries": [{"dataSourceId": "DS1"}]}])

        def get_data_source(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "dataSource": {
                    "dataSourceConfiguration": {
                        "managedKnowledgeBaseConnectorConfiguration": {
                            "connectorParameters": {
                                "type": "S3",
                                "connectionConfiguration": {"bucketName": "bucket"},
                                "filterConfiguration": {
                                    "inclusionPrefixes": ["rag/bedrock/index/"]
                                },
                            }
                        }
                    }
                }
            }

    assert _data_source_target(Bedrock(), "KB1", None) == (
        "DS1",
        "bucket",
        "rag/bedrock/index/",
    )


def test_data_source_target_parses_aws_json_string() -> None:
    class Bedrock:
        def get_data_source(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "dataSource": {
                    "dataSourceConfiguration": {
                        "managedKnowledgeBaseConnectorConfiguration": {
                            "connectorParameters": json.dumps(
                                {
                                    "type": "S3",
                                    "connectionConfiguration": {"bucketName": "bucket"},
                                    "filterConfiguration": {
                                        "inclusionPrefixes": ["rag/bedrock/index/"]
                                    },
                                }
                            )
                        }
                    }
                }
            }

    assert _data_source_target(Bedrock(), "KB1", "DS1") == (
        "DS1",
        "bucket",
        "rag/bedrock/index/",
    )
