"""Build and publish the earliest cleaned corpus for Bedrock ingestion."""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlparse

import boto3

from membership_rag.chunking import SourceAwareChunker, write_bedrock_chunks
from membership_rag.config import load_rag_config
from membership_rag.corpus import check_corpus_directory, load_documents


def _split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an S3 URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/") + "/"


def earliest_cleaned_run(s3: Any, bucket: str, prefix: str) -> str:
    """Choose the first timestamped child folder by its UTC run name."""
    run_prefixes: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        run_prefixes.extend(
            entry["Prefix"]
            for entry in page.get("CommonPrefixes", [])
            if re.fullmatch(
                r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d+_UTC/",
                entry["Prefix"].removeprefix(prefix),
            )
        )
    if not run_prefixes:
        raise FileNotFoundError(f"No run folders found at s3://{bucket}/{prefix}")
    return min(run_prefixes)


def _download_run(
    s3: Any, bucket: str, run_prefix: str, dest: Path
) -> tuple[int, str | None, bool | None]:
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=run_prefix):
        for entry in page.get("Contents", []):
            key = entry["Key"]
            name = key.removeprefix(run_prefix)
            if "/" not in name and (name.endswith(".jsonl") or name == "manifest.json"):
                keys.append(key)
    if not any(key.endswith(".jsonl") for key in keys):
        raise FileNotFoundError(f"No JSONL files found at s3://{bucket}/{run_prefix}")
    if f"{run_prefix}manifest.json" not in keys:
        raise FileNotFoundError(f"No manifest.json found at s3://{bucket}/{run_prefix}")

    for key in keys:
        s3.download_file(bucket, key, str(dest / key.removeprefix(run_prefix)))

    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    run_id = run_prefix.rstrip("/").split("/")[-1]
    if manifest.get("run_id") != run_id:
        raise ValueError(f"Manifest run_id does not match selected folder {run_id!r}")
    expected = {
        uri.removeprefix(f"s3://{bucket}/")
        for uri in manifest.get("cleaned_outputs", {}).values()
    }
    if expected and not expected.issubset(keys):
        raise ValueError("Selected cleaned run is missing files named in its manifest")
    return (
        sum(key.endswith(".jsonl") for key in keys),
        manifest.get("status"),
        manifest.get("quality_gate_passed"),
    )


def _data_source_target(bedrock: Any, knowledge_base_id: str, data_source_id: str | None) -> tuple[str, str, str]:
    if data_source_id is None:
        ids: list[str] = []
        paginator = bedrock.get_paginator("list_data_sources")
        for page in paginator.paginate(knowledgeBaseId=knowledge_base_id):
            ids.extend(item["dataSourceId"] for item in page.get("dataSourceSummaries", []))
        if len(ids) != 1:
            raise ValueError(
                f"Knowledge base has {len(ids)} data sources; pass --data-source-id to select one"
            )
        data_source_id = ids[0]

    response = bedrock.get_data_source(
        knowledgeBaseId=knowledge_base_id,
        dataSourceId=data_source_id,
    )
    configuration = response["dataSource"]["dataSourceConfiguration"]
    connector = configuration.get("managedKnowledgeBaseConnectorConfiguration", {})
    parameters = connector.get("connectorParameters", {})
    if isinstance(parameters, str):
        parameters = json.loads(parameters)
    if not isinstance(parameters, dict):
        raise TypeError("Data source connectorParameters must be a JSON object")
    connection = parameters.get("connectionConfiguration", {})
    filters = parameters.get("filterConfiguration", {})
    prefixes = filters.get("inclusionPrefixes", [])
    bucket = connection.get("bucketName")
    if parameters.get("type") != "S3" or not bucket or len(prefixes) != 1:
        raise ValueError("Data source must be an S3 connector with one inclusion prefix")
    prefix = prefixes[0]
    if not prefix.startswith("rag/") or not prefix.endswith("/"):
        raise ValueError(f"Refusing to publish outside a rag/ prefix: {prefix!r}")
    return data_source_id, bucket, prefix


def _publish(s3: Any, source: Path, bucket: str, prefix: str) -> tuple[int, int]:
    paths = sorted(source.iterdir())
    names = {path.name for path in paths if path.is_file()}
    if not names or any(
        not name.endswith((".md", ".md.metadata.json")) for name in names
    ):
        raise ValueError("Build output contains unexpected files")

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(lambda path: s3.upload_file(str(path), bucket, prefix + path.name), paths))

    stale: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.removeprefix(prefix) not in names:
                stale.append(key)
    for offset in range(0, len(stale), 1000):
        response = s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": key} for key in stale[offset : offset + 1000]]},
        )
        if response.get("Errors"):
            raise RuntimeError(f"Failed to remove stale RAG files: {response['Errors'][:5]}")
    return len(names), len(stale)


def ingest_earliest(
    *,
    knowledge_base_id: str,
    profile: str | None = None,
    region: str = "ap-southeast-2",
    data_source_id: str | None = None,
    config_path: Path = Path("configs/membership_v2.yaml"),
) -> dict[str, Any]:
    config = load_rag_config(config_path)
    source_bucket, source_prefix = _split_s3_uri(config.corpus.s3_uri)
    session = boto3.Session(profile_name=profile, region_name=region)
    s3 = session.client("s3")
    bedrock = session.client("bedrock-agent")

    run_prefix = earliest_cleaned_run(s3, source_bucket, source_prefix)
    selected_id, target_bucket, target_prefix = _data_source_target(
        bedrock, knowledge_base_id, data_source_id
    )
    if target_bucket != source_bucket:
        raise ValueError("Cleaned corpus and RAG data source use different buckets")

    with TemporaryDirectory(prefix="membership-rag-") as temporary:
        input_dir = Path(temporary) / "input"
        output_dir = Path(temporary) / "output"
        input_dir.mkdir()
        file_count, manifest_status, quality_gate_passed = _download_run(
            s3, source_bucket, run_prefix, input_dir
        )
        print(
            f"Selected earliest cleaned run: s3://{source_bucket}/{run_prefix} "
            f"(status={manifest_status}, quality_gate_passed={quality_gate_passed})",
            file=sys.stderr,
            flush=True,
        )
        documents = load_documents(input_dir)
        chunks = SourceAwareChunker.from_rag_config(config).chunk_documents(documents)
        write_bedrock_chunks(chunks, output_dir)
        report = check_corpus_directory(output_dir)
        if not report.passed:
            raise ValueError(f"Corpus validation failed: {report.errors[:5]}")
        uploaded, removed = _publish(s3, output_dir, target_bucket, target_prefix)

    job = bedrock.start_ingestion_job(
        knowledgeBaseId=knowledge_base_id,
        dataSourceId=selected_id,
        description=f"Membership RAG from {run_prefix.rstrip('/').split('/')[-1]}",
    )["ingestionJob"]
    return {
        "cleaned_run": f"s3://{source_bucket}/{run_prefix}",
        "rag_prefix": f"s3://{target_bucket}/{target_prefix}",
        "source_files": file_count,
        "cleaned_manifest_status": manifest_status,
        "cleaned_quality_gate_passed": quality_gate_passed,
        "documents": len(documents),
        "chunks": len(chunks),
        "uploaded_files": uploaded,
        "removed_stale_files": removed,
        "knowledge_base_id": knowledge_base_id,
        "data_source_id": selected_id,
        "ingestion_job_id": job["ingestionJobId"],
        "ingestion_status": job["status"],
    }
