"""Command-line tools for operating and evaluating the RAG pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import boto3

from membership_rag.bedrock import BedrockRetriever
from membership_rag.corpus import check_corpus_directory
from membership_rag.evaluation import (
    EvaluationReport,
    EvaluationThresholds,
    QueryEvaluation,
    RetrievalEvaluator,
    build_evaluation_report,
    evaluate_saved_responses,
    load_evaluation_cases,
)
from membership_rag.ingestion import ingest_earliest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data/evaluation/retrieval_cases_production_v1.jsonl"
DEFAULT_THRESHOLDS = PROJECT_ROOT / "data/evaluation/production_thresholds.json"
DEFAULT_INGEST_CONFIG = PROJECT_ROOT / "configs/membership_v2.yaml"


def _setting(name: str) -> str | None:
    """Read a CLI setting from the environment or the project's local .env."""
    if os.environ.get(name):
        return os.environ[name]
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == name:
                return value.strip().strip("\"'")
    return None


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _write_report(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Report saved to {path}")


def _aws_session(profile: str | None, region: str) -> boto3.Session:
    return boto3.Session(profile_name=profile, region_name=region)


def _aws_identity(profile: str | None, region: str) -> dict[str, Any]:
    response = _aws_session(profile, region).client("sts").get_caller_identity()
    return {key: response[key] for key in ("Account", "Arn", "UserId")}


def command_aws_login(args: argparse.Namespace) -> int:
    if shutil.which("aws") is None:
        print("AWS CLI is not installed or is not on PATH.", file=sys.stderr)
        return 2

    command = ["aws", "login"]
    if args.profile:
        command.extend(["--profile", args.profile])
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return completed.returncode

    try:
        identity = _aws_identity(args.profile, args.region)
    except Exception as exc:  # noqa: BLE001 - surface any AWS credential failure.
        print(f"Login completed, but STS verification failed: {exc}", file=sys.stderr)
        return 2
    _print_json(identity)
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    checks: dict[str, Any] = {
        "python": sys.version.split()[0],
        "aws_cli": shutil.which("aws") is not None,
        "dataset": args.dataset.exists(),
        "thresholds": args.thresholds.exists(),
    }
    if not args.skip_aws:
        try:
            checks["aws_identity"] = _aws_identity(args.profile, args.region)
        except Exception as exc:  # noqa: BLE001 - doctor reports all AWS failures.
            checks["aws_error"] = str(exc)

    _print_json(checks)
    required_ok = all(checks[key] for key in ("aws_cli", "dataset", "thresholds"))
    aws_ok = args.skip_aws or "aws_identity" in checks
    return 0 if required_ok and aws_ok else 1


def _report_payload(
    report: EvaluationReport,
    results: list[QueryEvaluation],
) -> dict[str, Any]:
    payload = report.to_dict()
    payload["queries"] = [result.model_dump(mode="json") for result in results]
    return payload


def command_eval_offline(args: argparse.Namespace) -> int:
    cases = load_evaluation_cases(args.dataset)
    results = evaluate_saved_responses(cases, args.responses)
    thresholds = EvaluationThresholds.load(args.thresholds)
    report = build_evaluation_report(cases, results, thresholds)
    payload = _report_payload(report, results)
    _print_json(report.to_dict())
    _write_report(args.output, payload)
    return 0 if report.passed else 1


def command_eval_live(args: argparse.Namespace) -> int:
    knowledge_base_id = args.knowledge_base_id or os.environ.get("MEMBERSHIP_RAG_KB_ID")
    if not knowledge_base_id:
        print(
            "Set MEMBERSHIP_RAG_KB_ID or pass --knowledge-base-id.",
            file=sys.stderr,
        )
        return 2

    session = _aws_session(args.profile, args.region)
    client = session.client("bedrock-agent-runtime")
    retriever = BedrockRetriever(
        knowledge_base_id,
        args.region,
        client=client,
    )
    cases = load_evaluation_cases(args.dataset)
    results = RetrievalEvaluator(retriever).evaluate_cases(cases)
    thresholds = EvaluationThresholds.load(args.thresholds)
    report = build_evaluation_report(cases, results, thresholds)
    payload = _report_payload(report, results)
    _print_json(report.to_dict())
    _write_report(args.output, payload)
    return 0 if report.passed else 1


def command_corpus_check(args: argparse.Namespace) -> int:
    report = check_corpus_directory(args.path)
    _print_json(report.to_dict())
    return 0 if report.passed else 1


def command_ingest(args: argparse.Namespace) -> int:
    knowledge_base_id = args.knowledge_base_id or _setting("MEMBERSHIP_RAG_KB_ID")
    if not knowledge_base_id:
        print("Set MEMBERSHIP_RAG_KB_ID or pass --knowledge-base-id.", file=sys.stderr)
        return 2
    result = ingest_earliest(
        knowledge_base_id=knowledge_base_id,
        data_source_id=args.data_source_id,
        profile=args.profile,
        region=args.region,
        config_path=args.config,
    )
    _print_json(result)
    return 0


def _common_eval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--output", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="membership-rag",
        description="Operate and evaluate the Membership Atlas RAG pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    login = subparsers.add_parser("aws-login", help="Log in and verify AWS STS.")
    login.add_argument("--profile")
    login.add_argument("--region", default=os.environ.get("AWS_REGION", "ap-southeast-2"))
    login.set_defaults(handler=command_aws_login)

    doctor = subparsers.add_parser("doctor", help="Check local files and AWS access.")
    doctor.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    doctor.add_argument("--region", default=os.environ.get("AWS_REGION", "ap-southeast-2"))
    doctor.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    doctor.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    doctor.add_argument("--skip-aws", action="store_true")
    doctor.set_defaults(handler=command_doctor)

    offline = subparsers.add_parser(
        "eval-offline", help="Replay saved responses without AWS."
    )
    _common_eval_arguments(offline)
    offline.add_argument("--responses", type=Path, required=True)
    offline.set_defaults(handler=command_eval_offline)

    live = subparsers.add_parser(
        "eval-live", help="Run the production gate against Bedrock."
    )
    _common_eval_arguments(live)
    live.add_argument("--knowledge-base-id")
    live.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    live.add_argument("--region", default=os.environ.get("AWS_REGION", "ap-southeast-2"))
    live.set_defaults(handler=command_eval_live)

    corpus = subparsers.add_parser(
        "corpus-check", help="Validate generated chunks and metadata."
    )
    corpus.add_argument("path", type=Path)
    corpus.set_defaults(handler=command_corpus_check)

    ingest = subparsers.add_parser(
        "ingest", help="Build the earliest cleaned run and start Bedrock ingestion."
    )
    ingest.add_argument("--knowledge-base-id")
    ingest.add_argument("--data-source-id")
    ingest.add_argument("--profile", default=_setting("AWS_PROFILE"))
    ingest.add_argument("--region", default=_setting("AWS_REGION") or "ap-southeast-2")
    ingest.add_argument("--config", type=Path, default=DEFAULT_INGEST_CONFIG)
    ingest.set_defaults(handler=command_ingest)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - keep CLI errors concise for operators.
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
