# Atlas Chatbot

A knowledge assistant that connects an Angular chat interface to a Python retrieval-augmented generation (RAG) service on AWS.

The project brings together source-aware document preparation, Bedrock retrieval and generation, Cognito access control, and an evaluation harness. Public visitors and members use the same interface, while the server determines which sources each request may retrieve.

## Technical highlights

- **Access-aware retrieval:** access classes come from verified identity claims; retrieval filters and post-retrieval checks enforce the boundary.
- **Grounded answers:** responses cite retrieved sources, with checks for unsupported answers and private account information.
- **Document preparation:** source-aware chunking, stable chunk identifiers, metadata validation, and corpus quality checks.
- **Conversation handling:** page-memory history supports follow-up questions; a reload starts a new conversation.
- **Evaluation:** offline fixtures and a live Bedrock gate measure retrieval quality, abstention, latency, and access-control errors.
- **Frontend:** Angular signals, guest chat, Cognito sign-in/sign-up, and escaped Markdown rendering.

## Architecture

```text
Angular chat
  → same-origin API proxy / rewrite
  → API Gateway + Cognito for signed-in requests
  → Python Lambda
  → access-filtered Bedrock retrieval + answer generation
  → answer, citations, request ID
```

The separate [ingestion pipeline](https://github.com/namanhtran200421/fetching_script) supplies cleaned source documents. This repository prepares those documents for retrieval and serves the assistant.

## Explore the code

| Area | Entry point |
| --- | --- |
| Chat UI and authentication | [Frontend guide](frontend/README.md) |
| Retrieval, ingestion, guardrails, and deployment | [Backend guide](membership_atlas/README.md) |
| Retrieval implementation | [retrieval.py](membership_atlas/src/membership_rag/bedrock/retrieval.py) |
| Regression tests | [Backend tests](membership_atlas/tests/) |
| Evaluation cases and thresholds | [Evaluation data](membership_atlas/data/evaluation/) |

## Local development

The backend requires Python 3.12. AWS-dependent features need your own authorised AWS configuration and knowledge base; this is not a credentials-free local demo.

```bash
cd membership_atlas
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python rag_cli.py --help
python -m pytest
```

In a separate terminal, use a Node.js version supported by the Angular version in [package.json](frontend/package.json):

```bash
cd frontend
npm ci
npm start
```

Follow the [frontend configuration guide](frontend/README.md) for Cognito and API settings before connecting it to your own backend.

## Evaluation and limits

Run the deterministic fixture gate without calling Bedrock:

```bash
cd membership_atlas
python rag_cli.py eval-offline \
  --responses data/evaluation/fixtures/heldout_v1_responses.jsonl \
  --output data/evaluation/results/offline-report.json
```

The configured thresholds are release requirements, not a claim that every current deployment meets them. Live evaluation needs AWS access and may incur usage charges. Prompt-injection checks reduce risk but cannot guarantee every attack is caught. See the [backend guide](membership_atlas/README.md) for the full access model and release process.

