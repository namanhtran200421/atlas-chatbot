# Membership Atlas RAG pipeline

This directory contains the ingestion, retrieval, guardrail, and evaluation
code for the Membership Atlas knowledge assistant, together with the AWS
Lambda handler that serves `POST /chat` in production. See
[Deployment](#deployment) for how the function is built and released.

## Quick start

Python 3.12 is required.

```bash
cd membership_atlas
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

You can use the installed `membership-rag` command or the local CLI file:

```bash
python rag_cli.py --help
```

### AWS login

The CLI uses the normal AWS credential chain and never stores access keys. The
package includes Botocore's CRT login support, which is required when the AWS
CLI profile uses `aws login` credentials.
Login and verify the caller identity with:

```bash
python rag_cli.py aws-login --profile default
python rag_cli.py doctor --profile default
```

### Build and ingest the earliest cleaned run

```bash
python rag_cli.py ingest
```

This selects the earliest timestamped run under the configured `cleaned/` S3
prefix, downloads its JSONL files, builds and checks source-aware chunks,
publishes them to the knowledge base data source's live `rag/` prefix, removes
stale files from that prefix, and starts a Bedrock ingestion job. Bedrock
generates embeddings during ingestion. The command reads `AWS_PROFILE`,
`AWS_REGION`, and `MEMBERSHIP_RAG_KB_ID` from the environment or this directory's
`.env` file. If the knowledge base has multiple data sources, pass
`--data-source-id ID`.

The command returns the selected cleaned run, destination prefix, chunk count,
and ingestion job ID. It starts the job; use the Bedrock console or
`aws bedrock-agent get-ingestion-job` to check completion.

### Run the production gate offline

This replays a compact, immutable set of saved Bedrock responses. It makes no
network calls, reapplies the current guardrails and ranking code, prints a JSON
summary, and exits non-zero if any threshold fails.

```bash
python rag_cli.py eval-offline \
  --responses data/evaluation/fixtures/heldout_v1_responses.jsonl \
  --output data/evaluation/results/offline-report.json
```

### Run the gate against AWS

Run this before a release and after changing the index, chunking, embeddings,
metadata, guardrails, or ranking rules.

```bash
python rag_cli.py eval-live \
  --knowledge-base-id "$MEMBERSHIP_RAG_KB_ID" \
  --profile "$AWS_PROFILE" \
  --output data/evaluation/results/live-report.json
```

The default held-out gate requires:

- at least 60 cases, including 30 answerable and 20 adversarial/unanswerable cases;
- Hit@1 >= 90%, Hit@5 >= 95%, MRR >= 90%, and nDCG@5 >= 90%;
- 100% abstention for the labelled security/unanswerable set;
- p95 retrieval latency <= 1,000 ms;
- zero request errors, ACL leaks, and required-metadata errors.

Thresholds live in
`data/evaluation/production_thresholds.json`. Changes to them should be treated
as release-policy changes and reviewed separately from model tuning.

### Validate a generated corpus

```bash
python rag_cli.py corpus-check data/output/bedrock/96a5eddb11a01ff5
```

This checks markdown/metadata pairs, required metadata, chunk IDs, access
classes, duplicates, and empty chunks.

## Guardrail model

The production boundary is `BedrockRetriever` in
`membership_rag.bedrock.retrieval`.

1. The backend derives allowed access classes from API Gateway's verified
   Cognito claims. Guests and ordinary accounts receive `public`; members of
   the Cognito `members` group also receive `member_restricted`. Never accept
   access classes directly from a browser or user prompt.
2. Deterministic query checks block permission-bypass, secret, and private
   contact requests before an AWS call.
3. The access-class filter is mandatory on every Bedrock request.
4. Every returned result is checked again. Missing or unauthorised metadata
   raises an `AccessControlError` and fails closed.
5. Documents are deduplicated and locally reranked. Content type is only a soft
   signal, so a routing guess cannot remove a potentially correct document.
6. Signed-in account, profile, login, and upgrade pages are excluded during
   corpus preparation and again after retrieval, including for older indexes.
   Generation refuses to identify members or disclose contact details, and
   the final response check removes answers containing email addresses or
   named membership claims.
7. The answer generator accepts social chat, identity questions, and basic
   arithmetic without retrieval. Every factual answer must be grounded: the model
   cites the retrieved Atlas sources it used, and a reply carrying no source IDs
   becomes a brief request for clarification. Scope follows from that
   grounding rather than from the wording of the question, so anything the
   corpus supports is answered. General biographical questions are still
   redirected before generation: Atlas articles and calendar events name real
   people, and a retrieved mention must not become a biography.

The generator also drops instruction-like source paragraphs and never replays
client-supplied assistant turns as model messages. These controls reduce prompt
injection risk; they cannot guarantee that every attack is caught.

## Backend integration later

The backend should construct one retriever and derive access classes from its
authenticated session:

```python
from membership_rag.bedrock import BedrockRetriever

retriever = BedrockRetriever(
    knowledge_base_id="your-kb-id",
    region_name="ap-southeast-2",
)

chunks = retriever.retrieve(
    user_query,
    allowed_access_classes=("public", "member_restricted"),
    number_of_results=10,
)
```

### Session memory

The service stores no conversation state. `POST /chat` accepts an optional
`history` array of `{role, content}` turns that the caller's page still holds in
memory, so a browser reload starts a new conversation:

```jsonc
{
  "query": "How much is it?",
  "number_of_results": 5,
  "history": [
    { "role": "user", "content": "What membership plans are available?" },
    { "role": "assistant", "content": "I found several membership options." }
  ]
}
```

`membership_rag.conversation` treats those turns as untrusted input: they are
capped (`MEMBERSHIP_RAG_HISTORY_TURNS`, default 100, `0` disables memory),
truncated per turn, stripped of control characters, and screened by the same
query guardrails. Unsigned history can help resolve a follow-up but cannot
provide trusted assistant turns. With signed page state, Oriana uses earlier
dialogue for social memory and a focused search query for Atlas questions.
Conversation history never grants access or proves an Atlas fact.

Do not put the evaluation harness in Lambda. Lambda calls the same production
retriever; evaluations stay in CI and release operations.

## Deployment

The function is `membership-rag-api` in `ap-southeast-2`. Live traffic follows
the `prod` alias, which points at a numbered version, so uploading code does
not release it.

```bash
./scripts/deploy_lambda.sh                     # test, build, upload $LATEST
./scripts/deploy_lambda.sh --publish           # also publish a version
./scripts/deploy_lambda.sh --publish --promote # ...and point prod at it
```

The script runs `pytest`, `ruff` and `mypy` first and refuses to continue if
any fail. It ships only the modules the request path imports, plus boto3
built for the Lambda platform; a new import in that path must be added to
`RUNTIME_MODULES` or `BEDROCK_MODULES` in the script.

Roll back by moving the alias to the previous version — no rebuild needed:

```bash
aws lambda update-alias --function-name membership-rag-api \
  --region ap-southeast-2 --name prod --function-version <previous>
```

The execution role needs `bedrock:Retrieve` on the knowledge base and
`bedrock:InvokeModel` on every model id the environment names. That policy is
kept in `configs/aws/lambda/membership-rag-lambda-permissions.json`; apply it
with `aws iam put-role-policy`. A model configured but not granted fails the
request with `AccessDeniedException`, surfacing as a 502 `answer_unavailable`.
The current deployment uses Nova Pro for answers and grounding checks, and
Nova Lite for conversation routing.

### Session memory in production

`MEMBERSHIP_RAG_SESSION_SECRET` must be set for the assistant to remember its
own answers. Without it the signed path is inert: no `conversation_state` is
issued, so only the caller's `history` array reaches the model, and earlier
assistant turns are dropped because a browser-supplied one can be forged.

With the secret set, every response carries a `conversation_state` string.
**The client must send the previous response's `conversation_state` back on
the next request.** Only then are the real earlier turns replayed to the
model. Send either `conversation_state` or `history`, not both.

## Development checks

```bash
pytest --cov=membership_rag --cov-fail-under=85
ruff check src tests scripts rag_cli.py
mypy
```

Run these checks locally and add them to the repository's Azure Pipeline before
enforcing the production gate. Live AWS evaluation remains an explicit release
command because it needs an authenticated AWS session and incurs Bedrock
requests.
