#!/usr/bin/env bash
#
# Build and deploy the Membership Atlas answer API to AWS Lambda.
#
# The package carries only the modules the request path imports, plus boto3.
# The evaluation harness, the corpus tooling and the CLI stay out of it: the
# README is explicit that evaluations belong in CI and release operations, not
# in the function.
#
#   ./scripts/deploy_lambda.sh                 # build, test and upload $LATEST
#   ./scripts/deploy_lambda.sh --publish       # also publish a numbered version
#   ./scripts/deploy_lambda.sh --publish --promote   # ...and move the prod alias
#
# Promotion is deliberately a separate flag. Uploading touches $LATEST only;
# live traffic follows the alias and does not move until you ask for it.

set -euo pipefail

FUNCTION_NAME="${MEMBERSHIP_RAG_FUNCTION:-membership-rag-api}"
REGION="${AWS_REGION:-ap-southeast-2}"
ALIAS="${MEMBERSHIP_RAG_ALIAS:-prod}"
PUBLISH=0
PROMOTE=0

for argument in "$@"; do
  case "$argument" in
    --publish) PUBLISH=1 ;;
    --promote) PROMOTE=1 ;;
    *) echo "unknown option: $argument" >&2; exit 2 ;;
  esac
done

if [[ "$PROMOTE" -eq 1 && "$PUBLISH" -eq 0 ]]; then
  echo "--promote requires --publish: an alias points at a numbered version" >&2
  exit 2
fi

REPOSITORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$REPOSITORY/src/membership_rag"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

# Modules reachable from the handler. Anything not listed is not shipped, so a
# new import in the request path must be added here.
RUNTIME_MODULES=(
  lambda_handler.py
  reference.py
  conversation.py
  session_state.py
  guardrails.py
  access_policy.py
  scope.py
  privacy.py
)
BEDROCK_MODULES=(
  __init__.py
  generation.py
  retrieval.py
  routing.py
  ranking.py
  verification.py
)

echo "==> Running the checks that gate a deployment"
cd "$REPOSITORY"
python -m pytest -q
python scripts/evaluate_conversation.py
python scripts/evaluate_session.py
ruff check src tests scripts rag_cli.py
mypy

echo "==> Staging the package"
mkdir -p "$STAGING/membership_rag/bedrock"
for module in "${RUNTIME_MODULES[@]}"; do
  cp "$SOURCE/$module" "$STAGING/membership_rag/$module"
done
for module in "${BEDROCK_MODULES[@]}"; do
  cp "$SOURCE/bedrock/$module" "$STAGING/membership_rag/bedrock/$module"
done
touch "$STAGING/membership_rag/__init__.py"
[[ -f "$SOURCE/py.typed" ]] && cp "$SOURCE/py.typed" "$STAGING/membership_rag/py.typed"

# Pinned to the Lambda runtime's platform so a wheel built for macOS never
# reaches the function.
pip install --quiet --target "$STAGING" \
  --platform manylinux2014_x86_64 --implementation cp --python-version 3.12 \
  --only-binary=:all: "boto3>=1.40,<2"

find "$STAGING" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf "$STAGING"/*.dist-info "$STAGING"/bin

echo "==> Verifying the staged package imports on its own"
PYTHONPATH="$STAGING" python -c "
import sys
sys.path.insert(0, '$STAGING')
from membership_rag.lambda_handler import handler
response = handler(
    {'requestContext': {'http': {'method': 'GET'}}, 'rawPath': '/health', 'headers': {}},
    None,
)
assert response['statusCode'] == 200, response
print('    health route OK')
"

ARCHIVE="$STAGING.zip"
trap 'rm -rf "$STAGING" "$ARCHIVE"' EXIT
(cd "$STAGING" && zip -qr "$ARCHIVE" . -x "*.pyc" -x "*__pycache__*")
echo "==> Package: $(du -h "$ARCHIVE" | cut -f1)"

echo "==> Uploading to $FUNCTION_NAME (\$LATEST)"
aws lambda update-function-code \
  --function-name "$FUNCTION_NAME" --region "$REGION" \
  --zip-file "fileb://$ARCHIVE" --output text \
  --query '[LastUpdateStatus,CodeSha256]'
aws lambda wait function-updated --function-name "$FUNCTION_NAME" --region "$REGION"

if [[ "$PUBLISH" -eq 1 ]]; then
  VERSION="$(aws lambda publish-version \
    --function-name "$FUNCTION_NAME" --region "$REGION" \
    --query 'Version' --output text)"
  echo "==> Published version $VERSION"
  if [[ "$PROMOTE" -eq 1 ]]; then
    aws lambda update-alias \
      --function-name "$FUNCTION_NAME" --region "$REGION" \
      --name "$ALIAS" --function-version "$VERSION" --output text \
      --query '[Name,FunctionVersion]'
    echo "==> Alias $ALIAS now serves version $VERSION"
    echo "    Roll back with: aws lambda update-alias --function-name $FUNCTION_NAME \\"
    echo "      --region $REGION --name $ALIAS --function-version <previous>"
  else
    echo "==> Alias $ALIAS left unchanged; promote with --promote"
  fi
fi

echo "==> Done"
