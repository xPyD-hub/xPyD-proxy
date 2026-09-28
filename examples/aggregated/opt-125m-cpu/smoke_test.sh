#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_LOG_DIR="${BACKEND_LOG_DIR:-${SCRIPT_DIR}/logs}"
PROXY_ENDPOINT="${PROXY_ENDPOINT:-http://127.0.0.1:8868}"
MODEL="facebook/opt-125m"
export BACKEND="${BACKEND:-127.0.0.1:8000}"

# shellcheck source=../../lib/proxy_api_smoke.sh
source "${SCRIPT_DIR}/../../lib/proxy_api_smoke.sh"

metrics_before="$(mktemp "${TMPDIR:-/tmp}/xpyd-metrics-before.XXXXXX")"
trap 'rm -f "${metrics_before}"' EXIT
capture_prometheus_metrics "${metrics_before}"

completion="$(
    curl --fail-with-body --silent --show-error \
        "${PROXY_ENDPOINT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "facebook/opt-125m",
            "prompt": "The proxy smoke test says",
            "max_tokens": 4,
            "temperature": 0
        }'
)"

python -c '
import json
import sys

output = json.load(sys.stdin)
assert output["object"] == "text_completion", output
assert output["model"] == "facebook/opt-125m", output
assert isinstance(output["id"], str) and output["id"], output
assert isinstance(output["created"], int), output
assert len(output["choices"]) == 1, output
assert output["choices"][0]["index"] == 0, output
assert output["choices"][0]["text"], "empty completion"
assert output["choices"][0]["finish_reason"] == "length", output
assert output["usage"]["completion_tokens"] == 4, output["usage"]
' <<<"${completion}"

smoke_chat_completion
smoke_streaming_metrics aggregated

health="$(
    curl --fail --silent --show-error "${PROXY_ENDPOINT}/health"
)"
python -c '
import json
import os
import sys

output = json.load(sys.stdin)
backend = output[os.environ["BACKEND"]]
assert backend["status"] == 200, output
assert backend["type"] == "text", output
' <<<"${health}"

models="$(
    curl --fail --silent --show-error "${PROXY_ENDPOINT}/v1/models"
)"
python -c '
import json
import sys

output = json.load(sys.stdin)
assert output["object"] == "list", output
assert [model["id"] for model in output["data"]] == [
    "facebook/opt-125m"
], output
' <<<"${models}"

unsupported_media_status="$(
    curl --silent --output /dev/null --write-out "%{http_code}" \
        "${PROXY_ENDPOINT}/v1/completions" \
        -H "Content-Type: text/plain" \
        -d "not json"
)"
[[ "${unsupported_media_status}" == "415" ]]

unknown_model_status="$(
    curl --silent --output /dev/null --write-out "%{http_code}" \
        "${PROXY_ENDPOINT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "unknown-model",
            "prompt": "This must not be routed",
            "max_tokens": 1
        }'
)"
[[ "${unknown_model_status}" == "404" ]]

smoke_all_endpoints
validate_prometheus_metrics aggregated "${metrics_before}" 3
rm -f "${metrics_before}"
trap - EXIT

echo "OPT-125M aggregated CPU API smoke tests passed."
