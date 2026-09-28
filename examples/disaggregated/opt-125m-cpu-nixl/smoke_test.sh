#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_LOG_DIR="${BACKEND_LOG_DIR:-${SCRIPT_DIR}/logs}"
PROXY_ENDPOINT="${PROXY_ENDPOINT:-http://127.0.0.1:8868}"
MODEL="facebook/opt-125m"
REQUEST_COUNT="${REQUEST_COUNT:-1}"

# shellcheck source=../../lib/proxy_api_smoke.sh
source "${SCRIPT_DIR}/../../lib/proxy_api_smoke.sh"

metrics_before="$(mktemp "${TMPDIR:-/tmp}/xpyd-metrics-before.XXXXXX")"
trap 'rm -f "${metrics_before}"' EXIT
capture_prometheus_metrics "${metrics_before}"

for ((request_index = 1; request_index <= REQUEST_COUNT; request_index++)); do
    completion="$(
        curl --fail-with-body --silent --show-error \
            "${PROXY_ENDPOINT}/v1/completions" \
            -H "Content-Type: application/json" \
            -d '{
                "model": "facebook/opt-125m",
                "prompt": "The disaggregated proxy test says",
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
assert output["choices"][0]["text"], output
assert output["choices"][0]["finish_reason"] == "length", output
assert output["usage"]["completion_tokens"] == 4, output["usage"]
' <<<"${completion}"
done

smoke_chat_completion
smoke_streaming_metrics disaggregated

models="$(
    curl --fail --silent --show-error "${PROXY_ENDPOINT}/v1/models"
)"
python -c '
import json
import sys

output = json.load(sys.stdin)
assert [model["id"] for model in output["data"]] == [
    "facebook/opt-125m"
], output
' <<<"${models}"

smoke_all_endpoints
validate_prometheus_metrics \
    disaggregated "${metrics_before}" "$((REQUEST_COUNT + 1))" \
    --expected-prefill "${EXPECTED_PREFILL_INSTANCES:-}" \
    --expected-decode "${EXPECTED_DECODE_INSTANCES:-}"
rm -f "${metrics_before}"
trap - EXIT

echo "OPT-125M NIXL TCP ${TOPOLOGY_NAME:-1P1D} smoke test passed."
