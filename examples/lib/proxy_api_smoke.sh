#!/usr/bin/env bash
# Shared proxy API assertions used by the CPU example smoke tests.
#
# Source this file and call the smoke_* functions. It depends on
# PROXY_ENDPOINT, MODEL and BACKEND_LOG_DIR, and works for aggregated and
# disaggregated topologies of any size, including phases where some nodes are
# deliberately offline.
#
#   PROXY_ENDPOINT  proxy base URL, defaults to http://127.0.0.1:8868
#   MODEL           served model name
#   ADMIN_API_KEY   optional; when set the admin assertions expect 403 for a
#                   wrong key instead of the 500 an unconfigured proxy returns

PROXY_ENDPOINT="${PROXY_ENDPOINT:-http://127.0.0.1:8868}"
MODEL="${MODEL:-facebook/opt-125m}"
PROMETHEUS_VALIDATOR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
)/validate_prometheus_metrics.py"
FORWARDING_VALIDATOR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
)/validate_forwarding.py"

# Status of a JSON POST, discarding the body.
post_status() {
    curl --silent --output /dev/null --write-out "%{http_code}" \
        "${PROXY_ENDPOINT}$1" \
        -H "Content-Type: application/json" \
        -d "$2"
}

assert_status() {
    local label=$1 expected=$2 actual=$3
    [[ "${actual}" == "${expected}" ]] || {
        echo "ERROR: ${label} expected HTTP ${expected}, got ${actual}." >&2
        return 1
    }
}

# Unsupported model tasks may return 4xx, but only a new backend log entry
# proves the request was forwarded rather than rejected by the proxy.
assert_forwarded() {
    python "${FORWARDING_VALIDATOR}" --url "${PROXY_ENDPOINT}" \
        --path "$1" --body "$2" \
        --log-dir "${BACKEND_LOG_DIR:?Set BACKEND_LOG_DIR to the backend logs directory}"
}

smoke_chat_completion() {
    echo "=== Chat completion endpoint ==="

    local result body status
    result="$(
        curl --silent --show-error --write-out $'\n%{http_code}' \
            "${PROXY_ENDPOINT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"${MODEL}\",
                \"messages\": [{\"role\": \"user\", \"content\": \"Say hello\"}],
                \"max_tokens\": 4,
                \"temperature\": 0
            }"
    )"
    body="${result%$'\n'*}"
    status="${result##*$'\n'}"
    assert_status "/v1/chat/completions" 200 "${status}"

    python -c '
import json
import sys

output = json.load(sys.stdin)
assert output["object"] == "chat.completion", output
assert output["model"] == "facebook/opt-125m", output
assert isinstance(output["id"], str) and output["id"], output
assert isinstance(output["created"], int), output
assert len(output["choices"]) == 1, output
choice = output["choices"][0]
assert choice["index"] == 0, output
assert choice["message"]["role"] == "assistant", output
assert choice["message"]["content"], output
assert choice["finish_reason"] == "length", output
assert output["usage"]["completion_tokens"] == 4, output["usage"]
' <<<"${body}"
    echo "  /v1/chat/completions ok"
}

capture_prometheus_metrics() {
    python "${PROMETHEUS_VALIDATOR}" capture \
        --url "${PROXY_ENDPOINT}" \
        --output "$1"
}

smoke_streaming_metrics() {
    local mode=$1
    local stream_file stream_pid observed=0
    stream_file="$(mktemp "${TMPDIR:-/tmp}/xpyd-metrics-stream.XXXXXX")"

    curl --fail --silent --show-error --no-buffer \
        "${PROXY_ENDPOINT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL}\",
            \"prompt\": \"The streaming metrics test says\",
            \"max_tokens\": 64,
            \"temperature\": 0,
            \"ignore_eos\": true,
            \"stream\": true,
            \"stream_options\": {\"include_usage\": true}
        }" >"${stream_file}" &
    stream_pid=$!

    for _ in {1..200}; do
        if python "${PROMETHEUS_VALIDATOR}" active \
            --url "${PROXY_ENDPOINT}" --mode "${mode}" 2>/dev/null; then
            observed=1
            break
        fi
        if ! kill -0 "${stream_pid}" 2>/dev/null; then
            break
        fi
        sleep 0.05
    done
    wait "${stream_pid}"
    if ((observed == 0)); then
        rm -f "${stream_file}"
        echo "ERROR: Prometheus gauges did not expose the active request." >&2
        return 1
    fi

    python -c '
import json
import sys

events = [
    line.removeprefix("data: ")
    for line in sys.stdin.read().splitlines()
    if line.startswith("data: ")
]
assert events and events[-1] == "[DONE]", events
chunks = [json.loads(event) for event in events[:-1]]
assert any(chunk.get("choices") for chunk in chunks), chunks
usage = [chunk["usage"] for chunk in chunks if chunk.get("usage")]
assert usage and usage[-1]["completion_tokens"] == 64, chunks
' <"${stream_file}"
    rm -f "${stream_file}"
}

validate_prometheus_metrics() {
    local mode=$1 before=$2 completion_delta=$3
    shift 3
    python "${PROMETHEUS_VALIDATOR}" compare \
        --url "${PROXY_ENDPOINT}" \
        --mode "${mode}" \
        --before "${before}" \
        --completion-delta "${completion_delta}" \
        --chat-delta 1 \
        "$@"
}

smoke_informational_endpoints() {
    echo "=== Informational endpoints ==="

    local verb ping version status instances
    for verb in GET POST; do
        ping="$(
            curl --fail --silent --show-error --request "${verb}" \
                "${PROXY_ENDPOINT}/ping"
        )"
        python -c '
import json
import sys

output = json.load(sys.stdin)
assert output, "ping returned no instances"
assert any(entry.get("status") == 200 for entry in output.values()), output
' <<<"${ping}"
        echo "  ${verb} /ping ok"
    done

    version="$(
        curl --fail --silent --show-error "${PROXY_ENDPOINT}/version"
    )"
    python -c '
import json
import sys

output = json.load(sys.stdin)
assert len(output) == 1, output
entry = next(iter(output.values()))
assert entry["status"] == 200, output
assert entry["data"]["version"], output
' <<<"${version}"
    echo "  /version ok"

    status="$(
        curl --fail --silent --show-error "${PROXY_ENDPOINT}/status"
    )"
    python -c '
import json
import sys

output = json.load(sys.stdin)
assert set(output) == {
    "prefill_node_count",
    "decode_node_count",
    "prefill_nodes",
    "decode_nodes",
}, output
assert output["prefill_node_count"] == len(output["prefill_nodes"]), output
assert output["decode_node_count"] == len(output["decode_nodes"]), output
' <<<"${status}"
    echo "  /status ok"

    instances="$(
        curl --fail --silent --show-error "${PROXY_ENDPOINT}/status/instances"
    )"
    python -c '
import json
import sys

output = json.load(sys.stdin)
known = [
    instance
    for key in ("prefill_instances", "decode_instances", "aggregated_instances")
    for instance in output.get(key, [])
]
assert known, output
assert any(instance["status"] == "healthy" for instance in known), output
for instance in known:
    assert instance["circuit"] in {"closed", "open", "half_open"}, instance
    assert instance["active_requests"] >= 0, instance
' <<<"${instances}"
    echo "  /status/instances ok"
}

smoke_passthrough_endpoints() {
    echo "=== Passthrough endpoints ==="

    local tokenized tokens detokenized path
    tokenized="$(
        curl --fail-with-body --silent --show-error \
            "${PROXY_ENDPOINT}/tokenize" \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"${MODEL}\", \"prompt\": \"Hello proxy\"}"
    )"
    tokens="$(
        python -c '
import json
import sys

output = json.load(sys.stdin)
assert output["count"] > 0, output
assert len(output["tokens"]) == output["count"], output
print(json.dumps(output["tokens"]))
' <<<"${tokenized}"
    )"
    echo "  /tokenize ok (${tokens})"

    detokenized="$(
        curl --fail-with-body --silent --show-error \
            "${PROXY_ENDPOINT}/detokenize" \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"${MODEL}\", \"tokens\": ${tokens}}"
    )"
    python -c '
import json
import sys

output = json.load(sys.stdin)
assert "Hello proxy" in output["prompt"], output
' <<<"${detokenized}"
    echo "  /detokenize ok"

    assert_forwarded "/v1/embeddings" \
        "{\"model\": \"${MODEL}\", \"input\": \"Hello proxy\"}"
    assert_forwarded "/pooling" \
        "{\"model\": \"${MODEL}\", \"messages\": \"Hello proxy\"}"
    for path in /score /v1/score; do
        assert_forwarded "${path}" \
            "{\"model\": \"${MODEL}\", \"text_1\": \"a\", \"text_2\": \"b\", \"predictions\": \"\"}"
    done
    for path in /rerank /v1/rerank /v2/rerank; do
        assert_forwarded "${path}" \
            "{\"model\": \"${MODEL}\", \"query\": \"a\", \"documents\": [\"b\"]}"
    done
    assert_forwarded "/invocations" \
        "{\"model\": \"${MODEL}\", \"prompt\": \"Hello proxy\", \"max_tokens\": 1}"
}

smoke_passthrough_validation() {
    echo "=== Passthrough request validation ==="

    # The proxy rejects an incomplete body itself, before contacting a backend.
    local path
    for path in /tokenize /detokenize /v1/embeddings /pooling /score /v1/score \
        /rerank /v1/rerank /v2/rerank /invocations; do
        assert_status "${path} with missing fields" 400 \
            "$(post_status "${path}" '{}')"
        assert_status "OPTIONS ${path}" 200 "$(
            curl --silent --output /dev/null --write-out "%{http_code}" \
                --request OPTIONS "${PROXY_ENDPOINT}${path}"
        )"
    done
    echo "  all 10 passthrough endpoints validate their body and answer OPTIONS"
}

smoke_options_endpoints() {
    echo "=== OPTIONS endpoints ==="

    local path
    for path in /status /health /ping /v1/models /version \
        /v1/completions /v1/chat/completions /instances/add /instances/remove; do
        assert_status "OPTIONS ${path}" 200 "$(
            curl --silent --output /dev/null --write-out "%{http_code}" \
                --request OPTIONS "${PROXY_ENDPOINT}${path}"
        )"
    done
    echo "  all registered non-passthrough OPTIONS endpoints ok"
}

admin_request() {
    local path=$1 key=$2 body=$3
    curl --silent --output /dev/null --write-out "%{http_code}" \
        "${PROXY_ENDPOINT}${path}" \
        -H "Content-Type: application/json" \
        -H "x-api-key: ${key}" \
        -d "${body}"
}

admin_post() {
    admin_request /instances/add "$1" "$2"
}

smoke_admin_endpoint() {
    echo "=== Admin endpoint ==="

    local before after
    before="$(curl --fail --silent --show-error "${PROXY_ENDPOINT}/status")"

    assert_status "/instances/add without API key" 422 "$(
        post_status /instances/add \
            "{\"type\": \"prefill\", \"instance\": \"127.0.0.1:9100\"}"
    )"

    if [[ -n "${ADMIN_API_KEY:-}" ]]; then
        assert_status "/instances/add with a wrong API key" 403 "$(
            admin_post "definitely-wrong" \
                "{\"type\": \"prefill\", \"instance\": \"127.0.0.1:9100\"}"
        )"
        assert_status "/instances/add with an invalid role" 400 "$(
            admin_post "${ADMIN_API_KEY}" \
                "{\"type\": \"invalid\", \"instance\": \"127.0.0.1:9100\"}"
        )"
        assert_status "/instances/add with an invalid address" 400 "$(
            admin_post "${ADMIN_API_KEY}" \
                "{\"type\": \"prefill\", \"instance\": \"not-an-ip:9100\"}"
        )"
        assert_status "/instances/add with an out-of-range port" 400 "$(
            admin_post "${ADMIN_API_KEY}" \
                "{\"type\": \"prefill\", \"instance\": \"127.0.0.1:70000\"}"
        )"
        assert_status "/instances/add for an unreachable instance" 400 "$(
            admin_post "${ADMIN_API_KEY}" \
                "{\"type\": \"prefill\", \"instance\": \"127.0.0.1:9100\"}"
        )"
        assert_status "/instances/remove without API key" 422 "$(
            post_status /instances/remove \
                "{\"type\": \"decode\", \"instance\": \"127.0.0.1:9100\"}"
        )"
        assert_status "/instances/remove with a wrong API key" 403 "$(
            admin_request /instances/remove "definitely-wrong" \
                "{\"type\": \"decode\", \"instance\": \"127.0.0.1:9100\"}"
        )"
        assert_status "/instances/remove with an invalid role" 400 "$(
            admin_request /instances/remove "${ADMIN_API_KEY}" \
                "{\"type\": \"invalid\", \"instance\": \"127.0.0.1:9100\"}"
        )"
        assert_status "/instances/remove with an invalid timeout" 400 "$(
            admin_request /instances/remove "${ADMIN_API_KEY}" \
                "{\"type\": \"decode\", \"instance\": \"127.0.0.1:9100\", \"timeout_seconds\": -1}"
        )"
        assert_status "/instances/remove for an unknown instance" 404 "$(
            admin_request /instances/remove "${ADMIN_API_KEY}" \
                "{\"type\": \"decode\", \"instance\": \"127.0.0.1:9100\"}"
        )"
    else
        assert_status "/instances/add without ADMIN_API_KEY configured" 500 "$(
            admin_post "any-key" \
                "{\"type\": \"prefill\", \"instance\": \"127.0.0.1:9100\"}"
        )"
    fi

    # None of the admin calls above may mutate the instance pools.
    after="$(curl --fail --silent --show-error "${PROXY_ENDPOINT}/status")"
    [[ "${after}" == "${before}" ]] || {
        echo "ERROR: a rejected admin call changed the instance pools." >&2
        echo "before: ${before}" >&2
        echo "after:  ${after}" >&2
        return 1
    }
    echo "  rejected admin calls left the instance pools untouched"
}

smoke_admin_success() {
    local role=$1 instance=$2 before after result body status
    echo "=== Admin endpoint success path ==="

    before="$(curl --fail --silent --show-error \
        "${PROXY_ENDPOINT}/status/instances")"
    result="$(
        curl --silent --show-error --write-out $'\n%{http_code}' \
            "${PROXY_ENDPOINT}/instances/add" \
            -H "Content-Type: application/json" \
            -H "x-api-key: ${ADMIN_API_KEY}" \
            -d "{\"type\": \"${role}\", \"instance\": \"${instance}\"}"
    )"
    body="${result%$'\n'*}"
    status="${result##*$'\n'}"
    if ! assert_status "/instances/add success path" 200 "${status}"; then
        echo "response: ${body}" >&2
        return 1
    fi
    after="$(curl --fail --silent --show-error \
        "${PROXY_ENDPOINT}/status/instances")"

    BEFORE="${before}" AFTER="${after}" ROLE="${role}" INSTANCE="${instance}" \
        BODY="${body}" python - <<'PY'
import json
import os

before = json.loads(os.environ["BEFORE"])
after = json.loads(os.environ["AFTER"])
role = os.environ["ROLE"]
instance = os.environ["INSTANCE"]
body = json.loads(os.environ["BODY"])
pool_key = f"{role}_instances"
before_addresses = {item["address"] for item in before[pool_key]}
after_addresses = {item["address"] for item in after[pool_key]}
assert instance not in before_addresses, before
assert instance in after_addresses, after
assert len(after_addresses) == len(before_addresses) + 1, (before, after)
added = next(item for item in after[pool_key] if item["address"] == instance)
assert added["status"] == "healthy", added
assert body == {"message": f"Added {instance} to {role}_instances."}, body
PY
    echo "  added ${instance} to the ${role} pool"
}

smoke_admin_remove_success() {
    local role=$1 instance=$2 expected_inference=$3 result body status instances
    echo "=== Admin remove success path ==="

    result="$(
        curl --silent --show-error --write-out $'\n%{http_code}' \
            "${PROXY_ENDPOINT}/instances/remove" \
            -H "Content-Type: application/json" \
            -H "x-api-key: ${ADMIN_API_KEY}" \
            -d "{
                \"type\": \"${role}\",
                \"instance\": \"${instance}\",
                \"timeout_seconds\": 30
            }"
    )"
    body="${result%$'\n'*}"
    status="${result##*$'\n'}"
    if ! assert_status "/instances/remove success path" 200 "${status}"; then
        echo "response: ${body}" >&2
        return 1
    fi

    instances="$(curl --fail --silent --show-error \
        "${PROXY_ENDPOINT}/status/instances")"
    INSTANCES="${instances}" ROLE="${role}" INSTANCE="${instance}" \
        BODY="${body}" python - <<'PY'
import json
import os

instances = json.loads(os.environ["INSTANCES"])
role = os.environ["ROLE"]
instance = os.environ["INSTANCE"]
body = json.loads(os.environ["BODY"])
assert instance not in {
    item["address"] for item in instances[f"{role}_instances"]
}, instances
assert body == {"message": f"Removed {instance} from {role}_instances."}, body
PY

    assert_status "inference after removing ${instance}" "${expected_inference}" "$(
        post_status /v1/completions \
            "{\"model\": \"${MODEL}\", \"prompt\": \"after removal\", \"max_tokens\": 1}"
    )"
    echo "  drained and removed ${instance} from the ${role} pool"
}

smoke_admin_remove_draining() {
    local role=$1 requested_instance=$2 expected_inference=$3
    local instance="${requested_instance}"
    local stream_file remove_file stream_pid remove_pid active state new_status
    local observed
    stream_file="$(mktemp "${TMPDIR:-/tmp}/xpyd-drain-stream.XXXXXX")"
    remove_file="$(mktemp "${TMPDIR:-/tmp}/xpyd-drain-remove.XXXXXX")"
    echo "=== Admin remove draining path ==="

    curl --silent --show-error --no-buffer \
        "${PROXY_ENDPOINT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL}\",
            \"prompt\": \"drain this request\",
            \"max_tokens\": 32,
            \"temperature\": 0,
            \"ignore_eos\": true,
            \"stream\": true
        }" >"${stream_file}" &
    stream_pid=$!

    active=0
    for _ in {1..200}; do
        observed="$(
            curl --fail --silent --show-error \
                "${PROXY_ENDPOINT}/status/instances" |
                INSTANCE="${requested_instance}" ROLE="${role}" python -c '
import json
import os
import sys

role = os.environ["ROLE"]
instances = json.load(sys.stdin)[f"{role}_instances"]
requested = os.environ["INSTANCE"]
match = next(
    (
        item
        for item in instances
        if item["active_requests"] > 0
        and (not requested or item["address"] == requested)
    ),
    None,
)
print(
    "{} {}".format(match["address"], match["active_requests"])
    if match
    else "- 0"
)
'
        )"
        read -r instance active <<<"${observed}"
        if ((active > 0)); then
            break
        fi
        sleep 0.05
    done
    if ((active == 0)); then
        wait "${stream_pid}" || true
        rm -f "${stream_file}" "${remove_file}"
        echo "ERROR: did not observe an active request on ${instance}." >&2
        return 1
    fi

    curl --silent --show-error --write-out $'\n%{http_code}' \
        "${PROXY_ENDPOINT}/instances/remove" \
        -H "Content-Type: application/json" \
        -H "x-api-key: ${ADMIN_API_KEY}" \
        -d "{
            \"type\": \"${role}\",
            \"instance\": \"${instance}\",
            \"timeout_seconds\": 120
        }" >"${remove_file}" &
    remove_pid=$!

    state=""
    for _ in {1..200}; do
        state="$(
            curl --fail --silent --show-error \
                "${PROXY_ENDPOINT}/status/instances" |
                INSTANCE="${instance}" ROLE="${role}" python -c '
import json
import os
import sys

role = os.environ["ROLE"]
instances = json.load(sys.stdin)[f"{role}_instances"]
match = next(
    (item for item in instances if item["address"] == os.environ["INSTANCE"]),
    None,
)
print(match["status"] if match else "removed")
'
        )"
        if [[ "${state}" == "draining" ]]; then
            break
        fi
        sleep 0.05
    done

    if ! kill -0 "${remove_pid}" 2>/dev/null; then
        wait "${stream_pid}" || true
        wait "${remove_pid}" || true
        rm -f "${stream_file}" "${remove_file}"
        echo "ERROR: removal returned before the active request drained." >&2
        return 1
    fi

    new_status="$(
        post_status /v1/completions \
            "{\"model\": \"${MODEL}\", \"prompt\": \"new request\", \"max_tokens\": 1}"
    )"
    wait "${stream_pid}"
    wait "${remove_pid}"

    local result body status
    result="$(cat "${remove_file}")"
    body="${result%$'\n'*}"
    status="${result##*$'\n'}"
    rm -f "${stream_file}" "${remove_file}"

    [[ "${state}" == "draining" ]] || {
        echo "ERROR: expected ${instance} to enter draining, got ${state}." >&2
        return 1
    }
    assert_status \
        "new inference while ${instance} drains" \
        "${expected_inference}" \
        "${new_status}"
    assert_status "/instances/remove draining path" 200 "${status}"
    local instances
    instances="$(curl --fail --silent --show-error \
        "${PROXY_ENDPOINT}/status/instances")"
    INSTANCES="${instances}" ROLE="${role}" INSTANCE="${instance}" \
        BODY="${body}" python - <<'PY'
import json
import os

instances = json.loads(os.environ["INSTANCES"])
role = os.environ["ROLE"]
instance = os.environ["INSTANCE"]
body = json.loads(os.environ["BODY"])
assert instance not in {
    item["address"] for item in instances[f"{role}_instances"]
}, instances
assert body == {"message": f"Removed {instance} from {role}_instances."}, body
PY
    echo "  stopped new scheduling and drained ${instance} before removal"
}

smoke_all_endpoints() {
    smoke_informational_endpoints
    smoke_passthrough_endpoints
    smoke_passthrough_validation
    smoke_options_endpoints
    smoke_admin_endpoint
}
