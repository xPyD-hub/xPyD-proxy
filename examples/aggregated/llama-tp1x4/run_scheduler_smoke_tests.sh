#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPYD_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROXY_ENDPOINT="${PROXY_ENDPOINT:-http://127.0.0.1:8868}"
SMOKE_NUM_PROMPTS="${SMOKE_NUM_PROMPTS:-32}"
RESULT_ROOT="${RESULT_ROOT:-${SCRIPT_DIR}/scheduler_results}"
SCHEDULERS=(
    roundrobin
    loadbalanced
    consistent_hash
    power_of_two
    cache_aware
)
PROXY_PID=""

cleanup() {
    if [[ -n "${PROXY_PID}" ]] && kill -0 "${PROXY_PID}" 2>/dev/null; then
        kill "${PROXY_PID}"
        wait "${PROXY_PID}" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

wait_for_proxy() {
    local deadline=$((SECONDS + 120))

    until curl --silent --fail --max-time 2 \
        "${PROXY_ENDPOINT}/status/instances" >/dev/null; do
        if ! kill -0 "${PROXY_PID}" 2>/dev/null; then
            echo "ERROR: xPyD exited before becoming ready." >&2
            return 1
        fi
        if ((SECONDS >= deadline)); then
            echo "ERROR: xPyD did not become ready within 120 seconds." >&2
            return 1
        fi
        sleep 1
    done
}

wait_for_proxy_exit() {
    local deadline=$((SECONDS + 30))

    while curl --silent --output /dev/null --max-time 1 \
        "${PROXY_ENDPOINT}/status/instances"; do
        if ((SECONDS >= deadline)); then
            echo "ERROR: xPyD did not release its port." >&2
            return 1
        fi
        sleep 1
    done
}

if curl --silent --output /dev/null --max-time 2 \
    "${PROXY_ENDPOINT}/status/instances"; then
    echo "ERROR: ${PROXY_ENDPOINT} is already running; stop it first." >&2
    exit 1
fi

mkdir -p "${RESULT_ROOT}"

for scheduler in "${SCHEDULERS[@]}"; do
    config="${SCRIPT_DIR}/xpyd_${scheduler}.yaml"
    result_dir="${RESULT_ROOT}/${scheduler}"
    proxy_log="${result_dir}/proxy.log"
    mkdir -p "${result_dir}"

    echo "=== Testing ${scheduler} ==="
    cd "${XPYD_ROOT}"
    python3 "${SCRIPT_DIR}/../../lib/scheduler_semantics.py" \
        --topology aggregated --scheduler "${scheduler}"
    PYTHONUNBUFFERED=1 bash "${SCRIPT_DIR}/../../lib/run_proxy.sh" \
        proxy -c "${config}" >"${proxy_log}" 2>&1 &
    PROXY_PID=$!
    wait_for_proxy

    RESULT_DIR="${result_dir}" \
    RESULT_FILENAME="bench.json" \
    NUM_PROMPTS="${SMOKE_NUM_PROMPTS}" \
    RANDOM_INPUT_LEN=256 \
    RANDOM_OUTPUT_LEN=32 \
    RANDOM_RANGE_RATIO=0.8 \
    REQUEST_RATE=8 \
        bash "${SCRIPT_DIR}/vllm_bench.sh" >"${result_dir}/bench.log" 2>&1

    grep -Eo 'instance=127\.0\.0\.1:800[0-3]' "${proxy_log}" \
        | sort \
        | uniq -c \
        | tee "${result_dir}/routing_counts.txt"

    kill "${PROXY_PID}"
    wait "${PROXY_PID}" 2>/dev/null || true
    PROXY_PID=""
    wait_for_proxy_exit
done

echo "All scheduler smoke tests completed."
