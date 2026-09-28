#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
NIXL_EXAMPLE="${SCRIPT_DIR}/../opt-125m-cpu-nixl"
LOG_DIR="${SCRIPT_DIR}/logs/scheduler"
PROXY_PID=""
NODE_PIDS=()
SCHEDULERS=(roundrobin loadbalanced consistent_hash power_of_two cache_aware)
export ADMIN_API_KEY=xpyd-scheduler-test-key

cleanup() {
    local pid
    for pid in "${PROXY_PID}" "${NODE_PIDS[@]-}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}"
            wait "${pid}" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT INT TERM

wait_for_url() {
    local url=$1 pid=$2 deadline=$((SECONDS + 600))
    until curl --silent --fail --max-time 2 "${url}" >/dev/null; do
        kill -0 "${pid}" 2>/dev/null || return 1
        ((SECONDS < deadline)) || return 1
        sleep 1
    done
}

wait_for_all_instances() {
    local deadline=$((SECONDS + 60))
    until curl --silent --fail http://127.0.0.1:8868/status/instances |
        python -c '
import json, sys
status = json.load(sys.stdin)
expected_p = {"127.0.0.1:8100", "127.0.0.1:8101"}
expected_p |= {"127.0.0.1:8102", "127.0.0.1:8103"}
expected_d = {"127.0.0.1:8200", "127.0.0.1:8201"}
healthy = lambda role: {
    item["address"] for item in status[f"{role}_instances"]
    if item["status"] == "healthy"
}
raise SystemExit(0 if healthy("prefill") == expected_p
                 and healthy("decode") == expected_d else 1)
'; do
        ((SECONDS < deadline)) || return 1
        sleep 1
    done
}

wait_for_proxy_close() {
    local deadline=$((SECONDS + 30))
    while curl --silent --max-time 1 http://127.0.0.1:8868/status >/dev/null; do
        ((SECONDS < deadline)) || return 1
        sleep 1
    done
}

start_node() {
    local role=$1 port=$2 side_port=$3 log=$4 pid
    echo "Starting ${role} node on port ${port}..."
    bash "${NIXL_EXAMPLE}/vllm_server.sh" \
        "${role}" "${port}" "${side_port}" >"${log}" 2>&1 &
    pid=$!
    NODE_PIDS+=("${pid}")
    if ! wait_for_url "http://127.0.0.1:${port}/health" "${pid}"; then
        echo "${role} node on port ${port} failed to become healthy."
        tail -n 50 "${log}"
        return 1
    fi
    echo "${role} node on port ${port} is healthy."
}

start_proxy() {
    local scheduler=$1 config log
    config="${SCRIPT_DIR}/scheduler_configs/${scheduler}.yaml"
    log="${LOG_DIR}/${scheduler}.log"
    echo "Starting proxy with ${scheduler} scheduler..."
    (
        cd "${ROOT}"
        exec env PYTHONPATH="${ROOT}" python -c \
            "from xpyd.proxy import main; main()" proxy --config "${config}"
    ) >"${log}" 2>&1 &
    PROXY_PID=$!
    if ! wait_for_url http://127.0.0.1:8868/status/instances "${PROXY_PID}"; then
        echo "Proxy with ${scheduler} scheduler failed to start."
        tail -n 50 "${log}"
        return 1
    fi
    echo "Proxy with ${scheduler} scheduler is ready."
}

mkdir -p "${LOG_DIR}"
: >"${LOG_DIR}/phases.log"

# The first policy starts proxy-first; offline nodes must not block startup.
start_proxy roundrobin
start_node prefill 8100 5600 "${LOG_DIR}/prefill-0.log"
start_node prefill 8101 5601 "${LOG_DIR}/prefill-1.log"
start_node prefill 8102 5602 "${LOG_DIR}/prefill-2.log"
start_node prefill 8103 5603 "${LOG_DIR}/prefill-3.log"
start_node decode 8200 5700 "${LOG_DIR}/decode-0.log"
start_node decode 8201 5701 "${LOG_DIR}/decode-1.log"

for scheduler in "${SCHEDULERS[@]}"; do
    python "${SCRIPT_DIR}/../../lib/scheduler_semantics.py" \
        --topology disaggregated --scheduler "${scheduler}"
    if [[ "${scheduler}" != "roundrobin" ]]; then
        start_proxy "${scheduler}"
    fi
    echo "=== ${scheduler} ===" | tee -a "${LOG_DIR}/phases.log"
    wait_for_all_instances
    python "${SCRIPT_DIR}/smoke_test.py" "${scheduler}"
    echo "Stopping proxy with ${scheduler} scheduler..."
    kill "${PROXY_PID}"
    wait "${PROXY_PID}" 2>/dev/null || true
    PROXY_PID=""
    wait_for_proxy_close
done

echo "Disaggregated 4P2D CPU NIXL scheduler matrix passed."
