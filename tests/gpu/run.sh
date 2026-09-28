#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_ROOT="${GPU_TEST_LOG_DIR:-${SCRIPT_DIR}/logs}"
TIMEOUT_MINUTES="${GPU_TEST_TIMEOUT_MINUTES:-90}"
MATRIX_REQUESTS="${GPU_TEST_MATRIX_REQUESTS:-256}"
MATRIX_CONCURRENCY="${GPU_TEST_MATRIX_CONCURRENCY:-128}"
MODEL="${MODEL:-/workspace/Meta-Llama-3-8B-Instruct/}"
DEFAULT_CASES=(aggregated direct mixed)
ALL_CASES=(aggregated direct mixed lmcache-2p2d nixl-2p2d lmcache-8p8d nixl-8p8d)
SELECTED_CASES=()
RESULTS=()

declare -A CASE_SCRIPT=(
    [aggregated]="examples/aggregated/llama-tp1x4/run_all.sh"
    [direct]="examples/disaggregated/Llama_2P2D_Direct/run_all.sh"
    [mixed]="examples/mixed/llama-pd1x1-aggregated-tp1x2/run_all.sh"
    [lmcache-2p2d]="examples/disaggregated/Llama_2P2D_LMCache_ZMQ/run_all.sh"
    [nixl-2p2d]="examples/disaggregated/Llama_2P2D_NIXL/run_all.sh"
    [lmcache-8p8d]="examples/disaggregated/Llama_8P8D_LMCache_ZMQ/run_all.sh"
    [nixl-8p8d]="examples/disaggregated/Llama_8P8D_NIXL/run_all.sh"
)

declare -A CASE_DESCRIPTION=(
    [aggregated]="4-node aggregated scheduler smoke tests"
    [direct]="direct 2P2D lifecycle smoke test"
    [mixed]="mixed 1P1D and 2-node aggregated smoke test"
    [lmcache-2p2d]="LMCache ZMQ 2P2D benchmark"
    [nixl-2p2d]="NIXL 2P2D benchmark"
    [lmcache-8p8d]="LMCache ZMQ 8P8D scheduler matrix"
    [nixl-8p8d]="NIXL 8P8D scheduler matrix"
)

usage() {
    cat <<'EOF'
Usage: tests/gpu/run.sh [options]

Run local GPU integration scenarios and produce a pass/fail summary.

Options:
  --case NAME       Run one scenario; repeat to select multiple scenarios
  --all             Run every scenario, including LMCache and NIXL matrices
  --list            List available scenarios and exit
  --model PATH      Model weights directory
  --timeout MINUTES Per-scenario timeout (default: 90)
  --log-dir PATH    Result directory (default: tests/gpu/logs)
  -h, --help        Show this help

With no --case or --all, the aggregated, direct 2P2D, and mixed smoke
scenarios run. The model defaults to /workspace/Meta-Llama-3-8B-Instruct/.
EOF
}

list_cases() {
    local name
    for name in "${ALL_CASES[@]}"; do
        printf '%-16s %s\n' "${name}" "${CASE_DESCRIPTION[${name}]}"
    done
}

contains_case() {
    local wanted=$1 item
    for item in "${ALL_CASES[@]}"; do
        [[ "${item}" == "${wanted}" ]] && return 0
    done
    return 1
}

while (($#)); do
    case "$1" in
        --case)
            [[ $# -ge 2 ]] || { echo "ERROR: --case requires a name." >&2; exit 2; }
            contains_case "$2" || {
                echo "ERROR: unknown GPU test case '$2'. Use --list to see valid names." >&2
                exit 2
            }
            SELECTED_CASES+=("$2")
            shift 2
            ;;
        --all)
            SELECTED_CASES=("${ALL_CASES[@]}")
            shift
            ;;
        --list)
            list_cases
            exit 0
            ;;
        --model)
            [[ $# -ge 2 ]] || { echo "ERROR: --model requires a path." >&2; exit 2; }
            MODEL=$2
            shift 2
            ;;
        --timeout)
            [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || {
                echo "ERROR: --timeout requires a positive integer." >&2
                exit 2
            }
            TIMEOUT_MINUTES=$2
            shift 2
            ;;
        --log-dir)
            [[ $# -ge 2 ]] || { echo "ERROR: --log-dir requires a path." >&2; exit 2; }
            LOG_ROOT=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option '$1'." >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ((${#SELECTED_CASES[@]} == 0)); then
    SELECTED_CASES=("${DEFAULT_CASES[@]}")
fi

if [[ ! "${MATRIX_REQUESTS}" =~ ^[1-9][0-9]*$ ||
        ! "${MATRIX_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: GPU_TEST_MATRIX_REQUESTS and GPU_TEST_MATRIX_CONCURRENCY must be positive integers." >&2
    exit 2
fi

for command in curl nvidia-smi python3 timeout vllm; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "ERROR: required command '${command}' was not found in PATH." >&2
        exit 2
    fi
done

if [[ ! -e "${MODEL}" ]]; then
    echo "ERROR: model path does not exist: ${MODEL}" >&2
    exit 2
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if ((GPU_COUNT < 4)); then
    echo "ERROR: the GPU suite requires at least 4 GPUs; found ${GPU_COUNT}." >&2
    exit 2
fi
if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits >/dev/null; then
    echo "ERROR: unable to query active NVIDIA compute processes." >&2
    exit 2
fi

mkdir -p "${LOG_ROOT}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${LOG_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"
SUMMARY_FILE="${RUN_DIR}/summary.tsv"
printf 'case\tstatus\tduration_seconds\tlog\n' >"${SUMMARY_FILE}"

export MODEL
export SERVED_MODEL_NAME="/workspace/Meta-Llama-3-8B-Instruct/"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

gpu_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | sed -n 's/^[[:space:]]*\([0-9][0-9]*\)[[:space:]]*$/\1/p' \
        | sort -nu
}

ports_are_free() {
    python3 - "$@" <<'PY'
import socket
import sys

for value in sys.argv[1:]:
    port = int(value)
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        raise SystemExit(1)
    finally:
        sock.close()
PY
}

wait_for_cleanup() {
    local baseline_file=$1 deadline=$((SECONDS + 120))
    local current_file="${RUN_DIR}/.current-gpu-pids"

    while ((SECONDS < deadline)); do
        if ! gpu_pids >"${current_file}"; then
            sleep 2
            continue
        fi
        if ports_are_free 8868 8000 8001 8002 8003 \
                8100 8101 8102 8103 8104 8105 8106 8107 \
                8200 8201 8202 8203 8204 8205 8206 8207 \
                5600 5601 5602 5603 5604 5605 5606 5607 \
                5700 5701 5702 5703 5704 5705 5706 5707 &&
                [[ -z "$(comm -13 "${baseline_file}" "${current_file}")" ]]; then
            return 0
        fi
        sleep 2
    done

    echo "ERROR: test ports or GPU processes were not released within 120 seconds." >&2
    echo "New GPU PIDs:" >&2
    comm -13 "${baseline_file}" "${current_file}" >&2
    return 1
}

case_preflight() {
    local name=$1

    if ! ports_are_free 8868 8000 8001 8002 8003 \
            8100 8101 8102 8103 8104 8105 8106 8107 \
            8200 8201 8202 8203 8204 8205 8206 8207 \
            5600 5601 5602 5603 5604 5605 5606 5607 \
            5700 5701 5702 5703 5704 5705 5706 5707; then
        echo "ERROR: one or more test ports are already in use." >&2
        return 1
    fi

    case "${name}" in
        lmcache-*)
            python3 -c 'import lmcache' >/dev/null 2>&1 || {
                echo "ERROR: ${name} requires the Python package 'lmcache'." >&2
                return 1
            }
            ;;
        nixl-*)
            python3 -c 'import nixl' >/dev/null 2>&1 || {
                echo "ERROR: ${name} requires the Python package 'nixl'." >&2
                return 1
            }
            ;;
    esac
}

run_case() {
    local name=$1 script="${REPO_ROOT}/${CASE_SCRIPT[${name}]}"
    local log_file="${RUN_DIR}/${name}.log"
    local baseline_file="${RUN_DIR}/.${name}-gpu-pids"
    local started=${SECONDS} status="PASS" rc=0
    local -a scenario_env=()

    echo
    echo "=== ${name}: ${CASE_DESCRIPTION[${name}]} ==="
    if ! gpu_pids >"${baseline_file}"; then
        echo "ERROR: unable to capture the initial GPU process list." | tee "${log_file}"
        printf '%s\tFAIL\t0\t%s\n' "${name}" "${log_file}" >>"${SUMMARY_FILE}"
        RESULTS+=("${name}:FAIL:0")
        return 1
    fi

    if ! case_preflight "${name}" 2>&1 | tee "${log_file}"; then
        status="FAIL"
        rc=1
    else
        if [[ "${name}" == *-8p8d ]]; then
            scenario_env=(
                env
                "REQUESTS=${MATRIX_REQUESTS}"
                "CONCURRENCY=${MATRIX_CONCURRENCY}"
            )
        fi
        timeout --signal=TERM --kill-after=60s "${TIMEOUT_MINUTES}m" \
            "${scenario_env[@]}" bash "${script}" 2>&1 | tee -a "${log_file}"
        rc=${PIPESTATUS[0]}
        if ((rc != 0)); then
            status="FAIL"
            if ((rc == 124)); then
                echo "ERROR: ${name} timed out after ${TIMEOUT_MINUTES} minutes." | tee -a "${log_file}"
            fi
        fi
    fi

    if ! wait_for_cleanup "${baseline_file}" 2>&1 | tee -a "${log_file}"; then
        status="FAIL"
        rc=1
    fi

    local duration=$((SECONDS - started))
    printf '%s\t%s\t%s\t%s\n' "${name}" "${status}" "${duration}" "${log_file}" \
        >>"${SUMMARY_FILE}"
    RESULTS+=("${name}:${status}:${duration}")
    echo "=== ${name}: ${status} (${duration}s) ==="
    return "${rc}"
}

suite_status=0
for name in "${SELECTED_CASES[@]}"; do
    run_case "${name}" || suite_status=1
done

rm -f "${RUN_DIR}"/.*-gpu-pids "${RUN_DIR}/.current-gpu-pids"

echo
printf '%-18s %-6s %s\n' "CASE" "RESULT" "DURATION"
printf '%-18s %-6s %s\n' "----------------" "------" "--------"
for result in "${RESULTS[@]}"; do
    IFS=: read -r name status duration <<<"${result}"
    printf '%-18s %-6s %ss\n' "${name}" "${status}" "${duration}"
done
echo
echo "Summary: ${SUMMARY_FILE}"
exit "${suite_status}"
