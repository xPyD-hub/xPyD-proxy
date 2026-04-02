#!/usr/bin/env bash

set -euo pipefail

# For OAM
DEFAULT_DECODE_IPS=("10.239.129.81" "10.239.129.165" "10.239.129.67" "10.239.129.21")
DEFAULT_PREFILL_IPS=("10.239.129.9" "10.239.129.67" "10.239.129.21" "10.239.128.165" "10.239.128.244" "10.239.128.153")

if [[ -n "${XPYD_DECODE_IPS:-}" ]]; then
    read -r -a DECODE_IPS <<<"$XPYD_DECODE_IPS"
else
    DECODE_IPS=("${DEFAULT_DECODE_IPS[@]}")
fi

if [[ -n "${XPYD_PREFILL_IPS:-}" ]]; then
    read -r -a PREFILL_IPS <<<"$XPYD_PREFILL_IPS"
else
    PREFILL_IPS=("${DEFAULT_PREFILL_IPS[@]}")
fi

DEFAULT_PREFILL_BASE_PORT=8100
DEFAULT_DECODE_BASE_PORT=8200
DEFAULT_PROXY_PORT=${XPYD_PROXY_PORT:-8868}
DEFAULT_MODE="advanced"

usage() {
    cat <<'EOF'
Usage:
  bash xpyd_start_proxy.sh \
    --prefill-nodes <n> --prefill-tp-size <n> --prefill-dp-size <n> --prefill-world-size-per-node <n> \
    --decode-nodes <n>  --decode-tp-size <n>  --decode-dp-size <n>  --decode-world-size-per-node <n> \
    [--prefill-base-port <n>] [--decode-base-port <n>] [--model <path>] [--mode advanced|basic|benchmark|benchmark_decode]

Short aliases:
  -pn  -pt  -pd  -pw
  -dn  -dt  -dd  -dw

Notes:
  - --model can also be set via model_path env var (CLI takes precedence).
  - tp_size and dp_size must be powers of two.
  - tp_size * dp_size must equal nodes * world_size_per_node.
  - One instance = one TP group.
  - Instance count = dp_size.
  - One instance exposes only one IP:PORT to proxy.
EOF
}

error() {
    echo "ERROR: $*" >&2
    exit 1
}

is_positive_int() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_power_of_two() {
    local value=$1
    (( value > 0 && (value & (value - 1)) == 0 ))
}

require_value() {
    local flag=$1
    local value=${2:-}
    if [[ -z "$value" ]]; then
        error "Missing value for $flag"
    fi
}

validate_positive_int() {
    local name=$1
    local value=$2
    if ! is_positive_int "$value"; then
        error "$name must be a positive integer, got: $value"
    fi
}

validate_power_of_two() {
    local name=$1
    local value=$2
    if ! is_power_of_two "$value"; then
        error "$name must be a power of two, got: $value"
    fi
}

validate_topology() {
    local role=$1
    local nodes=$2
    local tp_size=$3
    local dp_size=$4
    local world_size_per_node=$5
    local ip_count=$6

    validate_positive_int "$role nodes" "$nodes"
    validate_positive_int "$role tp_size" "$tp_size"
    validate_positive_int "$role dp_size" "$dp_size"
    validate_positive_int "$role world_size_per_node" "$world_size_per_node"

    validate_power_of_two "$role tp_size" "$tp_size"
    validate_power_of_two "$role dp_size" "$dp_size"

    if (( tp_size * dp_size != nodes * world_size_per_node )); then
        error "$role topology invalid: tp_size * dp_size must equal nodes * world_size_per_node"
    fi

    if (( nodes > ip_count )); then
        error "$role nodes exceeds available IP list length ($nodes > $ip_count)"
    fi
}

build_instance_endpoints() {
    local role=$1
    local nodes=$2
    local tp_size=$3
    local dp_size=$4
    local world_size_per_node=$5
    local base_port=$6
    local array_name=$7
    local -n ip_list="$array_name"

    local -a node_instance_counts=()
    local -a endpoints=()
    local instance_index start_rank main_node local_index port

    for ((instance_index = 0; instance_index < dp_size; instance_index++)); do
        start_rank=$((instance_index * tp_size))
        main_node=$((start_rank / world_size_per_node))

        if (( main_node >= nodes )); then
            error "$role topology expansion failed: computed main node index $main_node out of range"
        fi

        local_index=${node_instance_counts[$main_node]:-0}
        port=$((base_port + local_index))
        node_instance_counts[$main_node]=$((local_index + 1))
        endpoints+=("${ip_list[$main_node]}:${port}")
    done

    printf '%s ' "${endpoints[@]}"
}

PREFILL_NODES=""
PREFILL_TP_SIZE=""
PREFILL_DP_SIZE=""
PREFILL_WORLD_SIZE_PER_NODE=""
DECODE_NODES=""
DECODE_TP_SIZE=""
DECODE_DP_SIZE=""
DECODE_WORLD_SIZE_PER_NODE=""
PREFILL_BASE_PORT=$DEFAULT_PREFILL_BASE_PORT
DECODE_BASE_PORT=$DEFAULT_DECODE_BASE_PORT
MODE=$DEFAULT_MODE
MODEL_PATH="${model_path:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefill-nodes|-pn)
            require_value "$1" "${2:-}"
            PREFILL_NODES=$2
            shift 2
            ;;
        --prefill-tp-size|-pt)
            require_value "$1" "${2:-}"
            PREFILL_TP_SIZE=$2
            shift 2
            ;;
        --prefill-dp-size|-pd)
            require_value "$1" "${2:-}"
            PREFILL_DP_SIZE=$2
            shift 2
            ;;
        --prefill-world-size-per-node|-pw)
            require_value "$1" "${2:-}"
            PREFILL_WORLD_SIZE_PER_NODE=$2
            shift 2
            ;;
        --decode-nodes|-dn)
            require_value "$1" "${2:-}"
            DECODE_NODES=$2
            shift 2
            ;;
        --decode-tp-size|-dt)
            require_value "$1" "${2:-}"
            DECODE_TP_SIZE=$2
            shift 2
            ;;
        --decode-dp-size|-dd)
            require_value "$1" "${2:-}"
            DECODE_DP_SIZE=$2
            shift 2
            ;;
        --decode-world-size-per-node|-dw)
            require_value "$1" "${2:-}"
            DECODE_WORLD_SIZE_PER_NODE=$2
            shift 2
            ;;
        --prefill-base-port)
            require_value "$1" "${2:-}"
            PREFILL_BASE_PORT=$2
            shift 2
            ;;
        --decode-base-port)
            require_value "$1" "${2:-}"
            DECODE_BASE_PORT=$2
            shift 2
            ;;
        --mode|-m)
            require_value "$1" "${2:-}"
            MODE=$2
            shift 2
            ;;
        --model)
            require_value "$1" "${2:-}"
            MODEL_PATH="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            error "Unknown argument: $1"
            ;;
    esac
done

for required_name in \
    PREFILL_NODES PREFILL_TP_SIZE PREFILL_DP_SIZE PREFILL_WORLD_SIZE_PER_NODE \
    DECODE_NODES DECODE_TP_SIZE DECODE_DP_SIZE DECODE_WORLD_SIZE_PER_NODE; do
    if [[ -z "${!required_name}" ]]; then
        error "Missing required argument: ${required_name,,}. Run with --help for usage."
    fi
done

validate_positive_int "prefill base port" "$PREFILL_BASE_PORT"
validate_positive_int "decode base port" "$DECODE_BASE_PORT"

validate_topology "prefill" \
    "$PREFILL_NODES" \
    "$PREFILL_TP_SIZE" \
    "$PREFILL_DP_SIZE" \
    "$PREFILL_WORLD_SIZE_PER_NODE" \
    "${#PREFILL_IPS[@]}"

validate_topology "decode" \
    "$DECODE_NODES" \
    "$DECODE_TP_SIZE" \
    "$DECODE_DP_SIZE" \
    "$DECODE_WORLD_SIZE_PER_NODE" \
    "${#DECODE_IPS[@]}"

if [[ -z "$MODEL_PATH" ]]; then
    error "model path is not set. Use --model <path> or set model_path env var."
fi

PREFILL_ARGS=$(build_instance_endpoints \
    "prefill" \
    "$PREFILL_NODES" \
    "$PREFILL_TP_SIZE" \
    "$PREFILL_DP_SIZE" \
    "$PREFILL_WORLD_SIZE_PER_NODE" \
    "$PREFILL_BASE_PORT" \
    PREFILL_IPS)

DECODE_ARGS=$(build_instance_endpoints \
    "decode" \
    "$DECODE_NODES" \
    "$DECODE_TP_SIZE" \
    "$DECODE_DP_SIZE" \
    "$DECODE_WORLD_SIZE_PER_NODE" \
    "$DECODE_BASE_PORT" \
    DECODE_IPS)

case "$MODE" in
    benchmark)
        CMD="python3 -m xpyd.proxy \
        --model $MODEL_PATH \
        --prefill $PREFILL_ARGS \
        --decode $DECODE_ARGS \
        --port $DEFAULT_PROXY_PORT \
        --repeat_p_request 1 \
        --repeat_d_times 639 \
        --benchmark_mode"
        ;;
    advanced|basic|benchmark_decode)
        CMD="python3 -m xpyd.proxy \
        --model $MODEL_PATH \
        --prefill $PREFILL_ARGS \
        --decode $DECODE_ARGS \
        --port $DEFAULT_PROXY_PORT"
        ;;
    *)
        error "Unsupported mode: $MODE"
        ;;
esac

if [[ -n "${XPYD_LOG:-}" ]]; then
    timestamp=$(date +"%Y%m%d_%H%M%S")
    log_file="$XPYD_LOG/ProxyServer_${timestamp}.log"
    CMD="$CMD 2>&1 | tee $log_file"
fi

echo "Running: $CMD"

if [[ "${XPYD_DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

eval "$CMD"
