#!/usr/bin/env bash
# -----------------------------------------------------------
# run_benchmark.sh — spin up dummy prefill/decode nodes + proxy
# and drive them with `vllm bench serve`.
#
# Topology:
#   Prefill : TP8 × DP2  → 2 instances  (ports 8100-8101)
#   Decode  : TP1 × DP16 → 16 instances (ports 8200-8215)
#   Proxy   : port 8868
# -----------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# ---------- tunables (override via env) ----------
NUM_PREFILL=${NUM_PREFILL:-2}
NUM_DECODE=${NUM_DECODE:-16}
PREFILL_BASE_PORT=${PREFILL_BASE_PORT:-8100}
DECODE_BASE_PORT=${DECODE_BASE_PORT:-8200}
PROXY_PORT=${PROXY_PORT:-8868}
MODEL_PATH=${MODEL_PATH:-tokenizers/DeepSeek-R1}

# vllm bench defaults
NUM_PROMPTS=${NUM_PROMPTS:-100}
INPUT_LEN=${INPUT_LEN:-3000}
OUTPUT_LEN=${OUTPUT_LEN:-200}
REQUEST_RATE=${REQUEST_RATE:-3.6}
BURSTINESS=${BURSTINESS:-100}

PIDS=()

cleanup() {
    echo ""
    echo ">>> Cleaning up background processes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null && wait "$pid" 2>/dev/null || true
    done
    echo ">>> Done."
}
trap cleanup EXIT INT TERM

wait_for_port() {
    local port=$1 retries=30
    for ((i=0; i<retries; i++)); do
        if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1 || \
           curl -sf "http://127.0.0.1:${port}/ping" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.3
    done
    echo "ERROR: port $port not ready after ${retries} retries" >&2
    return 1
}

# ---------- start prefill nodes ----------
PREFILL_ADDRS=()
echo ">>> Starting $NUM_PREFILL prefill nodes..."
for ((i=0; i<NUM_PREFILL; i++)); do
    port=$((PREFILL_BASE_PORT + i))
    uvicorn dummy_nodes.prefill_node:app --host 127.0.0.1 --port "$port" \
        --log-level warning &
    PIDS+=($!)
    PREFILL_ADDRS+=("127.0.0.1:${port}")
done

# ---------- start decode nodes ----------
DECODE_ADDRS=()
echo ">>> Starting $NUM_DECODE decode nodes..."
for ((i=0; i<NUM_DECODE; i++)); do
    port=$((DECODE_BASE_PORT + i))
    uvicorn dummy_nodes.decode_node:app --host 127.0.0.1 --port "$port" \
        --log-level warning &
    PIDS+=($!)
    DECODE_ADDRS+=("127.0.0.1:${port}")
done

# ---------- wait for nodes ----------
echo ">>> Waiting for nodes to be ready..."
for ((i=0; i<NUM_PREFILL; i++)); do
    wait_for_port $((PREFILL_BASE_PORT + i))
done
for ((i=0; i<NUM_DECODE; i++)); do
    wait_for_port $((DECODE_BASE_PORT + i))
done
echo ">>> All nodes ready."

# ---------- start proxy ----------
echo ">>> Starting proxy on port $PROXY_PORT..."
python3 core/MicroPDProxyServer.py \
    --model "$MODEL_PATH" \
    --prefill "${PREFILL_ADDRS[@]}" \
    --decode "${DECODE_ADDRS[@]}" \
    --port "$PROXY_PORT" &
PIDS+=($!)
sleep 2
echo ">>> Proxy started."

# ---------- run benchmark ----------
echo ">>> Running vllm bench serve..."
echo "    prompts=$NUM_PROMPTS  input_len=$INPUT_LEN  output_len=$OUTPUT_LEN"
echo "    request_rate=$REQUEST_RATE  burstiness=$BURSTINESS"

vllm bench serve \
    --host 127.0.0.1 \
    --port "$PROXY_PORT" \
    --model dummy \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --burstiness "$BURSTINESS" \
    --request-rate "$REQUEST_RATE"

echo ">>> Benchmark complete."
