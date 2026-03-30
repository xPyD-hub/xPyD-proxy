#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Run vllm bench serve against dummy prefill/decode nodes + proxy
#
# Topology:
#   - 2 prefill nodes (TP8, DP2)
#   - 16 decode nodes (TP1, DP16)
#   - 1 proxy
#
# Usage:
#   bash benchmarks/run_benchmark.sh [--num-prompts N] [--request-rate R]
#
# Defaults: --num-prompts 100 --request-rate 3.6
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL="$REPO_ROOT/tokenizers/DeepSeek-R1"
export DUMMY_MODEL_ID="$MODEL"

NUM_PROMPTS=${NUM_PROMPTS:-100}
REQUEST_RATE=${REQUEST_RATE:-3.6}

# Parse optional CLI args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
        --request-rate) REQUEST_RATE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

PIDS=()
cleanup() {
    echo "Cleaning up..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting 2 prefill nodes..."
for port in 8100 8101; do
    python3 -m uvicorn dummy_nodes.prefill_node:app \
        --host 127.0.0.1 --port "$port" --log-level error &
    PIDS+=($!)
done

echo "Starting 16 decode nodes..."
for port in $(seq 8200 8215); do
    python3 -m uvicorn dummy_nodes.decode_node:app \
        --host 127.0.0.1 --port "$port" --log-level error &
    PIDS+=($!)
done

sleep 4

echo "Starting proxy on port 8868..."
PREFILL_ARGS="127.0.0.1:8100 127.0.0.1:8101"
DECODE_ARGS=""
for p in $(seq 8200 8215); do DECODE_ARGS="$DECODE_ARGS 127.0.0.1:$p"; done

python3 core/MicroPDProxyServer.py \
    --model "$MODEL" \
    --prefill $PREFILL_ARGS \
    --decode $DECODE_ARGS \
    --port 8868 &
PIDS+=($!)

sleep 5

# Verify proxy is up
if ! curl -s http://127.0.0.1:8868/status > /dev/null; then
    echo "ERROR: Proxy failed to start"
    exit 1
fi

echo "Proxy ready. Running benchmark..."
echo "  num-prompts: $NUM_PROMPTS"
echo "  request-rate: $REQUEST_RATE"

vllm bench serve \
    --host 127.0.0.1 --port 8868 \
    --model "$MODEL" \
    --tokenizer gpt2 \
    --dataset-name random \
    --random-input-len 3000 --random-output-len 200 \
    --num-prompts "$NUM_PROMPTS" \
    --burstiness 100 \
    --request-rate "$REQUEST_RATE" \
    --endpoint /v1/completions

echo "Benchmark complete."
