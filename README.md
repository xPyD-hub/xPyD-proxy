# MicroPDProxy

MicroPDProxyServer – a lightweight PD (Prefill-Decode) proxy implementation.

This project provides **dummy prefill and decode nodes** for local development
and debugging of a PD-separated proxy without any GPU or model dependencies.

The dummy nodes now expose the minimum compatibility surface required by the
validated proxy implementation under `core/`, including:

- `/v1/models`
- `/v1/completions`
- `/v1/chat/completions`
- `/health`
- `/ping`

## Architecture

MicroPDProxy implements a **Prefill-Decode (PD) separated** serving
architecture. Incoming requests are routed through two phases:

1. **Prefill** — sent to a prefill node for KV cache preparation (`max_tokens=1`, `stream=False`)
2. **Decode** — forwarded to a decode node for autoregressive token generation

The proxy handles scheduling (Round Robin or Load Balanced), health monitoring,
and dynamic instance management. See [`docs/architecture.md`](docs/architecture.md)
for the full architecture overview.

## Quick Start

```bash
pip install -r requirements.txt

# Start the dummy prefill node (port 8100)
uvicorn dummy_nodes.prefill_node:app --host 0.0.0.0 --port 8100

# Start the dummy decode node (port 8200)
uvicorn dummy_nodes.decode_node:app --host 0.0.0.0 --port 8200
```

## Usage

Useful docs:

- `docs/xpyd_start_proxy_usage.md` — parameterized script usage and topology rules
- `docs/one_click_dummy_proxy_setup.md` — end-to-end local dummy + proxy setup guide
- `docs/terminal_by_terminal_quickstart.md` — copy-paste terminal-by-terminal local setup

Both nodes expose OpenAI-compatible completion/chat-completion endpoints:

```bash
# Non-streaming request
curl http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dummy",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
  }'

# Streaming request
curl http://localhost:8200/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dummy",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10,
    "stream": true
  }'
```

## Docker Deployment

```bash
# Build and run the full local topology (2 prefill + 2 decode + proxy)
docker compose up --build

# Or run just the proxy against existing GPU nodes
docker build -t micropdproxy .
docker run -p 8868:8868 micropdproxy \
  python3 core/MicroPDProxyServer.py \
  --model tokenizers/DeepSeek-R1 \
  --prefill 10.0.0.1:8100 --decode 10.0.0.3:8200 \
  --port 8868
```

See [`docs/deployment.md`](docs/deployment.md) for production deployment details.

## Benchmark

Use vLLM's benchmark tool to test proxy throughput:

```bash
python -m vllm.entrypoints.openai.api_server  # start backend nodes first

python -m vllm bench serve \
  --base-url http://localhost:8868 \
  --model DeepSeek-R1 \
  --dataset-name sonnet \
  --sonnet-input-len 1024 \
  --sonnet-output-len 128 \
  --num-prompts 100 \
  --request-rate 10
```

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `PREFILL_DELAY_PER_TOKEN` | `0.001` | Simulated per-prompt-token prefill latency (seconds) |
| `DECODE_DELAY_PER_TOKEN` | `0.01` | Simulated per-decode-token generation latency (seconds) |

## Running Tests

```bash
pip install -r requirements.txt
python -m pytest tests/test_prefill_node.py tests/test_decode_node.py -v
python -m pytest tests/test_proxy_matrix.py -v
```

The matrix test validates the task combinations below against the real
`core/MicroPDProxyServer.py` implementation (without changing the core business
logic):

- `1 2 1`
- `2 2 1`
- `1 2 2`
- `1 2 4`
- `1 2 8`
- `2 2 2`
- `2 4 1`
- `2 4 2`

## Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Deployment Guide](docs/deployment.md)
- [Proxy Script Usage](docs/xpyd_start_proxy_usage.md)
- [Local Dummy Setup](docs/one_click_dummy_proxy_setup.md)
- [Terminal-by-Terminal Quickstart](docs/terminal_by_terminal_quickstart.md)
- [Contributing](CONTRIBUTING.md)
