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
