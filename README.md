# MicroPDProxy

Micro implementation of PD (Prefill-Decode) Proxy Server.

This project provides **dummy prefill and decode nodes** with OpenAI-compatible
`/v1/chat/completions` endpoints, so you can develop and debug a PD-separated
proxy without any GPU or model dependencies.

## Quick Start

```bash
pip install -r requirements.txt

# Start the dummy prefill node (port 8100)
uvicorn dummy_nodes.prefill_node:app --host 0.0.0.0 --port 8100

# Start the dummy decode node (port 8200)
uvicorn dummy_nodes.decode_node:app --host 0.0.0.0 --port 8200
```

## Usage

Both nodes expose an OpenAI-compatible chat completions endpoint:

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
python -m pytest tests/ -v
```
