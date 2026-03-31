# MicroPDProxy

MicroPDProxyServer – a lightweight PD (Prefill-Decode) proxy implementation.

This project provides **dummy prefill and decode nodes** for local development
and debugging of a PD-separated proxy without any GPU or model dependencies.

The dummy nodes expose the minimum compatibility surface required by the
validated proxy implementation under `core/`, including:

- `/v1/models`
- `/v1/completions`
- `/v1/chat/completions`
- `/health`
- `/ping`
- `/metrics` (Prometheus format)

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

### Option 1: YAML Configuration (recommended)

Create a YAML config file (see [`examples/proxy.yaml`](examples/proxy.yaml)):

```yaml
model: /path/to/model
port: 8868

prefill:
  nodes:
    - "10.0.0.1:8100"
    - "10.0.0.2:8100"
  tp_size: 8
  dp_size: 2
  world_size_per_node: 8

decode:
  nodes:
    - "10.0.0.3:8200"
    - "10.0.0.4:8200"
  tp_size: 1
  dp_size: 16
  world_size_per_node: 8

scheduling: loadbalanced
```

Start the proxy:

```bash
python core/MicroPDProxyServer.py --config proxy.yaml
```

The topology parameters expand into instance addresses automatically:
- **Prefill**: 2 nodes × (8 / 8) = 1 instance/node = 2 instances
- **Decode**: 2 nodes × (8 / 1) = 8 instances/node = 16 instances

A simple flat-list format is also supported (see [`examples/proxy-simple.yaml`](examples/proxy-simple.yaml)):

```yaml
model: /path/to/model
prefill:
  - "10.0.0.1:8100"
decode:
  - "10.0.0.2:8200"
  - "10.0.0.3:8200"
```

### Option 2: CLI Arguments

```bash
python core/MicroPDProxyServer.py \
  --model /path/to/model \
  --prefill 10.0.0.1:8100 10.0.0.2:8100 \
  --decode 10.0.0.3:8200 10.0.0.4:8200 \
  --port 8868 \
  --roundrobin
```

### Option 3: Parameterized Shell Script

For topology-driven deployments with TP/DP parameters:

```bash
bash core/xpyd_start_proxy.sh \
  --model /path/to/model \
  --prefill-nodes 2 --prefill-tp-size 8 --prefill-dp-size 2 --prefill-world-size-per-node 8 \
  --decode-nodes 2 --decode-tp-size 1 --decode-dp-size 16 --decode-world-size-per-node 8 \
  --prefill-base-port 8100 --decode-base-port 8200
```

### CLI Arguments Reference

| Argument | Short | Default | Description |
|---|---|---|---|
| `--config` | `-c` | — | Path to YAML configuration file |
| `--model` | `-m` | — | Model name / path (required unless in YAML) |
| `--prefill` | `-p` | — | Prefill node URLs (host:port) |
| `--decode` | `-d` | — | Decode node URLs (host:port) |
| `--port` | — | 8000 | Proxy listen port |
| `--roundrobin` | — | false | Use round-robin scheduling |
| `--generator_on_p_node` | — | false | Generate first token on prefill node |

When both `--config` and CLI arguments are provided, CLI arguments take precedence.

### YAML Config Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | — | Model name / path (required) |
| `port` | int | 8000 | Proxy listen port |
| `log_level` | string | warning | Log level: debug, info, warning, error |
| `prefill` | list or topology | [] | Prefill node config |
| `decode` | list or topology | — | Decode node config (required) |
| `scheduling` | string | loadbalanced | Scheduling policy: roundrobin, loadbalanced |
| `generator_on_p_node` | bool | false | Generate first token on prefill node |
| `admin_api_key` | string | — | Admin API key (env `ADMIN_API_KEY` overrides) |
| `openai_api_key` | string | — | OpenAI API key (env `OPENAI_API_KEY` overrides) |

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
| `ADMIN_API_KEY` | — | API key for admin endpoints (overrides YAML) |
| `OPENAI_API_KEY` | — | Bearer token for backend nodes (overrides YAML) |

## Running Tests

```bash
pip install -r requirements.txt

# Run the full test suite
PYTHONPATH=core:tests python -m pytest tests/ -v

# Run specific test groups
PYTHONPATH=core:tests python -m pytest tests/test_prefill_node.py tests/test_decode_node.py -v  # Node tests
PYTHONPATH=core:tests python -m pytest tests/test_proxy_matrix.py -v                            # Topology matrix
PYTHONPATH=core:tests python -m pytest tests/test_yaml_integration.py -v                        # YAML config integration
PYTHONPATH=core:tests python -m pytest tests/test_config.py tests/test_yaml_config.py -v        # Config validation
PYTHONPATH=core:tests python -m pytest tests/test_topology.py -v                                # Topology expansion
PYTHONPATH=core:tests python -m pytest tests/test_scheduler.py -v                               # Scheduler unit tests
PYTHONPATH=core:tests python -m pytest tests/test_metrics.py -v                                 # Prometheus metrics
```

## Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Deployment Guide](docs/deployment.md)
- [Proxy Script Usage](docs/xpyd_start_proxy_usage.md)
- [Local Dummy Setup](docs/one_click_dummy_proxy_setup.md)
- [Terminal-by-Terminal Quickstart](docs/terminal_by_terminal_quickstart.md)
- [Contributing](CONTRIBUTING.md)
