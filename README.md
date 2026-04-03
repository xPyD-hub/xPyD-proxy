# xPyD Proxy

A lightweight Prefill-Decode (PD) proxy for disaggregated LLM serving.

## Architecture

xPyD Proxy supports two operating modes:

### Prefill-Decode (P/D) Disaggregated Mode

Requests are routed through two phases with KV cache transfer:

1. **Prefill** — KV cache preparation on prefill nodes (`max_tokens=1`)
2. **Decode** — autoregressive token generation on decode nodes (receives KV cache from prefill)

### Dual-Role Mode

A single instance handles both prefill and decode in one pass — no KV transfer needed. This simplifies deployment when disaggregation is not required or for smaller-scale setups.

The proxy handles scheduling (load-balanced, round-robin, consistent hash, power-of-two, cache-aware), health monitoring, circuit breaking, and dynamic instance management. Multi-model routing allows serving multiple models through a single proxy with per-model scheduler configuration.

See [`docs/architecture.md`](docs/architecture.md) for details.

## Installation

```bash
pip install .

# Verify
xpyd --version
```

## Quick Start

```bash
# Generate a config template
xpyd proxy --init-config

# Edit xpyd.yaml with your model and node addresses, then:
xpyd proxy -c xpyd.yaml
```

## Configuration

All configuration is done via YAML. Three config formats are supported.

### Format 1: Legacy (Single Model)

Simple prefill/decode address lists for a single model:

```yaml
model: /path/to/model
prefill:
  - "10.0.0.3:8100"
decode:
  - "10.0.0.1:8200"
  - "10.0.0.2:8200"
port: 8000
scheduling: loadbalanced
```

Topology-style config is also supported in Format 1:

```yaml
model: /path/to/model
port: 8868

prefill:
  nodes:
    - "10.0.0.1:8100"
  tp_size: 8
  dp_size: 1
  world_size_per_node: 8

decode:
  nodes:
    - "10.0.0.2:8200"
    - "10.0.0.3:8200"
  tp_size: 1
  dp_size: 16
  world_size_per_node: 8
```

### Format 2: Instances (Multi-Model, Per-Instance Role)

Explicit per-instance configuration with role and model assignment. Supports `dual` role:

```yaml
instances:
  - address: "10.0.0.1:8000"
    role: prefill
    model: llama-3
  - address: "10.0.0.2:8000"
    role: decode
    model: llama-3
  - address: "10.0.0.3:8000"
    role: dual
    model: qwen-2
port: 8000
scheduling: loadbalanced
```

### Format 3: Models Shorthand (Multi-Model, Per-Model Scheduler)

Compact format with per-model scheduler override and `dual` shorthand:

```yaml
models:
  - name: llama-3
    prefill:
      - "10.0.0.1:8000"
    decode:
      - "10.0.0.2:8000"
    scheduler: round_robin
  - name: qwen-2
    dual:
      - "10.0.0.3:8000"
      - "10.0.0.4:8000"
    scheduler: loadbalanced
port: 8000
```

> **Note:** `instances` and `models` cannot be combined. Legacy `prefill`/`decode` lists cannot be used with `instances` or `models`.

See [`examples/proxy.yaml`](examples/proxy.yaml) for a fully-commented example.

### CLI Reference

```
xpyd proxy [OPTIONS]

Options:
  --config, -c PATH         Path to YAML config (default: ./xpyd.yaml or XPYD_CONFIG env)
  --init-config [PATH]      Generate a config template and exit
  --validate-config PATH    Validate a config file and exit
  --port PORT               Override port from config
  --log-level LEVEL         Override log level: debug|info|warning|error
  --version, -V             Show version and exit
```

```
xpyd fix-config CONFIG_PATH [OPTIONS]

Auto-fix common config mistakes (typos, missing ports, whitespace).

Arguments:
  CONFIG_PATH               Path to YAML config file to fix

Options:
  --write                   Write fixes back to file (creates timestamped .bak backup).
                            Note: does not preserve YAML comments or formatting.
  --interactive             Prompt for confirmation on ambiguous suggestions
```

### Config resolution order

1. `--config` / `-c` CLI argument
2. `XPYD_CONFIG` environment variable
3. `./xpyd.yaml` in the current directory

### YAML Config

```yaml
# Required
model: /path/to/model
decode:
  - "10.0.0.1:8200"
  - "10.0.0.2:8200"

# Optional
prefill:
  - "10.0.0.3:8100"
port: 8000
log_level: warning
scheduling: loadbalanced   # roundrobin | loadbalanced | consistent_hash | power_of_two | cache_aware
generator_on_p_node: false
```

See [`examples/proxy.yaml`](examples/proxy.yaml) for a fully-commented example.

### YAML Fields Reference

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | — | Model name / path (required in Format 1) |
| `port` | int | 8000 | Proxy listen port |
| `log_level` | string | warning | Log level: debug, info, warning, error |
| `prefill` | list or topology | [] | Prefill node config (Format 1) |
| `decode` | list or topology | — | Decode node config (Format 1, required) |
| `instances` | list | — | Per-instance config (Format 2): `{address, role, model}` |
| `models` | list | — | Per-model shorthand (Format 3): `{name, prefill, decode, dual, scheduler}` |
| `scheduling` | string | loadbalanced | Global scheduling policy |
| `scheduling_config` | dict | {} | Policy-specific options |
| `generator_on_p_node` | bool | false | Generate first token on prefill node |
| `admin_api_key` | string | — | Admin API key (env `ADMIN_API_KEY` overrides) |
| `openai_api_key` | string | — | OpenAI API key (env `OPENAI_API_KEY` overrides) |
| `startup.wait_timeout_seconds` | int | 600 | Max wait for nodes at startup |
| `startup.probe_interval_seconds` | int | 10 | Health probe interval |

**Valid `role` values:** `prefill`, `decode`, `dual`

**Valid `scheduling` values:** `loadbalanced`, `roundrobin` (alias: `round_robin`), `load_balanced`, `consistent_hash`, `power_of_two`, `cache_aware`

### API

The proxy exposes an OpenAI-compatible API:

- **`POST /v1/chat/completions`** — Chat completions (streaming and non-streaming)
- **`GET /v1/models`** — List all registered models in OpenAI-compatible format

### Startup Node Discovery

The proxy returns **503** on business endpoints until the minimum instance requirement is met: at least **1 prefill + 1 decode** node, or **1 dual** node per model must respond healthy. Health/status/metrics endpoints are always available.

## Docker

```bash
# Full local topology (prefill + decode + proxy)
docker compose up --build

# Proxy only, connecting to existing GPU nodes
docker build -t xpyd .
docker run -p 8868:8868 -v ./config.yaml:/app/xpyd.yaml xpyd
```

See [`docs/deployment.md`](docs/deployment.md) for production deployment.

## Benchmark

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

## Development

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run tests
python -m pytest tests/unit/ tests/integration/ -v

# Lint
pre-commit run --all-files
```

### Environment Variables

| Variable | Description |
|---|---|
| `XPYD_CONFIG` | Default config file path |
| `ADMIN_API_KEY` | Admin API key (overrides YAML) |
| `OPENAI_API_KEY` | Bearer token for backend nodes (overrides YAML) |
| `PREFILL_DELAY_PER_TOKEN` | Simulated prefill latency for dummy nodes (default: 0.001s) |
| `DECODE_DELAY_PER_TOKEN` | Simulated decode latency for dummy nodes (default: 0.01s) |

## Documentation

| Document | Description |
|---|---|
| [Architecture](docs/architecture.md) | System architecture overview |
| [API Reference](docs/api_reference.md) | HTTP API endpoints |
| [Configuration](docs/configuration.md) | YAML config reference |
| [CLI](docs/cli.md) | xpyd command-line tool |
| [Scheduling](docs/scheduling.md) | Load balancing strategies |
| [Resilience](docs/resilience.md) | Health checks, circuit breakers, retry |
| [Metrics](docs/metrics.md) | Prometheus metrics endpoint |
| [Deployment](docs/deployment.md) | Deployment and Docker guide |
| [Contributing](CONTRIBUTING.md) | Contribution guidelines |

## License

Apache-2.0
