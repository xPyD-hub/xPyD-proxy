# Configuration Guide

## Overview

MicroDisaggregatedProxy uses YAML-based configuration as the primary way to define proxy
behavior. YAML provides a structured, readable format that scales better than
CLI arguments as the number of configuration options grows. A single YAML file
captures the entire proxy topology — model names, tokenizer source, node addresses, tensor/data
parallelism parameters, scheduling policy, and authentication — making
deployments reproducible and version-controllable.

## YAML Schema Reference

Below is the complete schema with every supported field.

```yaml
# MicroDisaggregatedProxy Configuration

# Server
model: deepseek-ai/DeepSeek-V4-Flash  # required in legacy single-model mode
# Optional local root; model "org/name" maps to "<root>/org/name/"
# tokenizer_path: /models/tokenizers
host: 0.0.0.0                   # use 127.0.0.1 for loopback-only access
port: 8000                      # default: 8000
log_level: warning              # debug | info | warning | error

# Prefill nodes
prefill:
  nodes:                        # list of "ip:base_port" strings
    - "10.0.0.1:8100"
    - "10.0.0.2:8100"
  tp_size: 8                    # tensor-parallel size per instance
  dp_size: 2                    # data-parallel (total instances across all nodes)
  world_size_per_node: 8        # GPUs (or workers) available on each node

# Decode nodes
decode:
  nodes:
    - "10.0.0.3:8200"
    - "10.0.0.4:8200"
  tp_size: 1
  dp_size: 16
  world_size_per_node: 8

# Scheduling
scheduling: loadbalanced        # roundrobin | loadbalanced | consistent_hash | power_of_two | cache_aware
disaggregated_mode: direct     # direct | nixl | zmq
first_token_source: decode       # prefill | decode; applies to every disaggregated mode

# Authentication
admin_api_key: ""               # env override: ADMIN_API_KEY
openai_api_key: ""              # env override: OPENAI_API_KEY

# Startup (Task 8)
startup:
  wait_timeout_seconds: 600     # max seconds to wait for 1 prefill + 1 decode
  probe_interval_seconds: 10    # health-probe interval during startup
  heartbeat_interval_seconds: 30 # periodic P/D online/offline status log
```

### Field Details

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `model` | string | single-model only | — | Served model name in legacy single-model configuration. |
| `tokenizer_path` | string | no | — | Local tokenizer root. Model `org/name` must exist at `<tokenizer_path>/org/name/`. |
| `port` | integer | no | `8000` | TCP port the proxy listens on. |
| `host` | string | no | `0.0.0.0` | Listen IP address or `localhost`, configured in YAML. Use `127.0.0.1` for local-only access; IPv6 literals such as `::1` are also accepted. |
| `log_level` | string | no | `warning` | Logging verbosity. One of `debug`, `info`, `warning`, `error`. |
| `prefill.nodes` | list[string] | **yes** | — | Addresses of prefill backend nodes in `"ip:port"` format. |
| `prefill.tp_size` | integer | **yes** | — | Tensor-parallel degree for each prefill instance. |
| `prefill.dp_size` | integer | **yes** | — | Total number of prefill data-parallel instances. |
| `prefill.world_size_per_node` | integer | **yes** | — | Number of GPUs / workers per prefill node. |
| `decode.nodes` | list[string] | **yes** | — | Addresses of decode backend nodes in `"ip:port"` format. |
| `decode.tp_size` | integer | **yes** | — | Tensor-parallel degree for each decode instance. |
| `decode.dp_size` | integer | **yes** | — | Total number of decode data-parallel instances. |
| `decode.world_size_per_node` | integer | **yes** | — | Number of GPUs / workers per decode node. |
| `scheduling` | string | no | `loadbalanced` | Registered policy name: `roundrobin`, `loadbalanced`, `consistent_hash`, `power_of_two`, or `cache_aware`. See [Scheduling Policies](scheduling.md) for options and the extension interface. |
| `disaggregated_mode` | string | no | `direct` | KV transfer mode. `direct`, `nixl`, and `zmq` are supported. |
| `first_token_source` | string | no | `decode` | Backend that provides the first client-visible token. `prefill` and `decode` are supported uniformly in direct, NIXL, and ZMQ modes. |
| `admin_api_key` | string | no | `""` | API key for admin endpoints. Can also be set via `ADMIN_API_KEY` env var. |
| `openai_api_key` | string | no | `""` | API key for OpenAI-compatible endpoints. Can also be set via `OPENAI_API_KEY` env var. |
| `startup.wait_timeout_seconds` | integer | no | `600` | Maximum time (seconds) to wait for at least 1 prefill + 1 decode node during startup. |
| `startup.probe_interval_seconds` | integer | no | `10` | Interval (seconds) between health probes during startup discovery. |
| `startup.heartbeat_interval_seconds` | number | no | `30` | Interval between logs of the deployment mode and online P/D node counts. |

### Tokenizer loading and scheduler fallback

When `tokenizer_path` is set, xPyD strictly loads each model from its
model-named subdirectory. For example, model `facebook/opt-125m` must have
valid Hugging Face tokenizer files in
`<tokenizer_path>/facebook/opt-125m/`. A missing directory or invalid
tokenizer is a configuration error and stops the proxy with the expected
path in the error message.

Without `tokenizer_path`, xPyD downloads each discovered model's tokenizer
through `AutoTokenizer.from_pretrained()`. Successful loads use the configured
scheduler, which defaults to `loadbalanced`. A failed download emits a warning
and downgrades only that model to `roundrobin`. P/D modes continue operating;
when ZMQ needs exact token IDs, xPyD uses a healthy prefill backend's
`/tokenize` endpoint.

### ZMQ receiver token alignment

P-first requires `skip_last_n_tokens: 1` and
`discard_partial_chunks: false` on every `zmq.receivers` entry and in the
decoder's LMCache `kv_connector_extra_config`.

ZMQ D-first always sends the request through the selected prefill and waits
for its matching `ProxyNotif` before forwarding to the selected decoder. The
prefill HTTP response is not a transfer-completion barrier: LMCache submits
`batched_put` asynchronously, while `ProxyNotif` is only emitted after a
successful last-prefill write.

## Topology Expansion

The proxy expands node lists into individual instance addresses using the
topology parameters. The formula is:

```
instances_per_node = world_size_per_node / tp_size
total_instances    = len(nodes) * instances_per_node
```

The `total_instances` value must equal `dp_size`.

Each instance is assigned a port offset from the node's base port:

```
instance_port = base_port + (index * tp_size)
```

where `index` ranges from `0` to `instances_per_node - 1`.

### Example

Consider the following configuration:

```yaml
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
```

**Prefill expansion:**

- 2 nodes × (8 / 8) = 2 instances total (DP=2 ✓)
- `10.0.0.1:8100` → 1 instance: `10.0.0.1:8100`
- `10.0.0.2:8100` → 1 instance: `10.0.0.2:8100`

**Decode expansion:**

- 2 nodes × (8 / 1) = 16 instances total (DP=16 ✓)
- `10.0.0.3:8200` → 8 instances: ports `8200, 8201, 8202, …, 8207`
- `10.0.0.4:8200` → 8 instances: ports `8200, 8201, 8202, …, 8207`

## Configuration Precedence

When the same setting can be specified in multiple places, the proxy resolves
values in the following order (highest priority first):

1. **CLI arguments** — e.g. `--port 9000`
2. **Environment variables** — e.g. `ADMIN_API_KEY=secret`
3. **YAML config file** — the file passed via `--config` / `-c`
4. **Built-in defaults** — hardcoded fallback values

This means a CLI flag always wins over an environment variable, which in turn
wins over whatever is written in the YAML file.

## Example Configurations

### Minimal Configuration

The smallest valid config — a single prefill node and a single decode node:

```yaml
model: /models/llama-7b

prefill:
  nodes:
    - "127.0.0.1:8100"
  tp_size: 1
  dp_size: 1
  world_size_per_node: 1

decode:
  nodes:
    - "127.0.0.1:8200"
  tp_size: 1
  dp_size: 1
  world_size_per_node: 1
```

### Development Configuration

Useful for local development with dummy nodes:

```yaml
model: /models/llama-7b
port: 8000
log_level: debug

prefill:
  nodes:
    - "127.0.0.1:8100"
  tp_size: 1
  dp_size: 2
  world_size_per_node: 2

decode:
  nodes:
    - "127.0.0.1:8200"
  tp_size: 1
  dp_size: 4
  world_size_per_node: 4

scheduling: roundrobin
first_token_source: decode
```

### Production Configuration

Multi-node deployment with authentication and load balancing:

```yaml
model: /shared-nfs/models/llama-70b
port: 443
log_level: warning

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
first_token_source: decode

admin_api_key: "${ADMIN_API_KEY}"
openai_api_key: "${OPENAI_API_KEY}"

startup:
  wait_timeout_seconds: 300
  probe_interval_seconds: 5
```

## Validation

The proxy validates the configuration at startup and rejects invalid configs
with a descriptive error message. The following rules are enforced:

| Rule | Error Message |
|---|---|
| `model` must be a non-empty string | `"model" is required` |
| `port` must be an integer between 1 and 65535 | `"port" must be between 1 and 65535` |
| `log_level` must be one of `debug`, `info`, `warning`, `error` | `"log_level" must be one of: debug, info, warning, error` |
| `prefill.nodes` must be a non-empty list | `"prefill.nodes" must be a non-empty list` |
| `decode.nodes` must be a non-empty list | `"decode.nodes" must be a non-empty list` |
| Each node address must match `host:port` format | `Invalid node address: "{addr}" — expected "host:port"` |
| `tp_size` must be a positive integer | `"prefill.tp_size" must be a positive integer` |
| `dp_size` must be a positive integer | `"prefill.dp_size" must be a positive integer` |
| `world_size_per_node` must be a positive integer | `"prefill.world_size_per_node" must be a positive integer` |
| `world_size_per_node` must be divisible by `tp_size` | `"prefill.world_size_per_node" (8) must be divisible by "prefill.tp_size" (3)` |
| `len(nodes) * (world_size_per_node / tp_size)` must equal `dp_size` | `Topology mismatch: 2 nodes × (8 / 8) = 2, but dp_size = 4` |
| `scheduling` must be a recognized policy name | `Unknown scheduling policy: "foo"` |
| `startup.wait_timeout_seconds` must be a positive integer | `"startup.wait_timeout_seconds" must be a positive integer` |
| `startup.probe_interval_seconds` must be a positive integer | `"startup.probe_interval_seconds" must be a positive integer` |

Use `xpyd --validate-config proxy.yaml` to check a config file without
starting the server.
