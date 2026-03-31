# Configuration Guide

## Overview

MicroPDProxy uses YAML-based configuration as the primary way to define proxy
behavior. YAML provides a structured, readable format that scales better than
CLI arguments as the number of configuration options grows. A single YAML file
captures the entire proxy topology — model path, node addresses, tensor/data
parallelism parameters, scheduling policy, and authentication — making
deployments reproducible and version-controllable.

## YAML Schema Reference

Below is the complete schema with every supported field.

```yaml
# MicroPDProxy Configuration

# Server
model: /path/to/model           # required
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
scheduling: loadbalanced        # roundrobin | loadbalanced
generator_on_p_node: false      # whether the generator runs on prefill nodes

# Authentication
admin_api_key: ""               # env override: ADMIN_API_KEY
openai_api_key: ""              # env override: OPENAI_API_KEY

# Startup (Task 8)
startup:
  wait_timeout_seconds: 600     # max seconds to wait for 1 prefill + 1 decode
  probe_interval_seconds: 10    # health-probe interval during startup
```

### Field Details

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `model` | string | **yes** | — | Path to the model directory or tokenizer. Used by the proxy to validate requests. |
| `port` | integer | no | `8000` | TCP port the proxy listens on. |
| `log_level` | string | no | `warning` | Logging verbosity. One of `debug`, `info`, `warning`, `error`. |
| `prefill.nodes` | list[string] | **yes** | — | Addresses of prefill backend nodes in `"ip:port"` format. |
| `prefill.tp_size` | integer | **yes** | — | Tensor-parallel degree for each prefill instance. |
| `prefill.dp_size` | integer | **yes** | — | Total number of prefill data-parallel instances. |
| `prefill.world_size_per_node` | integer | **yes** | — | Number of GPUs / workers per prefill node. |
| `decode.nodes` | list[string] | **yes** | — | Addresses of decode backend nodes in `"ip:port"` format. |
| `decode.tp_size` | integer | **yes** | — | Tensor-parallel degree for each decode instance. |
| `decode.dp_size` | integer | **yes** | — | Total number of decode data-parallel instances. |
| `decode.world_size_per_node` | integer | **yes** | — | Number of GPUs / workers per decode node. |
| `scheduling` | string | no | `loadbalanced` | Scheduling policy name. Currently `roundrobin` or `loadbalanced`. |
| `generator_on_p_node` | boolean | no | `false` | If `true`, the generation loop runs on prefill nodes. |
| `admin_api_key` | string | no | `""` | API key for admin endpoints. Can also be set via `ADMIN_API_KEY` env var. |
| `openai_api_key` | string | no | `""` | API key for OpenAI-compatible endpoints. Can also be set via `OPENAI_API_KEY` env var. |
| `startup.wait_timeout_seconds` | integer | no | `600` | Maximum time (seconds) to wait for at least 1 prefill + 1 decode node during startup. |
| `startup.probe_interval_seconds` | integer | no | `10` | Interval (seconds) between health probes during startup discovery. |

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
generator_on_p_node: false
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
generator_on_p_node: false

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

Use `pdproxy --validate-config proxy.yaml` to check a config file without
starting the server.
