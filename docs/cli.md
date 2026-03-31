# CLI Reference *(planned)*

> **Status:** CLI packaging is in progress (Task 8). This document describes
> the planned interface.

## Installation

```bash
pip install .
```

For development:

```bash
pip install -e .
```

After installation, the `pdproxy` command is available system-wide.

## Quick Start

```bash
pdproxy -c proxy.yaml
```

This starts the proxy using the specified YAML configuration file. The proxy
will begin startup node discovery, probing configured backend nodes until at
least one prefill and one decode node are healthy, then start accepting
requests.

## CLI Reference

| Flag / Env | Description |
|---|---|
| `-c`, `--config FILE` | Path to YAML configuration file. |
| `--help` | Show help message and exit. |
| `--version` | Show version number and exit. |
| `--validate-config FILE` | Validate a YAML config file without starting the server. Exits with code 0 if valid, non-zero with error details if invalid. |
| `PDPROXY_CONFIG` | Environment variable alternative to `--config`. |

### Legacy CLI Arguments

For backward compatibility, the existing CLI arguments (`--model`, `--prefill`,
`--decode`, etc.) continue to work. However, YAML configuration is the
recommended approach for new deployments.

## Startup Node Discovery

When `pdproxy` starts, the following sequence occurs:

```
1. Parse configuration (CLI → env → YAML → defaults)
2. Start uvicorn (port opens immediately)
3. Return 503 "waiting for backend nodes" for all business requests
4. Background task: probe all configured nodes every <probe_interval_seconds>
5. As nodes respond healthy, add them to the scheduling pool
   Log: "[3/16 decode nodes ready]"
6. Once ≥1 prefill + ≥1 decode are ready:
   Log: "Proxy ready: N prefill, M decode nodes available"
   → Start accepting requests (200 OK)
7. If <wait_timeout_seconds> expires without 1P+1D → exit with error
```

This design ensures:

- The proxy port is reachable immediately (load balancers see it as "up").
- No requests are lost — clients receive a clear 503 until backends are ready.
- Nodes can start in any order; the proxy discovers them dynamically.

## Configuration Resolution

The proxy resolves configuration from multiple sources in the following
precedence order (highest priority first):

| Priority | Source | Example |
|---|---|---|
| 1 (highest) | CLI arguments | `pdproxy --port 9000` |
| 2 | Environment variables | `PDPROXY_CONFIG=proxy.yaml` |
| 3 | YAML config file | `port: 8000` in proxy.yaml |
| 4 (lowest) | Built-in defaults | port defaults to 8000 |

See [Configuration Guide](configuration.md) for the full YAML schema.

## Examples

### Start with a config file

```bash
pdproxy -c /etc/pdproxy/production.yaml
```

### Start with environment variable

```bash
export PDPROXY_CONFIG=/etc/pdproxy/production.yaml
pdproxy
```

### Validate configuration without starting

```bash
pdproxy --validate-config proxy.yaml
# Output: Configuration is valid.
# Exit code: 0
```

```bash
pdproxy --validate-config bad.yaml
# Output: Configuration error: "model" is required
# Exit code: 1
```

### Use default config file

If no `--config` or `PDPROXY_CONFIG` is set, `pdproxy` looks for
`./pdproxy.yaml` in the current directory:

```bash
cd /app
ls pdproxy.yaml   # exists
pdproxy           # automatically uses ./pdproxy.yaml
```

### Run directly without installing

The legacy invocation still works:

```bash
python core/MicroPDProxyServer.py --model /path/to/model --prefill ... --decode ...
```
