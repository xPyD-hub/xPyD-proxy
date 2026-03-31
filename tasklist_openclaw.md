# Task for OpenClaw

## Review Requirements (Strict Policy)
When reviewing PRs against this repository, bots and reviewers MUST adhere to the following strict guidelines:
1. **Line-by-line inspection:** Do not skim. Check every modified line.
2. **Robustness & Error Handling:** Look for unhandled exceptions, missing input validation (e.g., missing keys in JSON payloads), and edge case handling.
3. **Code Quality:** Reject hardcoded magic numbers, poor variable naming, and messy logic. Do not be lenient.
4. **Core Protection:** `core/MicroPDProxyServer.py` logic must remain intact and robust. If modifications are made, they must be purely formatting, safe pre-commit fixes, or necessary and robust bug fixes. Any regression in validation or core logic must result in a `REQUEST_CHANGES`.

---

## Project Overview
The most critical implementation file is the proxy server:
- <https://github.com/hlin99/MicroPDProxy/blob/main/core/MicroPDProxyServer.py>

The system uses a shell script `xpyd_start_proxy.sh` to configure and launch a distributed proxy handling prefill and decode nodes.

---

## Task 9 (PLANNED)

### Goal
Improve proxy resilience with health monitoring and failover.

### Scope

#### 9a: Backend health monitor
- Periodically ping all prefill/decode backend nodes (e.g. every 10s)
- Automatically remove unhealthy nodes from the active pool
- Re-add nodes when they recover
- Expose health status in `/status` endpoint

#### 9b: Request failover
- When a request to a backend node fails (connection error, timeout), automatically retry on the next available node
- Configurable retry count and timeout
- Do not retry on client errors (4xx)

### Constraints
- Must not break existing topology matrix tests
- All new behavior must be configurable and off by default for backward compatibility
- Add UT for health check and failover

### Testing / verification
- Health check correctly detects and removes dead nodes
- Failover retries on next node when backend fails
- CI green

---

## Task 8 (IN PROGRESS)

### Goal
Package the proxy as an installable CLI tool (`pdproxy`) with startup node discovery.

### Scope

#### 8a: CLI packaging
- Add `pyproject.toml` (or `setup.py`) with `console_scripts` entry point:
  ```
  [project.scripts]
  pdproxy = "core.MicroPDProxyServer:main"
  ```
- Install via `pip install .` (or `pip install -e .` for dev)
- After install, `pdproxy` command is available system-wide

#### 8b: CLI interface
- `pdproxy --config proxy.yaml` or `pdproxy -c proxy.yaml` — start with YAML config
- `PDPROXY_CONFIG=proxy.yaml pdproxy` — environment variable alternative
- Default: search for `./pdproxy.yaml` in current directory if no config specified
- Precedence: `--config` > `PDPROXY_CONFIG` env var > `./pdproxy.yaml`
- `pdproxy --help` — show all available options
- `pdproxy --version` — show version
- `pdproxy --validate-config proxy.yaml` — validate YAML without starting server
- Existing CLI arguments (`--model`, `--prefill`, `--decode`, etc.) continue to work for backward compatibility

#### 8c: Startup node discovery
- On startup, the proxy port opens immediately (uvicorn starts listening)
- Before at least 1 prefill + 1 decode node are ready, return **503 Service Unavailable** for all business requests (with message "waiting for backend nodes")
- Background task pings all configured prefill/decode nodes every `probe_interval_seconds` (default: 10s) on their `/health` endpoint
- When a node responds healthy, add it to the active scheduling pool
- Once minimum 1 prefill + 1 decode are ready → start accepting requests, log `"Proxy ready: N prefill, M decode nodes available"`
- As more nodes come online, dynamically add them to the pool (log each: `"[3/16 decode nodes ready]"`)
- If `wait_timeout_seconds` (default: 600 = 10 min) expires without minimum 1P+1D → exit with error

#### YAML config additions
```yaml
startup:
  wait_timeout_seconds: 600     # default: 600 (10 min), max wait for 1P+1D
  probe_interval_seconds: 10    # default: 10
```

### Constraints
- `pip install .` must work cleanly
- Existing `python core/MicroPDProxyServer.py` invocation must still work
- All existing tests must continue to pass
- Startup discovery must not block the event loop

### Testing / verification
- UT for CLI argument parsing and config resolution (CLI > env > file > default)
- UT for `--validate-config`
- UT for startup discovery: mock nodes coming online gradually, verify 503 → 200 transition
- UT for timeout: no nodes available → exit after timeout
- Integration test: `pdproxy -c test.yaml` starts and serves requests
- CI green

---

## Task 7 (DONE)

### Goal
Support YAML-based configuration as the primary way to pass parameters to the proxy, replacing the growing list of CLI arguments.

### YAML Schema

```yaml
# MicroPDProxy Configuration

# Server
model: /path/to/model           # required
port: 8000                      # default: 8000
log_level: warning              # debug | info | warning | error

# Nodes — each entry is "ip:base_port" with topology parameters
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

# Scheduling
scheduling: loadbalanced        # roundrobin | loadbalanced
generator_on_p_node: false

# Auth
admin_api_key: ""               # env: ADMIN_API_KEY
openai_api_key: ""              # env: OPENAI_API_KEY
```

Topology expansion example:
- **Prefill**: 2 nodes × (world_size 8 / tp_size 8) = 1 instance/node = 2 instances (DP=2 ✓)
- **Decode**: 2 nodes × (world_size 8 / tp_size 1) = 8 instances/node = 16 instances (DP=16 ✓)

---

## Task 6 (DONE)

### Goal
Introduce `core/config.py` as the single source of truth for all proxy configuration — CLI arguments, environment variables, defaults, and validation.

---

## Task 5 (DONE)

### Goal
Add observability to the proxy via a Prometheus-compatible `/metrics` endpoint.

- `proxy_requests_total` (counter, by endpoint)
- `proxy_request_duration_seconds` (histogram, by endpoint)
- `proxy_active_requests` (gauge)
- Streaming-aware request tracking (duration measured over full response lifetime)

---

## Task 4 (DONE)

### Goal
Refactor the scheduler code in `core/MicroPDProxyServer.py` into a clean module structure under `core/scheduler/`.

---

## Task 3 (DONE)

### Goal
Support a real benchmark-style validation flow using `vllm bench serve` semantics in a local dummy-based setup.

---

## Task 2 (DONE)

### Goal
Parameterize `core/xpyd_start_proxy.sh` so that the fixed configuration becomes command-line-parameter driven.

---

## Task 1 (DONE)

### Goal
Debug `dummy_nodes` to make them work correctly under specific proxy server configurations without modifying core business logic.

### Required configurations
`1 2 1`, `2 2 1`, `1 2 2`, `1 2 4`, `1 2 8`, `2 2 2`, `2 4 1`, `2 4 2`
