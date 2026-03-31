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

## Task 5 (IN PROGRESS)

### Goal
Add observability to the proxy via a Prometheus-compatible metrics endpoint.

### Add `/metrics` endpoint (Prometheus format)

`vllm bench serve` requests `/metrics` (Prometheus-style), which currently returns 404.

**Requirements:**
- Add a `/metrics` endpoint to the proxy
- Expose basic metrics in Prometheus text format:
  - `proxy_requests_total` (counter, by endpoint)
  - `proxy_request_duration_seconds` (histogram, by endpoint)
  - `proxy_active_requests` (gauge)
- Use `prometheus_client` library (already in requirements via vllm dependency)

### Constraints
- Do not modify core proxy business logic beyond adding the new endpoint
- All existing tests must continue to pass
- Add UT for the new endpoint

### Testing / verification
- `/metrics` endpoint returns valid Prometheus text format
- CI green

---

## Task 8 (PLANNED)

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
- Configurable retry count and timeout via `core/config.py` (from Task 6)
- Do not retry on client errors (4xx)

#### 9c: Configurable scheduling policy
- Allow selecting scheduling policy via config (`roundrobin` or `loadbalanced`)
- Extensible for future scheduling strategies

### Constraints
- Must not break existing topology matrix tests
- All new behavior must be configurable and off by default for backward compatibility
- Add UT for health check, failover, and scheduling selection

### Testing / verification
- Health check correctly detects and removes dead nodes
- Failover retries on next node when backend fails
- Scheduling policy selectable via config
- CI green

---

## Task 7 (IN PROGRESS)

### Goal
Support YAML-based configuration as the primary way to pass parameters to the proxy, replacing the growing list of CLI arguments.

### Motivation
The proxy currently requires many CLI arguments (`--model`, `--prefill`, `--decode`, `--port`, `--generator_on_p_node`, `--roundrobin`, etc.) plus environment variables (`ADMIN_API_KEY`, `OPENAI_API_KEY`). As features grow, this becomes unwieldy. A single YAML config file is easier to manage, version-control, and share.

### YAML Schema

```yaml
# MicroPDProxy Configuration

# ============================================================
# Server
# ============================================================
model: /path/to/model           # required
port: 8000                      # default: 8000
log_level: warning              # debug | info | warning | error

# ============================================================
# Nodes
# ============================================================
# Each node entry is "ip:base_port". The topology parameters
# (tp_size, dp_size, world_size_per_node) determine how many
# instances are spawned per node and how ports are assigned.
#
# Example: 2 prefill nodes (TP8, DP2) + 2 decode nodes (TP1, DP16)
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

# ============================================================
# Scheduling
# ============================================================
scheduling: loadbalanced        # roundrobin | loadbalanced
generator_on_p_node: false

# ============================================================
# Auth
# ============================================================
admin_api_key: ""               # env: ADMIN_API_KEY
openai_api_key: ""              # env: OPENAI_API_KEY
```

The above example expands to:
- **Prefill**: 2 nodes × (world_size 8 / tp_size 8) = 1 instance per node = 2 prefill instances (DP=2 ✓)
- **Decode**: 2 nodes × (world_size 8 / tp_size 1) = 8 instances per node = 16 decode instances (DP=16 ✓), ports 8200–8207

### Scope
- Add `--config` / `-c` CLI argument that accepts a path to a YAML file
- Parse YAML and expand topology into final instance address lists using the same rules as `xpyd_start_proxy.sh`
- Validate: instance format, port range, topology constraints (`tp_size * dp_size == nodes * world_size_per_node`, tp/dp must be powers of two)
- Precedence order: **CLI args > environment variables > YAML config > defaults**
- Clear error messages for invalid YAML (missing required fields, bad format, topology constraint violations)
- Add `PyYAML` to `requirements.txt`

### Constraints
- Existing CLI arguments must continue to work (backward compatible)
- If both `--config` and individual CLI args are provided, CLI args take precedence
- Must not change any observable proxy behavior
- All existing tests must continue to pass

### Testing / verification
- UT for YAML parsing, topology expansion, and validation
- UT for error handling (missing file, malformed YAML, invalid topology)
- UT for precedence (CLI overrides YAML)
- Integration test: proxy starts correctly from a YAML config file
- All existing tests still pass
- CI green

---

## Task 6 (DONE)

### Goal
Introduce `core/config.py` as the single source of truth for all proxy configuration — CLI arguments, environment variables, defaults, and validation.

### Scope
- Create `core/config.py` with a Pydantic `BaseSettings` model (or similar) that defines all proxy parameters:
  - `model` (str, required)
  - `prefill` (list of host:port strings, optional)
  - `decode` (list of host:port strings, required, at least one)
  - `port` (int, default 8000, valid range 1-65535)
  - `generator_on_p_node` (bool, default false)
  - `roundrobin` (bool, default false)
  - `admin_api_key` (str, optional, from env `ADMIN_API_KEY`)
  - `openai_api_key` (str, optional, from env `OPENAI_API_KEY`)
- Centralize all validation currently in `validate_parsed_serve_args` and `validate_instances`:
  - Instance format validation (`host:port`)
  - Port range check
  - At least one decode node required
- Provide clear error messages for invalid configuration
- Replace `argparse.Namespace` usage in `ProxyServer.__init__` with the new config object
- Keep CLI argument parsing as a thin layer that feeds into the config model

### Constraints
- Must not change any observable proxy behavior
- All existing tests must continue to pass
- Environment variable bindings must be backward compatible
- `core/MicroPDProxyServer.py` should import and use the config object instead of raw `args`

### Testing / verification
- UT for config validation (valid/invalid inputs, defaults, env overrides)
- All existing integration / benchmark tests still pass
- CI green

---

## Task 4 (DONE)

### Goal
Refactor the scheduler code in `core/MicroPDProxyServer.py` into a clean module structure under `core/scheduler/`.

### Scope
- **Only modify `core/`**
- Extract scheduler-related classes from `MicroPDProxyServer.py` into separate files

### New directory structure
```
core/scheduler/
├── __init__.py
├── scheduler_base.py       # SchedulingPolicy base class
├── round_robin.py           # RoundRobinSchedulingPolicy
└── load_balanced.py         # LoadBalancedScheduler
```

### Requirements
1. **`scheduler_base.py`** — define `SchedulingPolicy` as the abstract base class
2. **`round_robin.py`** — move `RoundRobinSchedulingPolicy` here, inheriting from base
3. **`load_balanced.py`** — move `LoadBalancedScheduler` here, inheriting from base
4. **`MicroPDProxyServer.py`** — import from the new module instead of defining inline
5. **Unit tests** — add tests for both schedulers under `tests/`
6. **CI** — ensure the new tests run in CI

### Constraints
- The refactor must not change any observable behavior
- All existing tests must continue to pass
- `MicroPDProxyServer.py` should import from `core.scheduler` after refactoring

### Testing / verification
- New UT covering both `RoundRobinSchedulingPolicy` and `LoadBalancedScheduler`
- All existing integration / benchmark tests still pass
- CI green

---

## Task 3 (DONE)

### Goal
Support a real benchmark-style validation flow using `vllm bench serve` semantics in a local dummy-based setup.

Reference real-world benchmark command:
```bash
vllm bench serve --host 10.239.129.239 --port 8868 --model /workspace/hf_models/Qwen3-0.6B-Base/ --dataset-name random --random-input-len 3000 --random-output-len 200 --num-prompts 100 --burstiness 100 --request-rate 3.6
```

### Details
- install `vllm` first via `pip install vllm`
- rewrite host / port / model according to the actual local test environment
- make the benchmark run successfully against:
  - 2 dummy prefill instances
  - 2 dummy decode instances
  - 1 proxy server

### Required topology
- prefill: TP8, DP2
- decode: TP1, DP16

### Validation target
Task three is considered complete only if the benchmark can run through the proxy successfully under the topology above. This must be validated end-to-end with real running processes.

### Expected work items
- prepare local dummy prefill / decode topology matching the task-three requirement
- start proxy with the correct parameterized shell script inputs
- install and invoke `vllm bench serve` with locally adjusted arguments
- debug any remaining incompatibility in the proxy / dummy setup if the benchmark cannot run through

### Testing / verification
- the benchmark command can connect to the proxy
- requests are accepted and processed through the proxy
- the dummy prefill / decode stack is sufficient to support the benchmark path
- any missing dummy behavior required by the benchmark flow is fixed if necessary

---

## Task 2 (DONE)

### Goal
Modify only `core/xpyd_start_proxy.sh` so that the current fixed configuration becomes command-line-parameter driven.

### Required parameters

#### Prefill parameters
- `--prefill-nodes` / `-pn`
- `--prefill-tp-size` / `-pt`
- `--prefill-dp-size` / `-pd`
- `--prefill-world-size-per-node` / `-pw`

#### Decode parameters
- `--decode-nodes` / `-dn`
- `--decode-tp-size` / `-dt`
- `--decode-dp-size` / `-dd`
- `--decode-world-size-per-node` / `-dw`

#### Optional parameters
- `--prefill-base-port`
- `--decode-base-port`

### Target command form
```bash
.sh \
  --prefill-nodes $a \
  --prefill-tp-size $b \
  --prefill-dp-size $c \
  --prefill-world-size-per-node $d \
  --decode-nodes $e \
  --decode-tp-size $f \
  --decode-dp-size $g \
  --decode-world-size-per-node $h
```

### Validation rules
The script must validate the input arguments:
1. All parameters must be positive integers.
2. `tp_size` and `dp_size` must be powers of two.
3. `tp_size * dp_size == nodes * world_size_per_node`
4. Node count cannot exceed IP list length: `nodes <= IP list length`

### Topology and mapping rules
- One instance = one TP group.
- N TP shards form one instance.
- Instance count = `dp_size`.
- `tp_size` defines how many TP shards are inside one instance.
- `dp_size` defines how many total instances exist.
- Each node corresponds to one IP. One node may host multiple instances using the same IP but incrementing ports.
- One instance exposes only one `IP:PORT` to the proxy (the main node IP).

### Mapping cases
**Case 1: `tp_size <= world_size_per_node`**
- Each node hosts `world_size_per_node / tp_size` instances.
- Multiple instances on the same node use the same IP and different incrementing ports.

**Case 2: `tp_size > world_size_per_node`**
- Each instance spans `tp_size / world_size_per_node` nodes.
- Different node shards belonging to the same instance use the same port. Only the main node is exposed.

### Implementation constraints
- Only modify `core/xpyd_start_proxy.sh`.
- Do not modify other core business logic files. Small fixes (paths, help output) are allowed.

### Testing requirements
- Same testing requirement as task one.
- Additional negative tests for invalid TP/DP combinations and topologies (e.g., non-powers of two, bad math). The script must fail gracefully.

---

## Task 1 (DONE)

### Goal
Debug `dummy_nodes` to make them work correctly under specific proxy server configurations without modifying core business logic. The content in `core/` was already validated on real hardware.

### Required configurations
- `bash xpyd_start_proxy.sh 1 2 1`
- `bash xpyd_start_proxy.sh 2 2 1`
- `bash xpyd_start_proxy.sh 1 2 2`
- `bash xpyd_start_proxy.sh 1 2 4`
- `bash xpyd_start_proxy.sh 1 2 8`
- `bash xpyd_start_proxy.sh 2 2 2`
- `bash xpyd_start_proxy.sh 2 4 1`
- `bash xpyd_start_proxy.sh 2 4 2`
