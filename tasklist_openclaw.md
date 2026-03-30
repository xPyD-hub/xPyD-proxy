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

## Task 4 (IN PROGRESS)

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
