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

## Task 10 (PLANNED)

### Goal
Add advanced load balancing strategies beyond round-robin and load-balanced. Inspired by [vllm-project/router](https://github.com/vllm-project/router).

### Scope

#### 10a: Consistent Hash
- Route requests from the same session/user to the same worker node
- Enables KV cache reuse across multi-turn conversations
- Hash key priority: `X-Session-ID` header > `user` field in request body > client IP
- When a node goes down, only affected sessions are redistributed (minimal disruption)
- YAML config:
  ```yaml
  scheduling: consistent_hash
  consistent_hash:
    header: "X-Session-ID"         # default header to hash on
  ```

#### 10b: Power of Two Choices
- Pick 2 random worker nodes, forward to the one with fewer active requests
- Simple yet effective — avoids the overhead of tracking all workers while still load-aware
- YAML config:
  ```yaml
  scheduling: power_of_two
  ```

#### 10c: Cache-Aware Routing
- Optimize for prefix cache hits by routing similar prompts to the same worker
- Hash the prompt prefix (first N tokens) to select the worker
- Configurable prefix length for hashing
- YAML config:
  ```yaml
  scheduling: cache_aware
  cache_aware:
    prefix_length: 256             # tokens to hash for routing
  ```

#### 10d: Policy Registry
- Implement a policy registry/factory pattern so new scheduling strategies can be added by:
  1. Writing a class that implements `SchedulingPolicy`
  2. Registering it in the registry
- `scheduling` YAML key selects the active policy by name
- Extensible for future custom policies

### Constraints
- All new policies must implement the existing `SchedulingPolicy` interface
- Default scheduling remains `loadbalanced` for backward compatibility
- Must not break existing tests
- Each policy must have comprehensive UT

### CI Testing Strategy

All scheduling policies are pure logic — no network, no processes, fast to test.

#### Tier 1: Unit Tests (fast, mock-based)

**Consistent Hash:**
```python
def test_same_session_same_worker():
    policy = ConsistentHashPolicy(workers=["w1", "w2", "w3", "w4"])
    # Same session ID → same worker every time
    w1 = policy.select(session_id="user-abc")
    w2 = policy.select(session_id="user-abc")
    w3 = policy.select(session_id="user-abc")
    assert w1 == w2 == w3

def test_different_sessions_distribute():
    policy = ConsistentHashPolicy(workers=["w1", "w2", "w3", "w4"])
    selected = {policy.select(session_id=f"user-{i}") for i in range(100)}
    assert len(selected) > 1  # not all on same worker

def test_minimal_redistribution_on_node_removal():
    policy = ConsistentHashPolicy(workers=["w1", "w2", "w3", "w4"])
    before = {f"s{i}": policy.select(session_id=f"s{i}") for i in range(100)}
    policy.remove_worker("w3")
    after = {f"s{i}": policy.select(session_id=f"s{i}") for i in range(100)}
    # Only sessions on w3 should move; others stay
    moved = sum(1 for s in before if before[s] != after[s])
    assert moved < 35  # ~25% expected (1 of 4 workers removed)

def test_hash_key_priority():
    # X-Session-ID header > user field > client IP
    policy = ConsistentHashPolicy(workers=["w1", "w2"])
    r1 = policy.select(header="sess-1", user=None, client_ip="1.2.3.4")
    r2 = policy.select(header="sess-1", user="different", client_ip="5.6.7.8")
    assert r1 == r2  # header takes priority
```

**Power of Two Choices:**
```python
def test_picks_least_loaded():
    policy = PowerOfTwoPolicy(workers=["w1", "w2", "w3"])
    # w1 has 10 active, w2 has 2 active, w3 has 5 active
    policy.set_load("w1", 10)
    policy.set_load("w2", 2)
    policy.set_load("w3", 5)
    # Run 100 selections — w2 should be picked most often
    counts = Counter(policy.select() for _ in range(1000))
    assert counts["w2"] > counts["w1"]

def test_random_pair_selection():
    policy = PowerOfTwoPolicy(workers=["w1", "w2", "w3", "w4"])
    # All equal load → should distribute roughly evenly
    pairs_seen = set()
    for _ in range(1000):
        policy.select()  # track which pairs are compared
        pairs_seen.add(tuple(sorted(policy.last_pair)))
    assert len(pairs_seen) > 1  # multiple pairs used
```

**Cache-Aware:**
```python
def test_same_prefix_same_worker():
    policy = CacheAwarePolicy(workers=["w1", "w2", "w3"], prefix_length=256)
    # Same prompt prefix → same worker
    prompt = "The quick brown fox " * 50  # >256 tokens
    w1 = policy.select(prompt=prompt)
    w2 = policy.select(prompt=prompt + "jumps over the lazy dog")
    assert w1 == w2  # same prefix → same worker

def test_different_prefix_can_differ():
    policy = CacheAwarePolicy(workers=["w1", "w2", "w3"], prefix_length=256)
    selected = set()
    for i in range(50):
        w = policy.select(prompt=f"Unique prompt {i} " * 100)
        selected.add(w)
    assert len(selected) > 1  # different prefixes spread across workers
```

**Policy Registry:**
```python
def test_register_and_select():
    registry = PolicyRegistry()
    registry.register("my_policy", MyCustomPolicy)
    policy = registry.create("my_policy", workers=["w1", "w2"])
    assert isinstance(policy, MyCustomPolicy)

def test_unknown_policy_raises():
    registry = PolicyRegistry()
    with pytest.raises(ValueError, match="Unknown scheduling policy"):
        registry.create("nonexistent", workers=["w1"])

def test_builtin_policies_registered():
    registry = PolicyRegistry()
    for name in ["roundrobin", "loadbalanced", "consistent_hash",
                  "power_of_two", "cache_aware"]:
        assert registry.has(name)
```

#### Tier 2: Integration Tests (real proxy, <30s)

```python
TEST_YAML = """
model: {tokenizer_path}
scheduling: consistent_hash
decode:
  - "127.0.0.1:{port1}"
  - "127.0.0.1:{port2}"
consistent_hash:
  header: "X-Session-ID"
"""

def test_consistent_hash_end_to_end():
    # 1. Start proxy + 2 decode dummy nodes with config above
    # 2. Send 10 requests with X-Session-ID: "session-A"
    #    → all should hit the same decode node
    # 3. Send 10 requests with X-Session-ID: "session-B"
    #    → all should hit the same decode node (may differ from A)
    # 4. Verify via proxy logs or /status which node handled which session

def test_power_of_two_prefers_less_loaded():
    # 1. Start proxy + 2 decode nodes
    # 2. Send 5 slow requests to make one node busy
    # 3. Send 1 fast request → should go to the less loaded node
    # 4. Verify via response timing or logs

def test_policy_selectable_via_yaml():
    # Test each scheduling value starts the proxy correctly:
    for policy in ["roundrobin", "loadbalanced", "consistent_hash",
                    "power_of_two", "cache_aware"]:
        # Start proxy with scheduling: {policy} → no startup error
        # Send one request → 200 OK
```

#### Testing Guidelines for Implementers and Reviewers

1. **Deterministic tests.** For consistent hash and cache-aware, use fixed
   inputs and verify exact worker assignment. Do not rely on random behavior.
2. **Statistical tests for random policies.** Power of two uses randomness —
   use large sample sizes (1000+) and verify distribution properties, not
   exact values.
3. **Test with 1 worker.** All policies must handle the degenerate case of
   a single available worker without error.
4. **Test with 0 workers.** All policies must raise a clear error or return
   None when no workers are available.
5. **Reviewers:** verify that each policy has tests for: normal operation,
   edge cases (1 worker, 0 workers), and the configuration path (YAML →
   policy selection).

---

## Task 9 (PLANNED)

### Goal
Make the proxy self-healing: detect failing nodes, stop sending them traffic, retry failed requests on healthy nodes, and automatically recover when nodes come back. Inspired by [vllm-project/router](https://github.com/vllm-project/router).

---

#### 9a: Instance Registry

**What:** A centralized `InstanceRegistry` that tracks every prefill/decode instance's state in one place. All other components (scheduler, circuit breaker, health monitor) read from and write to the registry.

**Why:** Currently instance state is scattered across `prefill_instances`, `decode_instances`, cyclers, and the scheduler's internal counters. Adding/removing an instance requires touching multiple places. A single registry makes state management reliable.

**Implementation:**
- `InstanceRegistry` stores per-instance: `address`, `role` (prefill/decode), `status` (healthy/unhealthy/unknown), `circuit_breaker_state`, `last_health_check`, `active_request_count`
- `get_available_instances(role)` → returns only healthy instances with closed circuit breakers
- `mark_healthy(addr)` / `mark_unhealthy(addr)` — called by health monitor
- `record_success(addr)` / `record_failure(addr)` — called after each request, feeds circuit breaker
- Scheduler reads from registry instead of raw instance lists

**Verification:**
```
# Start with 4 decode instances
registry.get_available_instances("decode")
→ ["10.0.0.1:8200", "10.0.0.2:8200", "10.0.0.3:8200", "10.0.0.4:8200"]

# Instance 2 marked unhealthy
registry.mark_unhealthy("10.0.0.2:8200")
registry.get_available_instances("decode")
→ ["10.0.0.1:8200", "10.0.0.3:8200", "10.0.0.4:8200"]

# Instance 2 recovers
registry.mark_healthy("10.0.0.2:8200")
registry.get_available_instances("decode")
→ ["10.0.0.1:8200", "10.0.0.2:8200", "10.0.0.3:8200", "10.0.0.4:8200"]
```

**Tests:** UT for add/remove, get_available with various states, concurrent access safety.

---

#### 9b: Circuit Breaker

**What:** Per-node circuit breaker that automatically stops sending requests to a node that is consistently failing, and gradually recovers when the node comes back.

**Why:** Without a circuit breaker, a dead node keeps receiving requests that all fail (wasting time + causing user errors). With it, failures are detected fast and traffic is redirected to healthy nodes.

**State machine:**
```
     success          failure_threshold reached
  ┌──────────┐       ┌──────────────────────┐
  │  CLOSED  │──────►│        OPEN          │
  │ (normal) │       │ (reject all, 503)    │
  └──────────┘       └──────────┬───────────┘
       ▲                        │ timeout expires
       │                        ▼
       │              ┌──────────────────────┐
       │              │      HALF-OPEN       │
       └──────────────│ (allow 1 probe req)  │
     success_threshold└──────────────────────┘
        reached              │ probe fails
                             └──► back to OPEN
```

**Example scenario:**
```
t=0s   Node 10.0.0.2:8200 starts failing
t=0-5s 5 consecutive request failures (failure_threshold=5)
       → Circuit OPENS for this node
t=5-35s All requests skip this node (routed to other nodes)
       → Users see no errors (transparent failover)
t=35s  timeout_duration=30s expires → Circuit goes HALF-OPEN
t=35s  One probe request sent to 10.0.0.2:8200
       → If success: send 1 more (success_threshold=2)
       → If both succeed: Circuit CLOSES, node is back in rotation
       → If probe fails: Circuit re-OPENS, wait another 30s
```

**YAML config:**
```yaml
circuit_breaker:
  enabled: false                  # default: false
  failure_threshold: 5            # consecutive failures → OPEN
  success_threshold: 2            # consecutive successes → CLOSED
  timeout_duration_seconds: 30    # OPEN → HALF-OPEN wait
  window_duration_seconds: 60     # sliding window for failure count
```

**Verification:**
- `GET /status` shows per-node circuit breaker state:
  ```json
  {
    "nodes": {
      "10.0.0.1:8200": {"status": "healthy", "circuit": "closed"},
      "10.0.0.2:8200": {"status": "unhealthy", "circuit": "open", "open_since": "2026-04-01T10:00:05Z"},
      "10.0.0.3:8200": {"status": "healthy", "circuit": "closed"}
    }
  }
  ```

**Tests:** UT for every state transition (closed→open, open→half-open, half-open→closed, half-open→open). Test with mock time to verify timeout behavior.

---

#### 9c: Retry with Exponential Backoff + Jitter

**What:** When a request to a node fails, automatically retry on a *different* healthy node, with increasing delay between retries to avoid overwhelming the system.

**Why:** Transient failures (network blip, brief overload) can be recovered from without the user ever seeing an error. Backoff + jitter prevents all retries from hitting the same node at the same time (thundering herd).

**Backoff formula:**
```
delay = min(initial_backoff_ms * multiplier^attempt, max_backoff_ms)
actual_delay = delay * (1 + random(-jitter_factor, +jitter_factor))
```

**Example scenario:**
```
Request to 10.0.0.1:8200 → 502 Bad Gateway
  Retry 1: wait ~100ms, try 10.0.0.3:8200 → 502
  Retry 2: wait ~200ms, try 10.0.0.4:8200 → 200 OK ✓
  User sees: normal response (slightly slower)

Without retry:
  User sees: 502 Bad Gateway
```

**Do NOT retry:**
- 4xx errors (client's fault, retrying won't help)
- Requests that already started streaming (can't replay a partial stream)
- When all nodes have open circuit breakers (nowhere to retry)

**YAML config:**
```yaml
retry:
  enabled: false                  # default: false
  max_retries: 2                  # default: 2
  initial_backoff_ms: 100         # default: 100
  max_backoff_ms: 10000           # default: 10000
  backoff_multiplier: 2.0         # default: 2.0
  jitter_factor: 0.1              # default: 0.1
  retryable_status_codes:         # default list below
    - 408  # Request Timeout
    - 429  # Too Many Requests
    - 500  # Internal Server Error
    - 502  # Bad Gateway
    - 503  # Service Unavailable
    - 504  # Gateway Timeout
```

**Verification:**
- Proxy logs show retry attempts with backoff:
  ```
  [RETRY] 10.0.0.1:8200 returned 502, attempt 1/2, backoff 103ms, retrying on 10.0.0.3:8200
  [RETRY] 10.0.0.3:8200 returned 502, attempt 2/2, backoff 214ms, retrying on 10.0.0.4:8200
  [OK] 10.0.0.4:8200 returned 200
  ```
- Metrics: `proxy_retry_total` counter by attempt number

**Tests:** UT with mock HTTP responses. Verify backoff timing within expected range. Verify jitter randomness. Verify no retry on 4xx. Verify no retry when streaming.

---

#### 9d: Health Monitor

**What:** Background task that continuously pings every instance and updates the Instance Registry, which in turn drives circuit breaker state transitions.

**Why:** Without active health checking, we only discover an instance is dead when a real user request fails. With health monitoring, dead instances are detected proactively (within 10 seconds) and removed from rotation before any user is affected.

**How it works:**
```
Every 10 seconds:
  For each instance in registry:
    GET http://{instance}/health (timeout: 3s)
    If 200 OK:
      registry.mark_healthy(instance)
      → If circuit was OPEN and timeout expired: transition to HALF-OPEN
    If timeout/error:
      registry.mark_unhealthy(instance)
      → Feeds into circuit breaker failure count
```

**Example scenario:**
```
t=0s    All 4 decode instances healthy
t=10s   Health check: instance 2 timeout → mark unhealthy (failure count: 1)
t=20s   Health check: instance 2 timeout → mark unhealthy (failure count: 2)
...
t=50s   Health check: instance 2 timeout → failure count reaches 5
        → Circuit breaker OPENS for instance 2
        → Log: "[HEALTH] 10.0.0.2:8200 circuit OPEN after 5 failures"
t=60s   Instance 2 comes back, health check returns 200
        → mark healthy, circuit → HALF-OPEN → probe succeeds → CLOSED
        → Log: "[HEALTH] 10.0.0.2:8200 recovered, circuit CLOSED"
```

**YAML config:**
```yaml
health_check:
  enabled: false                  # default: false
  interval_seconds: 10            # default: 10
  timeout_seconds: 3              # default: 3
```

**`/status` response when enabled:**
```json
{
  "prefill_nodes": [
    {"address": "10.0.0.1:8100", "status": "healthy", "circuit": "closed", "last_check": "2026-04-01T10:00:10Z"}
  ],
  "decode_nodes": [
    {"address": "10.0.0.3:8200", "status": "healthy", "circuit": "closed", "last_check": "2026-04-01T10:00:10Z"},
    {"address": "10.0.0.4:8200", "status": "unhealthy", "circuit": "open", "last_check": "2026-04-01T10:00:10Z"}
  ]
}
```

**Tests:** UT with mock HTTP server that toggles health responses. Verify unhealthy detection timing. Verify recovery flow. Verify circuit breaker integration.

---

### Constraints
- All new features default to **disabled** for backward compatibility
- Must not break existing topology matrix tests
- Circuit breaker, retry, and health check are independent — each can be enabled separately
- Startup instance discovery from Task 8 should integrate with the Instance Registry

### CI Testing Strategy

All Task 9 features must be verifiable in CI without external dependencies.
Tests are split into two tiers:

#### Tier 1: Unit Tests (fast, isolated, mock-based)

Run in milliseconds. Test component logic in isolation.

**Instance Registry:**
```python
def test_mark_unhealthy_removes_from_available():
    registry = InstanceRegistry()
    registry.add("decode", "10.0.0.1:8200")
    registry.add("decode", "10.0.0.2:8200")
    registry.mark_unhealthy("10.0.0.1:8200")
    assert registry.get_available("decode") == ["10.0.0.2:8200"]

def test_mark_healthy_restores():
    registry.mark_healthy("10.0.0.1:8200")
    assert "10.0.0.1:8200" in registry.get_available("decode")
```

**Circuit Breaker:**
```python
def test_opens_after_failure_threshold():
    cb = CircuitBreaker(failure_threshold=3)
    for _ in range(3):
        cb.record_failure()
    assert cb.state == "open"

def test_half_open_after_timeout(mock_time):
    cb = CircuitBreaker(failure_threshold=3, timeout_seconds=30)
    for _ in range(3):
        cb.record_failure()
    mock_time.advance(31)  # use freezegun or manual clock
    assert cb.state == "half_open"

def test_closes_after_success_threshold():
    # ... already in half_open state
    for _ in range(2):  # success_threshold=2
        cb.record_success()
    assert cb.state == "closed"
```

**Retry with Backoff:**
```python
def test_retries_on_502(mock_http):
    mock_http.side_effect = [
        Response(502),  # attempt 0 — fail
        Response(502),  # attempt 1 — fail
        Response(200),  # attempt 2 — success
    ]
    result = retry_request(mock_http, max_retries=2)
    assert result.status_code == 200
    assert mock_http.call_count == 3

def test_no_retry_on_400(mock_http):
    mock_http.return_value = Response(400)
    result = retry_request(mock_http, max_retries=2)
    assert result.status_code == 400
    assert mock_http.call_count == 1  # no retry

def test_backoff_timing():
    delays = compute_backoff_delays(
        max_retries=3, initial_ms=100, multiplier=2.0, jitter=0.0
    )
    assert delays == [100, 200, 400]
```

**Health Monitor:**
```python
@pytest.mark.asyncio
async def test_detects_unhealthy_node(mock_aiohttp):
    mock_aiohttp.get("http://10.0.0.1:8200/health", status=200)
    mock_aiohttp.get("http://10.0.0.2:8200/health", exception=TimeoutError)
    monitor = HealthMonitor(registry, interval=1, timeout=1)
    await monitor.check_once()
    assert registry.get_status("10.0.0.1:8200") == "healthy"
    assert registry.get_status("10.0.0.2:8200") == "unhealthy"
```

#### Tier 2: Integration Tests (real processes, <30s total)

Use real dummy nodes and proxy (like `test_proxy_matrix.py`).
Use aggressive config to keep CI fast:

```python
TEST_CONFIG = {
    "circuit_breaker": {
        "enabled": True,
        "failure_threshold": 2,        # open fast
        "success_threshold": 1,        # close fast
        "timeout_duration_seconds": 2,  # short wait
    },
    "retry": {
        "enabled": True,
        "max_retries": 1,
        "initial_backoff_ms": 10,      # near-instant
    },
    "health_check": {
        "enabled": True,
        "interval_seconds": 1,         # check every 1s
        "timeout_seconds": 1,
    },
}
```

**Scenario: Node failure → circuit open → recovery → circuit close**
```python
def test_circuit_breaker_end_to_end():
    # 1. Start proxy + 2 decode dummy nodes
    # 2. Send request → 200 OK (both nodes healthy)
    # 3. Kill decode node 1
    # 4. Wait for health check to detect (~2s)
    # 5. Send requests → all routed to node 2 (circuit open on node 1)
    # 6. GET /status → verify node 1 circuit="open"
    # 7. Restart decode node 1
    # 8. Wait for circuit half-open → probe → close (~4s)
    # 9. Send requests → balanced across both nodes again
    # Total: ~10s
```

**Scenario: Retry failover to different node**
```python
def test_retry_routes_to_different_node():
    # 1. Start proxy + 2 decode nodes
    # 2. Make node 1 return 502 (modify dummy node behavior)
    # 3. Send request → proxy retries on node 2 → 200 OK
    # 4. Verify proxy logs show retry attempt
    # 5. Verify metrics: proxy_retry_total incremented
```

#### Testing Guidelines for Implementers and Reviewers

1. **Mock time, not sleep.** Use `freezegun` or manual clock injection for
   circuit breaker timeouts. Never use `time.sleep()` in unit tests.
2. **Use `aioresponses`** for async HTTP mocking in health monitor tests.
3. **Keep integration tests under 30 seconds total.** Use the `TEST_CONFIG`
   above with aggressive timeouts.
4. **Every state transition must have a test.** Circuit breaker has 5
   transitions — test all 5. Do not skip half-open → open (probe failure).
5. **Test concurrency.** Instance Registry will be accessed from multiple async
   tasks. Add a test with concurrent read/write operations.
6. **Reviewers:** verify that every code path described above has a
   corresponding test. Reject PRs that add features without matching tests.

### Future (not in this task)
- K8s service discovery
- Separate metrics port

---

## Task 8 (IN PROGRESS)

### Goal
Package the proxy as an installable CLI tool (`xpyd`) with startup node discovery.

### Scope

#### 8a: CLI packaging
- Add `pyproject.toml` (or `setup.py`) with `console_scripts` entry point:
  ```
  [project.scripts]
  xpyd = "core.MicroPDProxyServer:main"
  ```
- Install via `pip install .` (or `pip install -e .` for dev)
- After install, `xpyd` command is available system-wide

#### 8b: CLI interface
- `xpyd --config proxy.yaml` or `xpyd -c proxy.yaml` — start with YAML config
- `XPYD_CONFIG=proxy.yaml xpyd` — environment variable alternative
- Default: search for `./xpyd.yaml` in current directory if no config specified
- Precedence: `--config` > `XPYD_CONFIG` env var > `./xpyd.yaml`
- `xpyd --help` — show all available options
- `xpyd --version` — show version
- `xpyd --validate-config proxy.yaml` — validate YAML without starting server
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
- Integration test: `xpyd -c test.yaml` starts and serves requests
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
