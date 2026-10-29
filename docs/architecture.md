# Architecture Overview

## System Architecture

```
                         ┌─────────────────────┐
                         │       Client         │
                         │  (OpenAI-compatible) │
                         └─────────┬───────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │       Proxy          │
                         │  (MicroDisaggregatedProxyServer)│
                         │  Routing + Scheduling│
                         └───┬─────────────┬───┘
                             │             │
                   ┌─────────▼───┐   ┌─────▼─────────┐
                   │  Prefill    │   │  Decode        │
                   │  Nodes      │   │  Nodes         │
                   │ (KV cache)  │   │ (token gen)    │
                   ├─────────────┤   ├────────────────┤
                   │ instance 0  │   │ instance 0     │
                   │ instance 1  │   │ instance 1     │
                   │ ...         │   │ ...            │
                   └─────────────┘   └────────────────┘
```

## Disaggregated Architecture

MicroDisaggregatedProxy implements a **Prefill-Decode disaggregated** serving architecture
for large language models. The two phases of autoregressive inference are split
across dedicated node pools:

| Phase | Role | Description |
|-------|------|-------------|
| **Prefill** | KV cache preparation | Processes the full prompt in a single forward pass, populating the KV cache. The proxy sends a trimmed request (`stream=False`, `max_tokens=1`) to a prefill node. |
| **Decode** | Token generation | Performs autoregressive token generation using the prepared KV cache. The original request (streaming or non-streaming) is forwarded to a decode node. |

This separation allows independent scaling: prefill nodes can be optimized for
throughput (large batch, high parallelism), while decode nodes can be optimized
for latency (smaller batch, faster per-token).

## Core Components

### Proxy (`xpyd/proxy.py`)

`Proxy` owns scheduling, tokenizer state, backend forwarding, and instance
membership. `ProxyServer` constructs these runtime components and starts uvicorn.
The existing `xpyd.proxy:main` CLI entry point remains supported.

### CLI and Application Assembly

- `xpyd/cli.py` parses arguments, loads and overrides configuration, and runs
  configuration-only commands. `load_config()` returns a `ProxyConfig` when a
  server should start, or `None` when a configuration command completes without
  starting one. It does not import or construct the proxy runtime.
- `xpyd/app.py` exposes
  `create_app(router, *, discovery, health_monitor=None, zmq_notifications=None)`.
  It mounts the supplied router and owns CORS, readiness checks, and the
  background-service lifecycle. It does not receive a `Proxy`, create runtime
  dependencies, or write state back to a coordinator.
- `ProxyServer` constructs discovery, the optional health monitor and ZMQ
  listener, and shares the same listener with completion execution and app
  lifecycle management. `ProxyServer.create_app()` wires the application without
  starting a network listener or background services.

Backends may be offline at startup. Discovery starts in the background, inference
returns 503 until ready, and health/status/metrics/admin endpoints remain
accessible.

### Endpoint Modules (`xpyd/routes/`)

Each logical endpoint has a module with `register(router, *, ...)`, taking only
the callbacks or data that endpoint needs.
`routes/__init__.py` registers them all. URL aliases share their logical module:
`score.py` handles `/score` and `/v1/score`, `rerank.py` handles all three rerank
paths, and `health.py` handles `/health` and `/ping`. `status.py` serves both
`/status` and `/status/instances`; `instances.py` handles addition and removal.
Existing OPTIONS handlers stay beside their endpoints.

`completions.py` and `chat_completions.py` are separate HTTP entry points. Each
receives an async `execute(request)` callback and an optional custom handler.
Passthrough endpoints receive a `forward(request, path, template)` callback;
health/version endpoints receive a fetch callback. Model/status endpoints use
query callbacks and the relevant live membership lists. Instance administration
receives only add/remove callbacks; metrics needs no injected runtime object.

`xpyd/completion_handler.py` takes explicit keyword-only operations for scheduling,
forwarding, tokenization, topology lookup, model lookup and outcome recording,
plus `CompletionOptions` and an optional notification interface. Options contain
only transfer configuration, not service objects. The aggregated sub-handler
receives only scheduling, forwarding, model lookup and outcome callbacks; pure
prompt helpers receive only a token-counting function or tokenizer.

`xpyd/interfaces.py` defines narrow callable/tokenizer/notification contracts.
There is no `ProxyProtocol`, generic service container, or all-purpose context
passed through request handling.

Runtime imports are one-way:

```text
proxy -> cli -> config
proxy -> app -> discovery / health_monitor
proxy -> routes -> completion_handler
proxy -> completion_handler
routes / completion_handler -> interfaces
```

Only the composition code in `proxy.py` selects concrete implementations and
binds endpoint-specific completion callbacks. It currently supplies bound
runtime methods, but the receiving modules depend only on the declared callable
contracts: independent functions or alternative services can be supplied instead.
Routes, app assembly, and completion helpers neither receive a `Proxy`/server
object nor import that implementation, including for type annotations.

Completion helpers do not import routes or app assembly. Resource ownership
during streaming is unchanged: request completion and reservation release remain
tied to response consumption. Membership lists remain live so runtime
addition/removal is reflected in status responses.

### Scheduling Policies

| Policy | Class | Description |
|--------|-------|-------------|
| **Round Robin** | `RoundRobinSchedulingPolicy` | Cycles through instances sequentially. Simple and predictable. |
| **Load Balanced** | `LoadBalancedScheduler` | Tracks in-flight request counts and token lengths per instance. Routes new requests to the least-loaded instance. |
| **Consistent Hash** | `ConsistentHashPolicy` | Routes session keys through a stable hash ring. |
| **Power of Two** | `PowerOfTwoPolicy` | Chooses the less-loaded of two sampled candidates. |
| **Cache Aware** | `CacheAwarePolicy` | Routes matching prompt prefixes to the same worker. |

All policies implement `SchedulingPolicy`. Configuration selects a registered
subclass through its `from_config()` factory; the proxy invokes common selection
and membership hooks without branching on concrete strategy types.
`scheduler/runtime.py` filters eligible nodes and owns active-request
reservations. Policy-specific token/KV accounting stays inside each subclass.
Aggregated and P/D request paths share this boundary; see
[Scheduling Policies](scheduling.md#common-policy-contract).

## Request Flow

```
1. Client sends request (e.g., POST /v1/chat/completions)
        │
        ▼
2. Proxy receives request, schedules a prefill instance
        │
        ▼
3. Proxy sends trimmed request to Prefill Node
   (stream=False, max_tokens=1 → KV cache warm-up)
        │
        ▼
4. Prefill Node returns (first token + KV cache ready)
        │
        ▼
5. Proxy schedules a decode instance
        │
        ▼
6. Proxy forwards original request to Decode Node
   (streaming or non-streaming as requested by client)
        │
        ▼
7. Decode Node streams/returns generated tokens
        │
        ▼
8. Proxy relays response back to Client
```

## Topology Concepts

MicroDisaggregatedProxy uses the following topology parameters (configured via
`xpyd_start_proxy.sh`):

| Concept | Description |
|---------|-------------|
| **TP size** (Tensor Parallelism) | Number of GPUs that collaborate on a single forward pass. Must be a power of 2. |
| **DP size** (Data Parallelism) | Number of independent instances (TP groups) that can serve requests in parallel. Must be a power of 2. |
| **Instance** | One TP group. Each instance exposes exactly **one IP:PORT** endpoint to the proxy. |
| **World size per node** | Number of GPUs available on a single physical node. |

The constraint is: `tp_size × dp_size = nodes × world_size_per_node`.

### Example

With 2 nodes, each having 8 GPUs, TP=8, DP=2:
- 2 instances (DP=2), each using 8 GPUs (TP=8)
- Each instance exposes one endpoint (e.g., `node0:8100`, `node1:8100`)
- Total GPUs used: 2 × 8 = 16 = 2 nodes × 8 GPUs/node ✓
