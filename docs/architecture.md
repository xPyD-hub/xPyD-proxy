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
                         │  (MicroPDProxyServer)│
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

## PD-Separated Architecture

MicroPDProxy implements a **Prefill-Decode (PD) separated** serving architecture
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

### Proxy (`core/MicroPDProxyServer.py`)

The central routing and scheduling component. It:

- Accepts OpenAI-compatible API requests from clients
- Routes requests through the prefill → decode pipeline
- Manages prefill and decode instance pools
- Supports dynamic instance addition/removal via admin API
- Provides health checking and status endpoints

### Scheduling Policies

| Policy | Class | Description |
|--------|-------|-------------|
| **Round Robin** | `RoundRobinSchedulingPolicy` | Cycles through instances sequentially. Simple and predictable. |
| **Load Balanced** | `LoadBalancedScheduler` | Tracks in-flight request counts and token lengths per instance. Routes new requests to the least-loaded instance. |

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

MicroPDProxy uses the following topology parameters (configured via
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
