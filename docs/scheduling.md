# Scheduling Policies

## Overview

MicroPDProxy supports multiple scheduling policies that control how incoming
requests are distributed across backend decode (and prefill) nodes. Different
workloads have different needs — some benefit from even distribution, others
from session affinity or cache locality. The scheduling policy is selected via
the `scheduling` field in the YAML configuration file.

## Round Robin

**Status:** Implemented

Round Robin distributes requests to backend nodes in a fixed cyclic order.
Each node receives one request before the cycle repeats.

### How It Works

The scheduler maintains an internal counter. On each request, the counter
increments and the request is forwarded to `nodes[counter % len(nodes)]`.

### Characteristics

- **Predictable** — every node gets exactly the same number of requests over
  time (assuming no failures).
- **No load awareness** — a slow node receives the same traffic as a fast one.
- **No session affinity** — consecutive requests from the same user may hit
  different nodes.
- **Zero overhead** — no state beyond a single integer counter.

### When to Use

- All backend nodes are identical in capacity.
- Request processing times are uniform.
- You want the simplest possible distribution.

### Configuration

```yaml
scheduling: roundrobin
```

No additional parameters.

## Load Balanced

**Status:** Implemented

Load Balanced routing tracks the number of active (in-flight) requests on each
node and sends new requests to the node with the fewest active requests.

### How It Works

The scheduler maintains a per-node active request counter. When a request
arrives, the node with the lowest counter is selected. The counter increments
on dispatch and decrements when the response completes (including streaming
responses).

### Characteristics

- **Load-aware** — naturally adapts to heterogeneous node performance.
- **No session affinity** — requests from the same user may land on different
  nodes.
- **Minimal overhead** — only per-node integer counters.
- **Handles stragglers** — slow nodes accumulate active requests, causing
  new traffic to flow elsewhere.

### When to Use

- Backend nodes have different capacities or response times.
- Request durations vary significantly.
- You want automatic adaptation without manual tuning.

### Configuration

```yaml
scheduling: loadbalanced
```

No additional parameters. This is the **default** policy.

## Consistent Hash *(planned)*

**Status:** Planned (see Task 10a)

Consistent Hash routes requests from the same session or user to the same
backend node, enabling KV cache reuse across multi-turn conversations.

### How It Works

The scheduler hashes a session identifier to select a node from a hash ring.
When a node is removed, only sessions mapped to that node are redistributed —
all other mappings remain stable.

### Hash Key Priority

The scheduler determines the hash key using the following priority:

1. `X-Session-ID` HTTP header (highest priority)
2. `user` field in the JSON request body
3. Client IP address (fallback)

### When to Use

- Multi-turn conversations where KV cache reuse reduces latency.
- Workloads with natural session identifiers.
- You want minimal disruption when nodes are added or removed.

### Configuration

```yaml
scheduling: consistent_hash
consistent_hash:
  header: "X-Session-ID"         # HTTP header to hash on (default)
```

## Power of Two Choices *(planned)*

**Status:** Planned (see Task 10b)

Power of Two Choices picks two random backend nodes and forwards the request to
whichever has fewer active requests.

### How It Works

On each request, the scheduler randomly selects two candidate nodes, queries
their active request counts, and routes to the less loaded one. This achieves
near-optimal load distribution with O(1) overhead — no need to scan all nodes.

### When to Use

- Large clusters where scanning all nodes is expensive.
- You want load awareness without the complexity of a full least-connections
  algorithm.
- Workloads with high request rates where per-request overhead matters.

### Configuration

```yaml
scheduling: power_of_two
```

No additional parameters.

## Cache-Aware Routing *(planned)*

**Status:** Planned (see Task 10c)

Cache-Aware routing hashes the prompt prefix to select a backend node,
maximizing prefix cache hits across requests with similar prompts.

### How It Works

The scheduler extracts the first N tokens of the prompt, hashes them, and maps
the hash to a backend node. Requests sharing the same prompt prefix are routed
to the same node, increasing the likelihood that the node's KV cache already
contains the prefix computation.

### When to Use

- Workloads with many requests sharing common system prompts or prefixes.
- Large models where prefix computation is expensive.
- You want to maximize GPU cache utilization.

### Configuration

```yaml
scheduling: cache_aware
cache_aware:
  prefix_length: 256             # number of tokens to hash (default: 256)
```

## Policy Selection

The active scheduling policy is set via the `scheduling` field in the YAML
configuration file:

```yaml
scheduling: loadbalanced         # or: roundrobin, consistent_hash, power_of_two, cache_aware
```

If omitted, the default is `loadbalanced`.

A planned Policy Registry (Task 10d) will allow new scheduling strategies to be
added by implementing the `SchedulingPolicy` interface and registering them
by name, without modifying existing code.

## Comparison Table

| Policy | Load Aware | Session Affinity | Cache Friendly | Overhead | Best For |
|---|---|---|---|---|---|
| Round Robin | No | No | No | O(1) | Homogeneous clusters, uniform requests |
| Load Balanced | Yes | No | No | O(N) | Heterogeneous nodes, variable latency |
| Consistent Hash *(planned)* | No | Yes | Partial | O(1) | Multi-turn conversations, KV cache reuse |
| Power of Two *(planned)* | Yes | No | No | O(1) | Large clusters, high throughput |
| Cache-Aware *(planned)* | No | Prompt-based | Yes | O(1) | Shared system prompts, prefix caching |

> **N** = number of backend nodes.
