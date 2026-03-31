# Resilience *(planned)*

> **Status:** All features in this document are planned (Task 9). None are
> implemented yet.

## Overview

MicroPDProxy's resilience layer consists of four components that work together
to detect failures, isolate bad nodes, retry failed requests, and
automatically recover when nodes come back online.

```
                    ┌──────────────────┐
                    │  Health Monitor   │
                    │ (periodic probes) │
                    └────────┬─────────┘
                             │ mark healthy / unhealthy
                             ▼
┌──────────┐        ┌──────────────────┐        ┌─────────────────┐
│ Incoming │───────►│ Instance Registry │───────►│ Circuit Breaker │
│ Request  │        │  (node state DB)  │        │  (per-node FSM) │
└──────────┘        └──────────────────┘        └────────┬────────┘
                                                         │ node available?
                                                         ▼
                                                ┌─────────────────┐
                                                │     Retry       │
                                                │ (exp. backoff)  │
                                                └─────────────────┘
```

**Flow:** The Health Monitor continuously probes nodes and updates the Instance
Registry. When a request arrives, the scheduler queries the registry for
available nodes (those that are healthy and have a closed circuit breaker). If
a request fails, the Retry component re-dispatches it to a different healthy
node with exponential backoff. Repeated failures trigger the Circuit Breaker to
open, removing the node from rotation until it recovers.

## Instance Registry

### What

A centralized `InstanceRegistry` that tracks every prefill and decode node's
state in one place. All other components — scheduler, circuit breaker, health
monitor — read from and write to the registry.

### Why

Currently node state is scattered across multiple data structures. Adding or
removing a node requires touching several places. A single registry makes
state management reliable and consistent.

### API

| Method | Description |
|---|---|
| `add(role, address)` | Register a node (prefill or decode). |
| `remove(address)` | Remove a node from the registry. |
| `get_available_nodes(role)` | Return only healthy nodes with closed circuit breakers. |
| `mark_healthy(address)` | Mark a node as healthy (called by Health Monitor). |
| `mark_unhealthy(address)` | Mark a node as unhealthy (called by Health Monitor). |
| `record_success(address)` | Record a successful request (feeds circuit breaker). |
| `record_failure(address)` | Record a failed request (feeds circuit breaker). |
| `get_status(address)` | Return current status, circuit state, and metadata. |

### Per-Node State

Each node entry stores: `address`, `role` (prefill/decode), `status`
(healthy/unhealthy/unknown), `circuit_breaker_state`, `last_health_check`,
`active_request_count`.

## Health Monitor

### What

A background async task that continuously pings every registered node and
updates the Instance Registry based on health check responses.

### Why

Without active health checking, dead nodes are only discovered when a real
user request fails. The Health Monitor detects failures proactively — typically
within one check interval — and removes bad nodes from rotation before users
are affected.

### How It Works

```
Every <interval_seconds> seconds:
  For each node in registry:
    GET http://{node}/health  (timeout: <timeout_seconds>)
    If 200 OK  → registry.mark_healthy(node)
    If error   → registry.mark_unhealthy(node)
```

### Example Timeline

```
t=0s    All 4 decode nodes healthy
t=10s   Health check: node 2 timeout → mark unhealthy (failure count: 1)
t=20s   Health check: node 2 timeout → failure count: 2
...
t=50s   Health check: node 2 timeout → failure count reaches threshold
        → Circuit breaker OPENS for node 2
t=60s   Node 2 comes back, health check returns 200
        → mark healthy → circuit transitions to HALF-OPEN → probe → CLOSED
```

### Configuration

```yaml
health_check:
  enabled: false                  # default: false
  interval_seconds: 10            # default: 10
  timeout_seconds: 3              # default: 3
```

## Circuit Breaker

### What

A per-node finite state machine that automatically stops sending requests to a
node that is consistently failing, and gradually recovers when the node comes
back.

### Why

Without a circuit breaker, a dead node keeps receiving requests that all fail,
wasting time and causing user-visible errors. With it, failures are detected
quickly and traffic is transparently redirected to healthy nodes.

### State Machine

```
     success            failure_threshold reached
  ┌───────────┐        ┌───────────────────────┐
  │  CLOSED   │───────►│         OPEN           │
  │ (normal)  │        │  (reject all → 503)    │
  └───────────┘        └───────────┬────────────┘
       ▲                           │ timeout expires
       │                           ▼
       │               ┌───────────────────────┐
       │               │       HALF-OPEN        │
       └───────────────│  (allow 1 probe req)   │
     success_threshold └───────────────────────┘
       reached                   │ probe fails
                                 └──► back to OPEN
```

### State Transitions

| From | To | Trigger |
|---|---|---|
| CLOSED | OPEN | Consecutive failures reach `failure_threshold` within `window_duration_seconds` |
| OPEN | HALF-OPEN | `timeout_duration_seconds` has elapsed since the circuit opened |
| HALF-OPEN | CLOSED | `success_threshold` consecutive successful probe requests |
| HALF-OPEN | OPEN | A probe request fails |

### Example Scenario Timeline

```
t=0s    Node 10.0.0.2:8200 starts failing
t=0-5s  5 consecutive request failures (failure_threshold=5)
        → Circuit OPENS for this node
t=5-35s All requests skip this node (routed to healthy nodes)
        → Users see no errors (transparent failover)
t=35s   timeout_duration=30s expires → Circuit goes HALF-OPEN
t=35s   One probe request sent to 10.0.0.2:8200
        → If success: send 1 more (success_threshold=2)
        → If both succeed: Circuit CLOSES, node is back in rotation
        → If probe fails: Circuit re-OPENS, wait another 30s
```

### Configuration

```yaml
circuit_breaker:
  enabled: false                  # default: false
  failure_threshold: 5            # consecutive failures to trigger OPEN
  success_threshold: 2            # consecutive successes to return to CLOSED
  timeout_duration_seconds: 30    # OPEN → HALF-OPEN wait time
  window_duration_seconds: 60     # sliding window for failure counting
```

### Status Endpoint

When enabled, `GET /status` includes per-node circuit breaker state:

```json
{
  "nodes": {
    "10.0.0.1:8200": {"status": "healthy", "circuit": "closed"},
    "10.0.0.2:8200": {"status": "unhealthy", "circuit": "open", "open_since": "2026-04-01T10:00:05Z"},
    "10.0.0.3:8200": {"status": "healthy", "circuit": "closed"}
  }
}
```

## Retry with Exponential Backoff

### What

When a request to a node fails with a retryable error, the proxy automatically
retries on a *different* healthy node, with increasing delay between attempts
to avoid overwhelming the system.

### Why

Transient failures (network blips, brief overloads) can be recovered from
without the user ever seeing an error. Backoff with jitter prevents all retries
from hitting the same node at the same time (thundering herd problem).

### Backoff Formula

```
delay = min(initial_backoff_ms * multiplier^attempt, max_backoff_ms)
actual_delay = delay * (1 + random(-jitter_factor, +jitter_factor))
```

### Example

```
Request to 10.0.0.1:8200 → 502 Bad Gateway
  Retry 1: wait ~100ms, try 10.0.0.3:8200 → 502
  Retry 2: wait ~200ms, try 10.0.0.4:8200 → 200 OK ✓
  User sees: normal response (slightly slower)

Without retry:
  User sees: 502 Bad Gateway
```

### Do / Don't Retry

| Retry | Don't Retry |
|---|---|
| 408 Request Timeout | 4xx client errors (400, 401, 403, 404, 422) |
| 429 Too Many Requests | Requests that already started streaming |
| 500 Internal Server Error | When all nodes have open circuit breakers |
| 502 Bad Gateway | |
| 503 Service Unavailable | |
| 504 Gateway Timeout | |

### Configuration

```yaml
retry:
  enabled: false                  # default: false
  max_retries: 2                  # default: 2
  initial_backoff_ms: 100         # default: 100
  max_backoff_ms: 10000           # default: 10000
  backoff_multiplier: 2.0         # default: 2.0
  jitter_factor: 0.1              # default: 0.1
  retryable_status_codes:         # default list:
    - 408
    - 429
    - 500
    - 502
    - 503
    - 504
```

## Configuration Reference

All resilience-related YAML configuration in one place:

```yaml
health_check:
  enabled: false
  interval_seconds: 10
  timeout_seconds: 3

circuit_breaker:
  enabled: false
  failure_threshold: 5
  success_threshold: 2
  timeout_duration_seconds: 30
  window_duration_seconds: 60

retry:
  enabled: false
  max_retries: 2
  initial_backoff_ms: 100
  max_backoff_ms: 10000
  backoff_multiplier: 2.0
  jitter_factor: 0.1
  retryable_status_codes:
    - 408
    - 429
    - 500
    - 502
    - 503
    - 504
```

All features default to **disabled** for backward compatibility. Each can be
enabled independently.

## Troubleshooting

| Symptom | Possible Cause | Resolution |
|---|---|---|
| All requests return 503 | All nodes are unhealthy or all circuit breakers are open | Check node health manually; verify network connectivity; check `/status` endpoint for circuit breaker states |
| Node marked unhealthy but is actually running | Health check timeout too short; network latency | Increase `health_check.timeout_seconds` |
| Circuit breaker opens too quickly | `failure_threshold` is too low or transient errors are common | Increase `failure_threshold` or `window_duration_seconds` |
| Circuit breaker never recovers | Node is genuinely down; `timeout_duration_seconds` is too long | Check node status; reduce `timeout_duration_seconds` for faster probing |
| Retries cause duplicate processing | Retrying non-idempotent requests | Ensure only idempotent endpoints are exposed through the proxy; streaming requests are never retried |
| Retry storms under load | Too many retries with insufficient backoff | Reduce `max_retries`; increase `initial_backoff_ms` and `backoff_multiplier` |
| High latency on retried requests | Backoff delays accumulating | Reduce `max_backoff_ms`; consider whether retries are appropriate for your latency SLA |
