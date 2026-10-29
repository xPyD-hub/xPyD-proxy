# Scheduling Policies

## Overview

MicroDisaggregatedProxy supports multiple scheduling policies that control how incoming
requests are distributed across backend decode (and prefill) instances.
Different workloads have different needs — some benefit from even distribution,
others from session affinity or cache locality. The scheduling policy is
selected via the `scheduling` field in the YAML configuration file.

## Round Robin

**Status:** Implemented

Round Robin distributes requests to backend instances in a fixed cyclic order.
Each instance receives one request before the cycle repeats.

### How It Works

The scheduler maintains an internal counter. On each request, the counter
increments and the request is forwarded to `instances[counter % len(instances)]`.

### Characteristics

- **Predictable** — every instance gets exactly the same number of requests
  over time (assuming no failures).
- **No load awareness** — a slow instance receives the same traffic as a fast
  one.
- **No session affinity** — consecutive requests from the same user may hit
  different instances.
- **Zero overhead** — no state beyond a single integer counter.

### When to Use

- All backend instances are identical in capacity.
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
instance and sends new requests to the instance with the fewest active
requests.

### How It Works

For aggregated instances, the scheduler selects the lowest active-request count.
For P/D, prefill selection minimizes in-flight prompt tokens; decode selection
minimizes active requests, breaking busy-node ties using in-flight prompt tokens.
P/D candidates must also satisfy the model-length limit. Reservations release
these counts when a request completes, fails, or is cancelled.

The policy does not query backends. At startup the proxy composition layer
queries P/D node capacities and passes an address-to-`max_model_len` mapping
into the policy. Unreachable nodes retain the existing logged fallback of
131072, so proxy-first startup remains supported. Dynamically added nodes
receive their validated capacity through `on_instance_added()`.

### Characteristics

- **Load-aware** — naturally adapts to heterogeneous instance performance.
- **No session affinity** — requests from the same user may land on different
  instances.
- **Minimal overhead** — only per-instance integer counters.
- **Handles stragglers** — slow instances accumulate active requests, causing
  new traffic to flow elsewhere.

### When to Use

- Backend instances have different capacities or response times.
- Request durations vary significantly.
- You want automatic adaptation without manual tuning.

### Configuration

```yaml
scheduling: loadbalanced
```

No additional parameters. This is the **default** policy.

## Consistent Hash

**Status:** Implemented

Consistent Hash routes requests from the same session or user to the same
backend instance, enabling KV cache reuse across multi-turn conversations.

### How It Works

The scheduler hashes a session identifier to select an instance from a hash
ring. When an instance is removed, only sessions mapped to that instance are
redistributed — all other mappings remain stable.

### Hash Key Priority

The scheduler determines the hash key using the following priority:

1. `X-Session-ID` HTTP header (highest priority)
2. `user` field in the JSON request body
3. Client IP address (fallback)

### When to Use

- Multi-turn conversations where KV cache reuse reduces latency.
- Workloads with natural session identifiers.
- You want minimal disruption when instances are added or removed.

### Configuration

```yaml
scheduling: consistent_hash
consistent_hash:
  virtual_nodes: 160            # virtual nodes per worker
```

## Power of Two Choices

**Status:** Implemented

Power of Two Choices picks two random backend instances and forwards the
request to whichever has fewer active requests.

### How It Works

On each request, the scheduler randomly selects two candidate instances,
queries their active request counts, and routes to the less loaded one. This
compares only two loads. Building and sorting the eligible candidate set still
has overhead proportional to the pool size; the full routing path is not O(1).

### When to Use

- Large clusters where scanning all instances is expensive.
- You want load awareness without the complexity of a full least-connections
  algorithm.
- Workloads with high request rates where per-request overhead matters.

### Configuration

```yaml
scheduling: power_of_two
```

No additional parameters.

## Cache-Aware Routing

**Status:** Implemented

Cache-Aware routing hashes the prompt prefix to select a backend instance,
maximizing prefix cache hits across requests with similar prompts.

### How It Works

The scheduler extracts the first N tokens of the prompt, hashes them, and maps
the hash to a backend instance. Requests sharing the same prompt prefix are
routed to the same instance, increasing the likelihood that the instance's KV
cache already contains the prefix computation.

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

The policy registry constructs `SchedulingPolicy` subclasses from the configured
name and options. Aggregated models can override the global policy with the
existing model-level `scheduler` setting. Tokenizer-load failure continues to
override that model with round-robin routing.

### Common policy contract

`xpyd/scheduler/scheduler_base.py` defines the interface for all five policies:

| Method | Responsibility |
|---|---|
| `from_config(...)` | Construct the subclass from common topology inputs and strategy options. |
| `select_node(context, candidates)` | Select an eligible address, or return `None` when no node can serve it. Reserve any strategy-specific load here. |
| `on_instance_added(role, address, max_model_len)` | Update internal state before membership is extended. |
| `on_instance_removed(role, address, index)` | Update internal state before drained membership is shortened. |
| `on_request_finished(context, address)` | Release strategy-specific load, regardless of request outcome. |

`SchedulingContext` carries the role (`aggregated`, `prefill`, or `decode`),
model, token lengths, session information, and prompt. `Candidate` provides
the eligible node address and current active-request count. Policies do not
need to understand HTTP requests, streaming responses, or retry handling.

`xpyd/scheduler/runtime.py` owns common scheduling behavior: registry-based
health/model/role filtering, serialized selection and active-request accounting,
and an idempotent `Reservation.release()`. The reservation retains the original
policy and context, so changing a model's tokenizer fallback cannot release
load against the wrong policy. Releasing a reservation does not record success
or failure; the request executor handles outcomes separately.

Only the runtime holds an `InstanceRegistry`; policies and their base class
neither accept nor store one. `from_config()` receives topology data,
`model_lengths`, tokenizer input and strategy options, not live registry state.
Policies declare `requires_model_lengths = True` when they need startup
capacities. The composition layer inspects that declaration via
`PolicyRegistry.get_policy_class()`, queries capacities only for those policies,
then uses the common factory. Aliases and custom subclasses follow the same
path without a branch on a concrete policy name or type.

For direct Python construction, capacity-aware P/D policies require supplied
capacities; missing entries are errors rather than implicit network queries:

```python
from xpyd.scheduler import LoadBalancedScheduler

policy = LoadBalancedScheduler(
    ["prefill:8100"],
    ["decode:8200"],
    model_lengths={"prefill:8100": 8192, "decode:8200": 16384},
)
```

The mapping is read by address, not by iteration order. Node membership lists
retain their shared lifecycle semantics; capacity values are copied into policy
state. Aggregated-only load balancing needs no P/D capacity data.

The proxy uses this same contract for aggregated and P/D requests. It does not
branch on concrete strategy types. Request code calls `Proxy.reserve(context)`
and retains the returned reservation until it can call `release()`.

This is a breaking Python API change, with no compatibility adapters:
`schedule()`, `schedule_completion()`, aggregated scheduling wrappers, policy
`select()` / `select_from()`, and cyclers have been removed. Construct policies
with `default_registry.build(...)`, not `PolicyRegistry.create()` or an explicit
policy class passed to `ProxyServer`. Import policy classes from `xpyd.scheduler`,
not `xpyd.proxy`. All policy selection and membership changes use the interface
above; active-request accounting belongs to the runtime, not to a second set of
power-of-two counters.

### Adding a policy

Implement `SchedulingPolicy.select_node()` and any required lifecycle hooks.
Override `from_config()` if the constructor needs special arguments; otherwise
the default factory passes `workers` and the configured options to the constructor.
Override it to consume additional common inputs such as `model_lengths`.
Register the class before loading YAML:

```python
from xpyd.scheduler import SchedulingPolicy, default_registry

class LastCandidatePolicy(SchedulingPolicy):
    def select_node(self, context, candidates):
        return candidates[-1].address if candidates else None

default_registry.register("last_candidate", LastCandidatePolicy)
```

Then set `scheduling: last_candidate`. Optional parameters can be placed in a
`last_candidate:` mapping or `scheduling_config.last_candidate`. No proxy or
endpoint changes are required. The common contract tests cover all roles,
dynamic membership, filtering, concurrent accounting, and exact-once release;
the CPU semantic harness additionally checks actual backend selection.

## Comparison Table

| Policy | Load Aware | Session Affinity | Cache Friendly | Overhead | Best For |
|---|---|---|---|---|---|
| Round Robin | No | No | No | O(1) | Homogeneous clusters, uniform requests |
| Load Balanced | Yes | No | No | O(N) | Heterogeneous instances, variable latency |
| Consistent Hash | No | Yes | Partial | Ring lookup and eligible-node traversal | Multi-turn conversations, KV cache reuse |
| Power of Two | Yes | No | No | Candidate preparation plus two load comparisons | Large clusters, high throughput |
| Cache-Aware | No | Prompt-based | Yes | Prefix tokenization and ring lookup | Shared system prompts, prefix caching |

> **N** = number of backend instances. All strategies also pay the shared
> candidate-filtering cost; affinity does not itself guarantee backend cache hits.
