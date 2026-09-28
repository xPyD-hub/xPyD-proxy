# Controlled scheduler semantics

`examples/lib/scheduler_semantics.py` runs the repository proxy with two
controllable HTTP backends per role. It requires only the proxy dependencies,
not vLLM or an accelerator:

```bash
python examples/lib/scheduler_semantics.py \
  --topology disaggregated --scheduler loadbalanced
```

For load-balanced and power-of-two, one request is held at a backend using an
explicit event while four other requests must avoid its reserved nodes.
With two candidates, power-of-two always samples both; there is no small-sample
coverage lottery. A round-robin replacement fails this assertion. Both P and D
reservations are checked in disaggregated mode.

Round-robin must repeat its cycle, consistent-hash must preserve a session
across different prompts, and cache-aware must preserve a full shared prefix
across different sessions. All request reservations must drain at the end.

The CPU NIXL and GPU aggregated matrices run these checks in addition to their
real inference smoke tests. The controlled backends test routing semantics,
not model correctness or real KV transfer. Logs remain in ignored `logs/`.
