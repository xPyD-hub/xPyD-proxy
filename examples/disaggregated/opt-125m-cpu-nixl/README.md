# OPT-125M disaggregated CPU/NIXL TCP example

This example runs real 1P1D and multi-node prefill/decode topologies on the
same Linux host. Every instance serves `facebook/opt-125m` on CPU, while NIXL
transfers KV cache data through UCX over TCP. The 1P1D topology is the smallest
disaggregated counterpart to the aggregated OPT-125M CPU example.

Standard GitHub-hosted Linux runners do not provide GPUs. NIXL therefore must
be built with UCX from source instead of using its CUDA-oriented PyPI quick
install. The workflow caches the resulting wheel and caps each vLLM process at
512 MiB of KV cache. UCX uses the host's default-route network interface to
avoid zero-bandwidth loopback devices. A self-hosted GPU runner remains the
recommended target for performance testing; this CPU scenario validates
behavior only.

Run it on Ubuntu 22.04 with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
../../aggregated/opt-125m-cpu/install_vllm_cpu.sh
./install_nixl_cpu.sh
../../aggregated/opt-125m-cpu/download_model.sh
./run_all.sh
```

Run a multi-node lifecycle with its matching configuration and node counts:

```bash
./run_topology.sh xpyd_2p1d.yaml 2 1
./run_topology.sh xpyd_1p2d.yaml 1 2
./run_topology.sh xpyd_2p2d.yaml 2 2
```

The 1P1D lifecycle uses `run_all.sh`. Multi-node configurations use
`run_topology.sh <config> <prefill-count> <decode-count>`. Both start xPyD
before the backends, check HTTP 503 while the topology is incomplete, discover
every node, perform NIXL TCP inference, then validate prefill and decode loss
and reconnection independently. Multi-node tests also inspect per-instance
metrics to ensure round-robin requests exercised every configured node.
`/status/instances` and the concise disaggregated heartbeat are checked.

Every lifecycle also runs the shared endpoint checks from
`../../lib/proxy_api_smoke.sh`, the same suite the aggregated CPU example uses:
both completion APIs, `/ping` on both verbs, `/version`, `/status`,
`/status/instances`, the ten passthrough endpoints with their request
validation, every registered `OPTIONS` route, and the admin endpoint's
rejection paths. Each lifecycle finishes by adding a healthy backend alias
through `/instances/add`, then drains and removes a decode node through
`/instances/remove`. A remaining decode node continues serving in multi-node
topologies, while 1P1D correctly returns 503 after its only decode is removed.
This is what proves passthrough requests reach a backend in disaggregated mode
rather than failing to select one.
OPT-125M is a generative model, so the pooling and scoring families are answered
with a 4xx by vLLM itself; the checks require both a non-5xx response and a **new**
matching POST entry in the backend access logs. Proxy-generated 404/400
responses without backend activity fail. Run smoke tests without unrelated
concurrent traffic; standalone invocations can set `BACKEND_LOG_DIR` (default:
the example's `logs/`) and require vLLM access logging to remain enabled.
While the topology is
incomplete, the passthrough endpoints and `/health` are asserted to answer 503.
`run_all.sh` and `run_topology.sh` export a throwaway `ADMIN_API_KEY` so the
admin endpoint can be exercised on a loopback-only proxy.
Runtime output, including NIXL BUFFER telemetry required by NIXL v1.3, is
stored in the ignored `logs/` directory.
