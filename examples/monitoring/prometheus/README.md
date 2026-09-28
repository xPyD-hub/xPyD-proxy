# Runnable Prometheus monitoring example

No existing Prometheus installation, Docker daemon, Grafana, or backend
instrumentation is required. The script downloads Prometheus **3.5.0** from its
official GitHub release, verifies its published SHA-256 checksum, and stores
the binaries under ignored `logs/tools/`. Internet access is required only for
the first download; Python needs this repository's runtime dependencies.

## Start a real inference demo

On an Intel XPU host with an XPU-enabled vLLM environment and a local model:

```bash
bash examples/monitoring/prometheus/run_all.sh \
  --model /workspace/Qwen3.5-0.8B \
  --device xpu --device-id 0 \
  --serve
```

Use `--device cuda` for a CUDA-enabled vLLM environment, or `--device cpu` for
a CPU-enabled vLLM environment and a compatible model. This option selects
the device environment; it does not install or convert a vLLM build.
The launcher uses the same Python interpreter for xPyD and vLLM. All services
bind to loopback by default; ports **18100**, **18868**, and **19090** must be available.

The command starts the repository proxy before the backend, confirms offline
503 responses, starts vLLM and Prometheus, and then performs:

1. Completion, chat, streaming, and invalid-request traffic.
2. Exact request-counter and duration-count deltas; observation of an active
   stream and return of every exposed active-request series to zero.
3. Real Prometheus scraping (`up=1`) and finite request-rate/p95 queries.
4. Proxy shutdown, `XPyDProxyDown` alert firing, proxy restart, alert resolution,
   and successful inference again.

Without `--serve`, the command exits after verification and stops its services.
With `--serve`, it keeps them running and sends a request every five seconds.
Attached disaggregated mode uses streaming requests for this continuous traffic.
Press **Ctrl-C** to stop owned services. It checks port release and preserves
logs; it never stops an existing attached backend/proxy. For accelerator runs,
also check device memory/processes with `xpu-smi ps` / `xpu-smi stats -d 0`
or `nvidia-smi` after exit, before restarting a backend.

Owned services run in separate process groups. Shutdown waits for their workers
even if the parent has already exited, escalating to SIGKILL if necessary.
Forced termination or failed cleanup is reported as an error with a nonzero exit
status, including when stopping with Ctrl-C or SIGTERM. During `--serve`, owned
processes are checked before and after each traffic request; if Prometheus or
another owned service exits, the demo reports the failure and cleans up.

## What you can see

| Address | Visible result |
|---|---|
| `http://127.0.0.1:19090/query` | PromQL queries and time-series graphs |
| `http://127.0.0.1:19090/targets` | `xpyd-demo` target, health and scrape errors |
| `http://127.0.0.1:19090/alerts` | `XPyDProxyDown` rule; inactive after recovery |
| `http://127.0.0.1:18868/metrics` | Raw proxy metrics |

### Open the frontend from another computer

Prometheus includes its own web frontend; the terminal-only machine does not
need a desktop or browser. It scrapes xPyD's `/metrics`, stores time series,
evaluates alerts, and serves the query/graph UI. Grafana is optional and is not
started by this example.

On the inference machine (A), run the XPU command above with **`--serve`** and
leave that terminal open. Without `--serve`, verification stops the services.
The owned proxy explicitly uses `host: 127.0.0.1`; the backend and, by default,
Prometheus also bind to loopback. Other proxy configurations retain the default
`host: 0.0.0.0` unless configured otherwise.

On the computer with a browser (B), open another terminal and run:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:19090:127.0.0.1:19090 user@machine-A
```

Keep that SSH connection open, then open **`http://127.0.0.1:19090/query` in B's
browser**. `/targets` shows whether scraping works and `/alerts` shows alert
state. No inbound firewall opening for 19090 or Grafana installation is needed.
If B's port 19090 is occupied, change only the first port to 29090 and browse
`http://127.0.0.1:29090/query`. To inspect raw metrics too, add
`-L 127.0.0.1:18868:127.0.0.1:18868` to the SSH command.

**Container boundary:** SSH forwarding resolves the destination `127.0.0.1`
in the SSH server's network namespace. If A's terminal is inside Docker but
SSH connects to the outer host, those loopback addresses are different (unless
using host networking). Connect to an existing SSH endpoint inside the
container, or forward 19090 through your existing remote-development/container
port-forwarding interface. An ordinary Docker port publication does not reach
a service bound only to the container's loopback. Do not change the services to
public listeners merely to make this tunnel work.

If the page fails, first check on A:
`curl -f http://127.0.0.1:19090/-/ready`.
If that succeeds, check the SSH tunnel, destination namespace, and B's local
port. If the page loads but charts are empty, check `/targets` for `up=1`, select
a recent time range, and keep demo traffic running.

#### Direct LAN access without an SSH tunnel

To explicitly expose only the Prometheus frontend, add
`--prometheus-host <Linux-IPv4-address>` to the demo command. For example, if
`192.0.2.10` is assigned inside the current network namespace (replace this
documentation address with your Linux machine's actual address):

```bash
bash examples/monitoring/prometheus/run_all.sh \
  --model /workspace/Qwen3.5-0.8B --device xpu --device-id 0 \
  --prometheus-host 192.0.2.10 --serve
```

From Windows, open `http://192.0.2.10:19090/query` in a browser; `/targets`
and `/alerts` are available on the same address. The proxy and backend remain
loopback-only. Prometheus has **no authentication** in this example; restrict
TCP 19090 to trusted clients using the host/network firewall. Do not expose it
to the public internet. This option does not change firewall rules.

With Docker host networking, the host IP can be bound directly. With bridge
networking, use `--prometheus-host 0.0.0.0` inside the container and publish
TCP 19090 on the host when creating the container. Preserve all existing
device mappings, mounts, and other settings if recreating a container; never
recreate a running inference container just to try a port mapping. The script
does not configure Docker or publish ports.

In Prometheus, try these expressions and select the graph view:

```promql
sum(rate(proxy_requests_total{job="xpyd-demo"}[1m]))
proxy_active_requests{job="xpyd-demo"}
1000 * histogram_quantile(0.95, sum by (le) (rate(proxy_request_duration_seconds_bucket{job="xpyd-demo"}[1m])))
```

The first is requests/second, the second is concurrent requests, and the third
is request latency p95 in milliseconds. Active requests can return to zero
between scrapes for short requests; the script also polls the raw endpoint
during a longer stream. A missing series is not treated as zero. Request
counters currently include validation failures and have no HTTP-status label:
they cannot by themselves provide a success/error-rate split.

Terminal output includes `PASS traffic`, `PASS gauges`, `PASS PromQL`,
`PASS alert`, and `PASS recovery`. Each run keeps `phases.log`, service logs,
`before.prom`, `after.prom`, `stream.txt`, and `queries.json` under
`logs/<timestamp>/`. These are evidence snapshots, not a permanent monitoring
deployment; the demo Prometheus retains one day of data while running.
The alert is visible in Prometheus; no Alertmanager notification is configured.

## Observe an existing backend or P/D proxy

An existing backend does not require vLLM to be installed in the script's
environment:

```bash
bash examples/monitoring/prometheus/run_all.sh \
  --backend-url http://127.0.0.1:8000 \
  --served-model-name my-model --serve
```

For an existing **streaming, decode-first** P/D deployment:

```bash
bash examples/monitoring/prometheus/run_all.sh \
  --proxy-url http://127.0.0.1:8868 \
  --served-model-name my-model \
  --mode disaggregated --serve
```

Use a dedicated test deployment without other traffic: the script deliberately
sends requests, including an invalid request, and checks exact metric deltas.
It does **not** stop an attached proxy to test alerts, and says so in its output.
The static alert-rule test still runs. An attached proxy may use another port;
only the demo's Prometheus port 19090 is reserved.

### The three P/D times

All three use the proxy's monotonic clock, without modifying vLLM or LMCache:

| Metric | Definition for decode-first streaming |
|---|---|
| `proxy_prefill_duration_seconds` | Complete P HTTP response received minus proxy request start |
| `proxy_ttft_seconds` | First D HTTP chunk received minus proxy request start |
| `proxy_kv_transfer_duration_seconds` | First D HTTP chunk minus complete P HTTP response (estimated KV transfer time) |

The last value includes D waiting, first-token computation and HTTP overhead.
It is **not pure hardware transfer time**. An HTTP chunk is not necessarily a
semantic token. Non-streaming requests and prefill-first TTFT have different
interpretations; do not use them to infer these three stages.

For one controlled streaming request, the script prints the three measured
values in milliseconds, checks `TTFT - prefill = estimated transfer`, and saves
`pd-timing.json`. It also queries all three histograms in Prometheus.
Histogram percentiles below cover the traffic in the window, which includes
the script's non-streaming smoke requests: the existing metrics do not label
streaming/non-streaming or first-token source separately. Use a dedicated
decode-first, streaming-only workload/window when interpreting the graphs as
the three-stage breakdown.

For example, estimated KV transfer p95:

```promql
1000 * histogram_quantile(
  0.95,
  sum by (le, model) (
    rate(proxy_kv_transfer_duration_seconds_bucket{job="xpyd-demo"}[1m])
  )
)
```

Replace the histogram name with `proxy_prefill_duration_seconds_bucket` or
`proxy_ttft_seconds_bucket` for the other two. Keep `prefill_instance` and
`decode_instance` in the `sum by` clause to separate node pairs.
The transfer difference is computed **per request** before histogram
aggregation. Never subtract two p95 values.

Aggregated deployments have no P/D stages, so the aggregated demo does not
claim to validate P/D histograms. Queue depth is currently a placeholder;
TPOT uses HTTP chunk counts and is approximate. These are documented rather
than presented as exact hardware measurements.

## CPU CI

```bash
python -m pytest tests/unit/test_monitoring_example.py tests/unit/test_disaggregated_metrics.py -q
python tests/monitoring/run_fixture.py
```

The second command runs real repository proxy and Prometheus processes against
**synthetic HTTP backends**, exercising aggregated and direct P/D monitoring.
It validates metric/alert plumbing, not inference quality or real KV transport.
Real inference is validated separately using the accelerator command above or
an attached real P/D deployment.
