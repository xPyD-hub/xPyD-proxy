# Monitoring Stack for xPyD-proxy

One-click Prometheus + Grafana deployment for disaggregated serving metrics.

## Quick Start

For a self-contained example without Docker or an existing Prometheus server,
see [the runnable monitoring example](../examples/monitoring/prometheus/README.md).

```bash
cd monitoring
docker compose up -d
```

- **Prometheus**: [http://localhost:9090](http://localhost:9090)
- **Grafana**: [http://localhost:3000](http://localhost:3000) (admin / admin)

## Architecture

```
xPyD-proxy (:8000/metrics) → Prometheus (:9090) → Grafana (:3000)
```

## Grafana Dashboard

A pre-provisioned dashboard **"xPyD Proxy — Disaggregated Serving Metrics"** is
automatically loaded with panels for:

| Panel | Description |
|-------|-------------|
| Request Rate (P vs D) | Per-instance prefill/decode request throughput |
| TTFT Distribution | Time-to-first-token p50/p95/p99 |
| TPOT Distribution | Time-per-output-token p50/p95/p99 |
| KV Transfer Time | KV cache transfer latency distribution |
| Prefill vs Decode Duration | Side-by-side latency comparison |
| Per-Instance Load | Request rate per backend instance |
| Active Requests Over Time | Concurrent prefill/decode gauge |
| Error Rate | Per-instance error breakdown |

## Configuration

### Prometheus

Edit `prometheus/prometheus.yml` to point at your xPyD-proxy instance(s):

```yaml
static_configs:
  - targets: ["your-proxy-host:8000"]
```

### Adding More Scrape Targets

For multi-proxy deployments, add additional targets or use service discovery.

## Metrics Reference

Disaggregated timing and routing metrics carry `prefill_instance`,
`decode_instance`, and `model` labels. The error counter instead carries
`instance`, `error_type`, and `model`.

| Metric | Type | Description |
|--------|------|-------------|
| `proxy_prefill_duration_seconds` | Histogram | Proxy request start to complete prefill HTTP response |
| `proxy_kv_transfer_duration_seconds` | Histogram | Estimated transfer gap: decode first HTTP chunk minus complete prefill HTTP response |
| `proxy_decode_duration_seconds` | Histogram | Decode phase duration |
| `proxy_ttft_seconds` | Histogram | Proxy-observed first-token approximation, not a client-side measurement |
| `proxy_tpot_seconds` | Histogram | Average time per output token |
| `proxy_e2e_latency_seconds` | Histogram | Total request latency |
| `proxy_prefill_active_requests` | Gauge | Requests in prefill stage |
| `proxy_decode_active_requests` | Gauge | Requests in decode stage |
| `proxy_prefill_queue_depth` | Gauge | Requests waiting for prefill (**placeholder** — always 0 until explicit queueing is implemented) |
| `proxy_prefill_requests_total` | Counter | Requests per prefill instance |
| `proxy_decode_requests_total` | Counter | Requests per decode instance |
| `proxy_instance_errors_total` | Counter | Errors per instance and type |

### Simplified P/D timing

For **decode-first** requests, all timestamps are taken on the proxy's
monotonic clock. Let `t0` be request start, `tP` the time the complete prefill
HTTP response is received, and `tD` the arrival of the first decode HTTP chunk:

- Prefill HTTP time: `tP - t0`.
- TTFT approximation: `tD - t0`.
- Estimated KV transfer time: `max(0, tD - tP)`.

The third value is deliberately a proxy-side estimate. It includes decode
queueing, first-token computation and HTTP overhead, not only KV transport.
No vLLM or LMCache instrumentation is required. An HTTP chunk is not necessarily
one semantic token; non-streaming responses can arrive only after generation.
Use **streaming, decode-first** requests for this interpretation.

The difference is observed per request, before histogram aggregation.
Never subtract prefill p95 from TTFT p95 to estimate transfer p95; query
`proxy_kv_transfer_duration_seconds_bucket` directly. For prefill-first requests,
the exported TTFT uses `tP - t0`, so `TTFT - prefill` is not the transfer estimate.
TPOT is also approximate because it uses HTTP chunk counts.
