# Monitoring Stack for xPyD-proxy

One-click Prometheus + Grafana deployment for PD disaggregation metrics.

## Quick Start

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

A pre-provisioned dashboard **"xPyD Proxy — PD Disaggregation Metrics"** is
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

All PD metrics carry `prefill_instance`, `decode_instance`, and `model` labels.

| Metric | Type | Description |
|--------|------|-------------|
| `proxy_prefill_duration_seconds` | Histogram | Prefill node response time |
| `proxy_kv_transfer_duration_seconds` | Histogram | KV transfer latency |
| `proxy_decode_duration_seconds` | Histogram | Decode phase duration |
| `proxy_ttft_seconds` | Histogram | End-to-end time to first token |
| `proxy_tpot_seconds` | Histogram | Average time per output token |
| `proxy_e2e_latency_seconds` | Histogram | Total request latency |
| `proxy_prefill_active_requests` | Gauge | Requests in prefill stage |
| `proxy_decode_active_requests` | Gauge | Requests in decode stage |
| `proxy_prefill_queue_depth` | Gauge | Requests waiting for prefill (**placeholder** — always 0 until explicit queueing is implemented) |
| `proxy_prefill_requests_total` | Counter | Requests per prefill instance |
| `proxy_decode_requests_total` | Counter | Requests per decode instance |
| `proxy_instance_errors_total` | Counter | Errors per instance and type |
