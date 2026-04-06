# Metrics

## Overview

MicroPDProxy exposes a Prometheus-compatible `/metrics` endpoint for real-time observability of proxy behavior, including request counts, latency distributions, and in-flight request tracking.

## Available Metrics

| Metric | Type | Labels | Description |
|---|---|---|---|
| `proxy_requests_total` | Counter | `endpoint` | Total number of requests received |
| `proxy_request_duration_seconds` | Histogram | `endpoint` | Request duration including full streaming lifetime |
| `proxy_active_requests` | Gauge | — | Number of currently in-flight requests |

## How It Works

Metrics collection uses the **RequestTracker** pattern:

1. `start_request()` is called when a request arrives, incrementing the active gauge and the total counter, and starting a duration timer.
2. `finish()` is called when the stream completes, decrementing the active gauge and recording the duration in the histogram.

`finish()` is **idempotent** — it is safe to call multiple times (e.g., in both normal completion and error paths) without double-counting.

## Endpoint

```
GET /metrics
```

Returns metrics in Prometheus text exposition format with `Content-Type: text/plain`.

## Grafana Examples

**QPS (requests per second):**

```promql
rate(proxy_requests_total[5m])
```

**P99 latency:**

```promql
histogram_quantile(0.99, rate(proxy_request_duration_seconds_bucket[5m]))
```

**Active requests:**

```promql
proxy_active_requests
```

## Configuration

Metrics are **always enabled** — no configuration is needed. The `/metrics` endpoint is available as soon as the proxy starts.

> **Note:** MicroPDProxy uses a dedicated `CollectorRegistry` to avoid exposing default process collectors, keeping the metrics output clean and focused on proxy-specific data.
