# SPDX-License-Identifier: Apache-2.0
"""Prometheus-compatible metrics for MicroPDProxy."""

import time

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Use a dedicated registry to avoid default process/platform collectors
# that may not be relevant and can cause issues in testing.
REGISTRY = CollectorRegistry()

proxy_requests_total = Counter(
    "proxy_requests_total",
    "Total number of proxy requests",
    ["endpoint"],
    registry=REGISTRY,
)

proxy_request_duration_seconds = Histogram(
    "proxy_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint"],
    registry=REGISTRY,
)

proxy_active_requests = Gauge(
    "proxy_active_requests",
    "Number of currently active requests",
    registry=REGISTRY,
)


def track_request_start(endpoint: str) -> float:
    """Record the start of a request. Returns the start timestamp."""
    proxy_requests_total.labels(endpoint=endpoint).inc()
    proxy_active_requests.inc()
    return time.monotonic()


def track_request_end(endpoint: str, start: float) -> None:
    """Record the end of a request."""
    elapsed = time.monotonic() - start
    proxy_request_duration_seconds.labels(endpoint=endpoint).observe(elapsed)
    proxy_active_requests.dec()


def get_metrics() -> bytes:
    """Return Prometheus text-format metrics."""
    return generate_latest(REGISTRY)
