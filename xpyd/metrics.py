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


# --- PD Disaggregation Metrics ---

proxy_prefill_duration_seconds = Histogram(
    "proxy_prefill_duration_seconds",
    "Prefill node response time in seconds",
    ["prefill_instance"],
    registry=REGISTRY,
)

proxy_kv_transfer_seconds = Histogram(
    "proxy_kv_transfer_seconds",
    "Estimated KV transfer time: T(decode_first_token) - T(prefill_response)",
    ["prefill_instance", "decode_instance"],
    registry=REGISTRY,
)

proxy_decode_duration_seconds = Histogram(
    "proxy_decode_duration_seconds",
    "Total decode phase duration in seconds",
    ["decode_instance"],
    registry=REGISTRY,
)

proxy_ttft_seconds = Histogram(
    "proxy_ttft_seconds",
    "End-to-end time to first token (user-perceived) in seconds",
    ["endpoint"],
    registry=REGISTRY,
)

proxy_tpot_seconds = Histogram(
    "proxy_tpot_seconds",
    "Average time per output token (decode phase) in seconds",
    ["endpoint"],
    registry=REGISTRY,
)


class FirstTokenTracker:
    """Wraps an async generator to track timing of first and last chunks.

    Used to measure KV transfer time and decode duration without
    changing the generator behavior.
    """

    def __init__(self, generator):
        self._gen = generator
        self.first_token_time: float | None = None
        self.last_token_time: float | None = None
        self.token_count: int = 0

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        async for chunk in self._gen:
            now = time.monotonic()
            if self.first_token_time is None:
                self.first_token_time = now
            self.last_token_time = now
            self.token_count += 1
            yield chunk


def record_pd_metrics(
    endpoint: str,
    prefill_instance: str,
    decode_instance: str,
    t_request_start: float,
    t_prefill_done: float,
    tracker: FirstTokenTracker,
) -> None:
    """Record PD disaggregation metrics after a request completes."""
    # Prefill duration
    proxy_prefill_duration_seconds.labels(
        prefill_instance=prefill_instance,
    ).observe(t_prefill_done - t_request_start)

    if tracker.first_token_time is not None:
        # KV transfer time
        kv_time = tracker.first_token_time - t_prefill_done
        proxy_kv_transfer_seconds.labels(
            prefill_instance=prefill_instance,
            decode_instance=decode_instance,
        ).observe(kv_time)

        # TTFT
        ttft = tracker.first_token_time - t_request_start
        proxy_ttft_seconds.labels(endpoint=endpoint).observe(ttft)

        # Decode duration and TPOT
        if tracker.last_token_time is not None and tracker.token_count > 0:
            decode_duration = tracker.last_token_time - tracker.first_token_time
            proxy_decode_duration_seconds.labels(
                decode_instance=decode_instance,
            ).observe(decode_duration)

            if tracker.token_count > 1:
                tpot = decode_duration / (tracker.token_count - 1)
                proxy_tpot_seconds.labels(endpoint=endpoint).observe(tpot)
