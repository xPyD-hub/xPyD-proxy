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
    ["prefill_instance", "decode_instance", "model"],
    registry=REGISTRY,
)

proxy_kv_transfer_duration_seconds = Histogram(
    "proxy_kv_transfer_duration_seconds",
    "Estimated KV transfer time: T(decode_first_token) - T(prefill_response)",
    ["prefill_instance", "decode_instance", "model"],
    registry=REGISTRY,
)

proxy_decode_duration_seconds = Histogram(
    "proxy_decode_duration_seconds",
    "Total decode phase duration in seconds",
    ["prefill_instance", "decode_instance", "model"],
    registry=REGISTRY,
)

proxy_ttft_seconds = Histogram(
    "proxy_ttft_seconds",
    "End-to-end time to first token (user-perceived) in seconds",
    ["prefill_instance", "decode_instance", "model"],
    registry=REGISTRY,
)

proxy_tpot_seconds = Histogram(
    "proxy_tpot_seconds",
    "Average time per output token (decode phase) in seconds",
    ["prefill_instance", "decode_instance", "model"],
    registry=REGISTRY,
)

proxy_e2e_latency_seconds = Histogram(
    "proxy_e2e_latency_seconds",
    "Total end-to-end request latency in seconds",
    ["prefill_instance", "decode_instance", "model"],
    registry=REGISTRY,
)

proxy_prefill_active_requests = Gauge(
    "proxy_prefill_active_requests",
    "Number of requests currently in prefill stage",
    ["prefill_instance", "decode_instance", "model"],
    registry=REGISTRY,
)

proxy_decode_active_requests = Gauge(
    "proxy_decode_active_requests",
    "Number of requests currently in decode stage",
    ["prefill_instance", "decode_instance", "model"],
    registry=REGISTRY,
)

# NOTE: proxy_prefill_queue_depth tracks requests waiting for prefill.
# Currently no explicit queueing exists in the proxy; this gauge is
# initialised to 0 and should be incremented/decremented if a prefill
# queue is added in the future.
proxy_prefill_queue_depth = Gauge(
    "proxy_prefill_queue_depth",
    "Number of requests waiting for prefill (0 if no queueing exists)",
    ["prefill_instance", "decode_instance", "model"],
    registry=REGISTRY,
)

proxy_prefill_requests_total = Counter(
    "proxy_prefill_requests_total",
    "Total number of requests routed to each prefill instance",
    ["prefill_instance", "decode_instance", "model"],
    registry=REGISTRY,
)

proxy_decode_requests_total = Counter(
    "proxy_decode_requests_total",
    "Total number of requests routed to each decode instance",
    ["prefill_instance", "decode_instance", "model"],
    registry=REGISTRY,
)

proxy_instance_errors_total = Counter(
    "proxy_instance_errors_total",
    "Total number of errors per instance and error type",
    ["instance", "error_type", "model"],
    registry=REGISTRY,
)


class FirstTokenTracker:
    """Wraps an async generator to track timing of first and last chunks.

    Used to measure KV transfer time and decode duration without
    changing the generator behavior.

    Note: ``chunk_count`` counts raw HTTP response chunks, which may not
    correspond 1:1 with semantic tokens (SSE events can be split or merged
    at the TCP level). TPOT derived from chunk_count is therefore approximate.
    """

    def __init__(self, generator):
        self._gen = generator
        self.first_chunk_time: float | None = None
        self.last_chunk_time: float | None = None
        self.chunk_count: int = 0

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        async for chunk in self._gen:
            now = time.monotonic()
            if self.first_chunk_time is None:
                self.first_chunk_time = now
            self.last_chunk_time = now
            self.chunk_count += 1
            yield chunk


def record_pd_metrics(
    prefill_instance: str,
    decode_instance: str,
    model: str,
    t_request_start: float,
    t_prefill_done: float,
    tracker: FirstTokenTracker,
    is_streaming: bool = False,
) -> None:
    """Record PD disaggregation metrics after a request completes.

    Note on TTFT: In PD mode, the first token the user sees comes from the
    prefill node (max_tokens=1). So TTFT = t_prefill_done - t_request_start.
    The decode tracker measures the *decode* first-chunk time, which includes
    KV transfer overhead.

    Note on TPOT: ``tracker.chunk_count`` counts raw HTTP chunks, not semantic
    tokens. TPOT is therefore approximate and only meaningful for streaming
    requests where chunks roughly correspond to individual SSE events.
    """
    # E2E latency
    e2e = time.monotonic() - t_request_start
    proxy_e2e_latency_seconds.labels(
        prefill_instance=prefill_instance,
        decode_instance=decode_instance,
        model=model,
    ).observe(e2e)

    # Prefill duration
    proxy_prefill_duration_seconds.labels(
        prefill_instance=prefill_instance,
        decode_instance=decode_instance,
        model=model,
    ).observe(t_prefill_done - t_request_start)

    # TTFT — user sees first token from prefill, so TTFT ≈ prefill duration
    proxy_ttft_seconds.labels(
        prefill_instance=prefill_instance,
        decode_instance=decode_instance,
        model=model,
    ).observe(t_prefill_done - t_request_start)

    if tracker.first_chunk_time is not None:
        # KV transfer time (clamped to >= 0)
        kv_time = max(0.0, tracker.first_chunk_time - t_prefill_done)
        proxy_kv_transfer_duration_seconds.labels(
            prefill_instance=prefill_instance,
            decode_instance=decode_instance,
            model=model,
        ).observe(kv_time)

        # Decode duration
        if tracker.last_chunk_time is not None and tracker.chunk_count > 0:
            decode_duration = tracker.last_chunk_time - tracker.first_chunk_time
            proxy_decode_duration_seconds.labels(
                prefill_instance=prefill_instance,
                decode_instance=decode_instance,
                model=model,
            ).observe(decode_duration)

            # TPOT — only for streaming; chunk_count is approximate
            if is_streaming and tracker.chunk_count > 1:
                tpot = decode_duration / (tracker.chunk_count - 1)
                proxy_tpot_seconds.labels(
                    prefill_instance=prefill_instance,
                    decode_instance=decode_instance,
                    model=model,
                ).observe(tpot)
