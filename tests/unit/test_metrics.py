"""Tests for the /metrics Prometheus endpoint."""


from xpyd.metrics import (
    get_metrics,
    proxy_active_requests,
    proxy_requests_total,
    track_request_end,
    track_request_start,
)


class TestMetricsModule:
    """Unit tests for the metrics module."""

    def setup_method(self):
        """Reset counters before each test to isolate state."""
        # Clear label-specific children so counters start at 0.
        proxy_requests_total._metrics.clear()
        proxy_active_requests._value.set(0)

    def test_get_metrics_returns_bytes(self):
        output = get_metrics()
        assert isinstance(output, bytes)

    def test_get_metrics_contains_metric_names(self):
        # Trigger at least one label so the metric appears in output.
        proxy_requests_total.labels(endpoint="/v1/completions").inc()
        output = get_metrics().decode()
        assert "proxy_requests_total" in output
        assert "proxy_request_duration_seconds" in output
        assert "proxy_active_requests" in output

    def test_counter_increments(self):
        proxy_requests_total.labels(endpoint="/v1/completions").inc()
        proxy_requests_total.labels(endpoint="/v1/completions").inc()
        proxy_requests_total.labels(endpoint="/v1/chat/completions").inc()
        output = get_metrics().decode()
        # Should have 2.0 for completions, 1.0 for chat
        assert 'proxy_requests_total{endpoint="/v1/completions"} 2.0' in output
        assert 'proxy_requests_total{endpoint="/v1/chat/completions"} 1.0' in output

    def test_active_requests_gauge(self):
        proxy_active_requests.inc()
        proxy_active_requests.inc()
        output = get_metrics().decode()
        assert "proxy_active_requests 2.0" in output
        proxy_active_requests.dec()
        output = get_metrics().decode()
        assert "proxy_active_requests 1.0" in output

    def test_track_request_lifecycle(self):
        """track_request_start/end correctly update gauge and duration."""
        start = track_request_start("/v1/completions")
        output = get_metrics().decode()
        assert "proxy_active_requests 1.0" in output
        track_request_end("/v1/completions", start)
        output = get_metrics().decode()
        assert "proxy_active_requests 0.0" in output
        assert "proxy_request_duration_seconds" in output
