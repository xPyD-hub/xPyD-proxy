"""Tests for PD disaggregation metrics."""

from __future__ import annotations

import pytest

from xpyd.metrics import (
    REGISTRY,
    FirstTokenTracker,
    proxy_decode_active_requests,
    proxy_decode_requests_total,
    proxy_e2e_latency_seconds,
    proxy_instance_errors_total,
    proxy_kv_transfer_duration_seconds,
    proxy_prefill_active_requests,
    proxy_prefill_queue_depth,
    proxy_prefill_requests_total,
    record_pd_metrics,
)


class TestFirstTokenTracker:
    @pytest.mark.asyncio
    async def test_tracks_first_token_time(self):
        async def gen():
            yield b"token1"
            yield b"token2"
            yield b"token3"

        tracker = FirstTokenTracker(gen())
        chunks = []
        async for chunk in tracker:
            chunks.append(chunk)

        assert len(chunks) == 3
        assert tracker.first_token_time is not None
        assert tracker.last_token_time is not None
        assert tracker.token_count == 3
        assert tracker.first_token_time <= tracker.last_token_time

    @pytest.mark.asyncio
    async def test_empty_generator(self):
        async def gen():
            return
            yield

        tracker = FirstTokenTracker(gen())
        chunks = []
        async for chunk in tracker:
            chunks.append(chunk)

        assert len(chunks) == 0
        assert tracker.first_token_time is None
        assert tracker.last_token_time is None
        assert tracker.token_count == 0

    @pytest.mark.asyncio
    async def test_single_token(self):
        async def gen():
            yield b"only"

        tracker = FirstTokenTracker(gen())
        chunks = []
        async for chunk in tracker:
            chunks.append(chunk)

        assert tracker.token_count == 1
        assert tracker.first_token_time == tracker.last_token_time

    @pytest.mark.asyncio
    async def test_preserves_chunks(self):
        """Tracker must not modify the chunks."""
        data = [b"chunk1", b"chunk2", b"chunk3"]

        async def gen():
            for d in data:
                yield d

        tracker = FirstTokenTracker(gen())
        result = [chunk async for chunk in tracker]
        assert result == data


class TestRecordPdMetrics:
    def test_records_all_metrics(self):
        tracker = FirstTokenTracker.__new__(FirstTokenTracker)
        tracker.first_token_time = 1.5
        tracker.last_token_time = 2.5
        tracker.token_count = 11

        # Should not raise
        record_pd_metrics(
            endpoint="/v1/completions",
            prefill_instance="10.0.0.1:8001",
            decode_instance="10.0.0.2:8002",
            model="test-model",
            t_request_start=1.0,
            t_prefill_done=1.3,
            tracker=tracker,
        )

    def test_no_tokens_received(self):
        tracker = FirstTokenTracker.__new__(FirstTokenTracker)
        tracker.first_token_time = None
        tracker.last_token_time = None
        tracker.token_count = 0

        # Should not raise — just records prefill duration
        record_pd_metrics(
            endpoint="/v1/completions",
            prefill_instance="10.0.0.1:8001",
            decode_instance="10.0.0.2:8002",
            model="test-model",
            t_request_start=1.0,
            t_prefill_done=1.3,
            tracker=tracker,
        )

    def test_model_label_present(self):
        """Verify the model label is recorded on all PD metrics."""
        tracker = FirstTokenTracker.__new__(FirstTokenTracker)
        tracker.first_token_time = 1.5
        tracker.last_token_time = 2.5
        tracker.token_count = 5

        record_pd_metrics(
            endpoint="/v1/completions",
            prefill_instance="10.0.0.1:8001",
            decode_instance="10.0.0.2:8002",
            model="llama-70b",
            t_request_start=1.0,
            t_prefill_done=1.3,
            tracker=tracker,
        )

        # Check that the model label is set on the e2e histogram
        sample = REGISTRY.get_sample_value(
            "proxy_e2e_latency_seconds_count",
            {
                "prefill_instance": "10.0.0.1:8001",
                "decode_instance": "10.0.0.2:8002",
                "model": "llama-70b",
            },
        )
        assert sample is not None and sample > 0


class TestNewMetricDefinitions:
    """Verify all 7 new metrics from Issue #129 are defined."""

    def test_e2e_latency_exists(self):
        assert proxy_e2e_latency_seconds is not None

    def test_prefill_active_requests_exists(self):
        assert proxy_prefill_active_requests is not None

    def test_decode_active_requests_exists(self):
        assert proxy_decode_active_requests is not None

    def test_prefill_queue_depth_exists(self):
        assert proxy_prefill_queue_depth is not None

    def test_prefill_requests_total_exists(self):
        assert proxy_prefill_requests_total is not None

    def test_decode_requests_total_exists(self):
        assert proxy_decode_requests_total is not None

    def test_instance_errors_total_exists(self):
        assert proxy_instance_errors_total is not None

    def test_kv_transfer_renamed(self):
        """proxy_kv_transfer_duration_seconds should exist (renamed)."""
        assert proxy_kv_transfer_duration_seconds is not None

    def test_counters_increment(self):
        """Counters can be incremented with model label."""
        proxy_prefill_requests_total.labels(
            prefill_instance="test:8001",
            model="test-model",
        ).inc()
        proxy_decode_requests_total.labels(
            decode_instance="test:8002",
            model="test-model",
        ).inc()
        proxy_instance_errors_total.labels(
            instance="test:8001",
            error_type="timeout",
            model="test-model",
        ).inc()

    def test_gauges_inc_dec(self):
        """Gauges can be incremented and decremented."""
        proxy_prefill_active_requests.labels(
            prefill_instance="test:8001",
            model="test-model",
        ).inc()
        proxy_prefill_active_requests.labels(
            prefill_instance="test:8001",
            model="test-model",
        ).dec()
        proxy_decode_active_requests.labels(
            decode_instance="test:8002",
            model="test-model",
        ).inc()
        proxy_decode_active_requests.labels(
            decode_instance="test:8002",
            model="test-model",
        ).dec()
