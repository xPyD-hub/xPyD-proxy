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
    async def test_tracks_first_chunk_time(self):
        async def gen():
            yield b"token1"
            yield b"token2"
            yield b"token3"

        tracker = FirstTokenTracker(gen())
        chunks = []
        async for chunk in tracker:
            chunks.append(chunk)

        assert len(chunks) == 3
        assert tracker.first_chunk_time is not None
        assert tracker.last_chunk_time is not None
        assert tracker.chunk_count == 3
        assert tracker.first_chunk_time <= tracker.last_chunk_time

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
        assert tracker.first_chunk_time is None
        assert tracker.last_chunk_time is None
        assert tracker.chunk_count == 0

    @pytest.mark.asyncio
    async def test_single_token(self):
        async def gen():
            yield b"only"

        tracker = FirstTokenTracker(gen())
        chunks = []
        async for chunk in tracker:
            chunks.append(chunk)

        assert tracker.chunk_count == 1
        assert tracker.first_chunk_time == tracker.last_chunk_time

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
        tracker.first_chunk_time = 1.5
        tracker.last_chunk_time = 2.5
        tracker.chunk_count = 11

        # Should not raise
        record_pd_metrics(
            prefill_instance="10.0.0.1:8001",
            decode_instance="10.0.0.2:8002",
            model="test-model",
            t_request_start=1.0,
            t_prefill_done=1.3,
            tracker=tracker,
            is_streaming=True,
        )

    def test_no_tokens_received(self):
        tracker = FirstTokenTracker.__new__(FirstTokenTracker)
        tracker.first_chunk_time = None
        tracker.last_chunk_time = None
        tracker.chunk_count = 0

        # Should not raise — just records prefill duration
        record_pd_metrics(
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
        tracker.first_chunk_time = 1.5
        tracker.last_chunk_time = 2.5
        tracker.chunk_count = 5

        record_pd_metrics(
            prefill_instance="10.0.0.1:8001",
            decode_instance="10.0.0.2:8002",
            model="llama-70b",
            t_request_start=1.0,
            t_prefill_done=1.3,
            tracker=tracker,
            is_streaming=True,
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

    def test_non_streaming_skips_tpot(self):
        """Non-streaming requests should skip TPOT but record other metrics."""
        tracker = FirstTokenTracker.__new__(FirstTokenTracker)
        tracker.first_chunk_time = 1.5
        tracker.last_chunk_time = 2.5
        tracker.chunk_count = 5

        record_pd_metrics(
            prefill_instance="10.0.0.3:8001",
            decode_instance="10.0.0.4:8002",
            model="non-stream-model",
            t_request_start=1.0,
            t_prefill_done=1.3,
            tracker=tracker,
            is_streaming=False,
        )

        # TPOT should NOT be recorded for non-streaming
        tpot_sample = REGISTRY.get_sample_value(
            "proxy_tpot_seconds_count",
            {
                "prefill_instance": "10.0.0.3:8001",
                "decode_instance": "10.0.0.4:8002",
                "model": "non-stream-model",
            },
        )
        assert tpot_sample is None

        # But decode duration SHOULD be recorded
        decode_sample = REGISTRY.get_sample_value(
            "proxy_decode_duration_seconds_count",
            {
                "prefill_instance": "10.0.0.3:8001",
                "decode_instance": "10.0.0.4:8002",
                "model": "non-stream-model",
            },
        )
        assert decode_sample is not None and decode_sample > 0


class TestNewMetricDefinitions:
    """Verify all metrics from Issue #129 are defined with correct labels."""

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
        """Counters can be incremented with all three labels."""
        proxy_prefill_requests_total.labels(
            prefill_instance="test:8001",
            decode_instance="test:8002",
            model="test-model",
        ).inc()
        proxy_decode_requests_total.labels(
            prefill_instance="test:8001",
            decode_instance="test:8002",
            model="test-model",
        ).inc()
        proxy_instance_errors_total.labels(
            instance="test:8001",
            error_type="timeout",
            model="test-model",
        ).inc()

    def test_gauges_inc_dec(self):
        """Gauges can be incremented and decremented with all three labels."""
        proxy_prefill_active_requests.labels(
            prefill_instance="test:8001",
            decode_instance="test:8002",
            model="test-model",
        ).inc()
        proxy_prefill_active_requests.labels(
            prefill_instance="test:8001",
            decode_instance="test:8002",
            model="test-model",
        ).dec()
        proxy_decode_active_requests.labels(
            prefill_instance="test:8001",
            decode_instance="test:8002",
            model="test-model",
        ).inc()
        proxy_decode_active_requests.labels(
            prefill_instance="test:8001",
            decode_instance="test:8002",
            model="test-model",
        ).dec()


class TestStreamingFlag:
    def test_non_streaming_skips_tpot(self):
        """When is_streaming=False, TPOT should not be recorded."""
        tracker = FirstTokenTracker.__new__(FirstTokenTracker)
        tracker.first_chunk_time = 1.5
        tracker.last_chunk_time = 2.5
        tracker.chunk_count = 11

        from xpyd.metrics import REGISTRY

        # Get current TPOT count before
        before = (
            REGISTRY.get_sample_value(
                "proxy_tpot_seconds_count",
                {
                    "prefill_instance": "skip-test:8001",
                    "decode_instance": "skip-test:8002",
                    "model": "skip-model",
                },
            )
            or 0
        )

        record_pd_metrics(
            prefill_instance="skip-test:8001",
            decode_instance="skip-test:8002",
            model="skip-model",
            t_request_start=1.0,
            t_prefill_done=1.3,
            tracker=tracker,
            is_streaming=False,  # non-streaming
        )

        after = (
            REGISTRY.get_sample_value(
                "proxy_tpot_seconds_count",
                {
                    "prefill_instance": "skip-test:8001",
                    "decode_instance": "skip-test:8002",
                    "model": "skip-model",
                },
            )
            or 0
        )

        # TPOT should NOT have been recorded
        assert after == before, "TPOT should not be recorded for non-streaming requests"


class TestTTFTPaths:
    """Test TTFT calculation for P-first and D-first token paths."""

    def test_p_first_ttft_uses_prefill_duration(self):
        """When first_token_from_prefill=True, TTFT = prefill duration."""
        tracker = FirstTokenTracker.__new__(FirstTokenTracker)
        tracker.first_chunk_time = 3.0  # decode first chunk (should be ignored)
        tracker.last_chunk_time = 4.0
        tracker.chunk_count = 5

        record_pd_metrics(
            prefill_instance="ttft-p:8001",
            decode_instance="ttft-p:8002",
            model="ttft-p-model",
            t_request_start=1.0,
            t_prefill_done=1.5,  # prefill duration = 0.5s
            tracker=tracker,
            first_token_from_prefill=True,
        )

        ttft_sum = REGISTRY.get_sample_value(
            "proxy_ttft_seconds_sum",
            {
                "prefill_instance": "ttft-p:8001",
                "decode_instance": "ttft-p:8002",
                "model": "ttft-p-model",
            },
        )
        assert ttft_sum is not None
        # TTFT should be ~0.5 (prefill duration), not ~2.0 (decode first chunk)
        assert abs(ttft_sum - 0.5) < 0.01

    def test_d_first_ttft_uses_decode_first_chunk(self):
        """When first_token_from_prefill=False, TTFT = decode first chunk time."""
        tracker = FirstTokenTracker.__new__(FirstTokenTracker)
        tracker.first_chunk_time = 3.0  # decode first chunk at t=3.0
        tracker.last_chunk_time = 4.0
        tracker.chunk_count = 5

        record_pd_metrics(
            prefill_instance="ttft-d:8001",
            decode_instance="ttft-d:8002",
            model="ttft-d-model",
            t_request_start=1.0,
            t_prefill_done=1.5,
            tracker=tracker,
            first_token_from_prefill=False,
        )

        ttft_sum = REGISTRY.get_sample_value(
            "proxy_ttft_seconds_sum",
            {
                "prefill_instance": "ttft-d:8001",
                "decode_instance": "ttft-d:8002",
                "model": "ttft-d-model",
            },
        )
        assert ttft_sum is not None
        # TTFT should be ~2.0 (decode first chunk - request start), not ~0.5
        assert abs(ttft_sum - 2.0) < 0.01
