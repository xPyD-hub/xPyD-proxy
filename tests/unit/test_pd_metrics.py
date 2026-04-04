"""Tests for PD disaggregation metrics."""

from __future__ import annotations

import pytest

from xpyd.metrics import (
    FirstTokenTracker,
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
            t_request_start=1.0,
            t_prefill_done=1.3,
            tracker=tracker,
        )
