# SPDX-License-Identifier: Apache-2.0
"""Tests for core/resilience.py — resilience with exponential backoff + jitter."""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock

import pytest
from resilience import ResilienceConfig, ResilienceHandler, compute_backoff

# ---------------------------------------------------------------------------
# compute_backoff
# ---------------------------------------------------------------------------


class TestComputeBackoff:
    """Verify backoff formula and jitter bounds."""

    def test_basic_backoff_attempt_zero(self):
        # delay = min(100 * 2^0, 10000) = 100
        # jitter_factor=0 → exact 100
        result = compute_backoff(0, 100, 2.0, 10000, 0.0)
        assert result == 100.0

    def test_basic_backoff_attempt_one(self):
        result = compute_backoff(1, 100, 2.0, 10000, 0.0)
        assert result == 200.0

    def test_backoff_capped_at_max(self):
        # 100 * 2^20 is huge, should be capped at 10000
        result = compute_backoff(20, 100, 2.0, 10000, 0.0)
        assert result == 10000.0

    def test_jitter_within_bounds(self):
        for _ in range(200):
            result = compute_backoff(0, 100, 2.0, 10000, 0.1)
            assert 90.0 <= result <= 110.0

    def test_jitter_within_bounds_higher_attempt(self):
        for _ in range(200):
            result = compute_backoff(2, 100, 2.0, 10000, 0.1)
            # base delay = 400, range: 360..440
            assert 360.0 <= result <= 440.0


# ---------------------------------------------------------------------------
# ResilienceHandler._should_retry
# ---------------------------------------------------------------------------


class TestShouldRetry:
    """Verify retry eligibility logic."""

    def setup_method(self):
        self.handler = ResilienceHandler(ResilienceConfig())

    def test_no_retry_4xx(self):
        for code in (400, 401, 403, 404, 422, 499):
            assert self.handler._should_retry(code, False) is False

    def test_retry_retryable_4xx(self):
        """408 and 429 are in retryable_status_codes by default."""
        assert self.handler._should_retry(408, False) is True
        assert self.handler._should_retry(429, False) is True

    def test_no_retry_streaming(self):
        assert self.handler._should_retry(502, True) is False

    def test_retry_retryable_codes(self):
        for code in (408, 429, 500, 502, 503, 504):
            assert self.handler._should_retry(code, False) is True

    def test_no_retry_non_retryable_5xx(self):
        assert self.handler._should_retry(501, False) is False

    def test_custom_retryable_codes(self):
        config = ResilienceConfig(retryable_status_codes=[418, 500])
        handler = ResilienceHandler(config)
        assert handler._should_retry(418, False) is True
        assert handler._should_retry(502, False) is False


# ---------------------------------------------------------------------------
# ResilienceHandler.execute
# ---------------------------------------------------------------------------


def _make_response(status_code: int, is_streaming: bool = False):
    resp = MagicMock()
    resp.status_code = status_code
    resp.is_streaming = is_streaming
    return resp


class TestRetryExecute:
    """Verify end-to-end retry execution."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        config = ResilienceConfig(enabled=True, max_retries=2)
        handler = ResilienceHandler(config)
        resp_ok = _make_response(200)

        request_fn = AsyncMock(return_value=resp_ok)
        select_fn = MagicMock(return_value="10.0.0.1:8200")

        result = await handler.execute(request_fn, select_fn)
        assert result.status_code == 200
        assert request_fn.await_count == 1

    @pytest.mark.asyncio
    async def test_retry_then_success(self):
        config = ResilienceConfig(
            enabled=True,
            max_retries=2,
            initial_backoff_ms=1,  # tiny for fast test
            jitter_factor=0.0,
        )
        handler = ResilienceHandler(config)

        responses = [_make_response(502), _make_response(200)]
        request_fn = AsyncMock(side_effect=responses)
        instances = iter(["10.0.0.1:8200", "10.0.0.2:8200", "10.0.0.3:8200"])
        select_fn = MagicMock(side_effect=lambda excluded=None: next(instances))

        result = await handler.execute(request_fn, select_fn)
        assert result.status_code == 200
        assert request_fn.await_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_fail(self):
        config = ResilienceConfig(
            enabled=True,
            max_retries=2,
            initial_backoff_ms=1,
            jitter_factor=0.0,
        )
        handler = ResilienceHandler(config)

        responses = [_make_response(502), _make_response(503), _make_response(500)]
        request_fn = AsyncMock(side_effect=responses)
        instances = iter(
            ["10.0.0.1:8200", "10.0.0.2:8200", "10.0.0.3:8200", "10.0.0.4:8200"]
        )
        select_fn = MagicMock(side_effect=lambda excluded=None: next(instances))

        result = await handler.execute(request_fn, select_fn)
        assert result.status_code == 500
        assert request_fn.await_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_4xx(self):
        config = ResilienceConfig(enabled=True, max_retries=2, initial_backoff_ms=1)
        handler = ResilienceHandler(config)

        request_fn = AsyncMock(return_value=_make_response(404))
        select_fn = MagicMock(return_value="10.0.0.1:8200")

        result = await handler.execute(request_fn, select_fn)
        assert result.status_code == 404
        assert request_fn.await_count == 1

    @pytest.mark.asyncio
    async def test_no_retry_on_streaming(self):
        config = ResilienceConfig(enabled=True, max_retries=2, initial_backoff_ms=1)
        handler = ResilienceHandler(config)

        request_fn = AsyncMock(return_value=_make_response(502, is_streaming=True))
        select_fn = MagicMock(return_value="10.0.0.1:8200")

        result = await handler.execute(request_fn, select_fn)
        assert result.status_code == 502
        assert request_fn.await_count == 1

    @pytest.mark.asyncio
    async def test_callbacks_invoked(self):
        config = ResilienceConfig(
            enabled=True,
            max_retries=1,
            initial_backoff_ms=1,
            jitter_factor=0.0,
        )
        handler = ResilienceHandler(config)

        responses = [_make_response(502), _make_response(200)]
        request_fn = AsyncMock(side_effect=responses)
        instances = iter(["a", "b", "c"])
        select_fn = MagicMock(side_effect=lambda excluded=None: next(instances))

        on_success = MagicMock()
        on_failure = MagicMock()

        result = await handler.execute(
            request_fn, select_fn, on_success=on_success, on_failure=on_failure
        )
        assert result.status_code == 200
        on_failure.assert_called_once()
        on_success.assert_called_once()

    @pytest.mark.anyio
    async def test_disabled_single_attempt_success(self):
        """When enabled=False, exactly one attempt is made (no retry)."""
        cfg = ResilienceConfig(enabled=False, max_retries=3)
        handler = ResilienceHandler(cfg)
        resp = MagicMock(status_code=200)
        request_fn = AsyncMock(return_value=resp)
        select_fn = MagicMock(return_value="node-1")
        on_success = MagicMock()
        result = await handler.execute(request_fn, select_fn, on_success=on_success)
        assert result.status_code == 200
        request_fn.assert_called_once_with("node-1")
        on_success.assert_called_once_with("node-1", resp)

    @pytest.mark.anyio
    async def test_disabled_single_attempt_failure(self):
        """When enabled=False and request fails, no retry is attempted."""
        cfg = ResilienceConfig(enabled=False, max_retries=3)
        handler = ResilienceHandler(cfg)
        resp = MagicMock(status_code=502)
        request_fn = AsyncMock(return_value=resp)
        select_fn = MagicMock(return_value="node-1")
        on_failure = MagicMock()
        result = await handler.execute(request_fn, select_fn, on_failure=on_failure)
        assert result.status_code == 502
        request_fn.assert_called_once_with("node-1")
        on_failure.assert_called_once_with("node-1", resp, 0)

    @pytest.mark.anyio
    async def test_select_instance_returns_none(self):
        """When select_instance_fn returns None during retry, stop retrying."""
        cfg = ResilienceConfig(enabled=True, max_retries=2, initial_backoff_ms=1)
        handler = ResilienceHandler(cfg)
        resp = MagicMock(status_code=502)
        request_fn = AsyncMock(return_value=resp)
        call_count = 0

        def select_fn(excluded=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "node-1"
            return None

        on_failure = MagicMock()
        result = await handler.execute(request_fn, select_fn, on_failure=on_failure)
        assert result.status_code == 502
        request_fn.assert_called_once_with("node-1")

    @pytest.mark.anyio
    async def test_select_instance_raises(self):
        """When select_instance_fn raises during retry, stop retrying."""
        cfg = ResilienceConfig(enabled=True, max_retries=2, initial_backoff_ms=1)
        handler = ResilienceHandler(cfg)
        resp = MagicMock(status_code=502)
        request_fn = AsyncMock(return_value=resp)
        call_count = 0

        def select_fn(excluded=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "node-1"
            raise RuntimeError("No instances available")

        on_failure = MagicMock()
        result = await handler.execute(request_fn, select_fn, on_failure=on_failure)
        assert result.status_code == 502
        request_fn.assert_called_once_with("node-1")


# ---------------------------------------------------------------------------
# Config YAML integration
# ---------------------------------------------------------------------------


class TestResilienceConfigYAML:
    """Verify ResilienceConfig integrates with ProxyConfig."""

    def test_default_disabled(self):
        config = ResilienceConfig()
        assert config.enabled is False
        assert config.max_retries == 2

    def test_from_yaml(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            """\
model: test-model
decode:
  - "127.0.0.1:8000"
retry:
  enabled: true
  max_retries: 3
  initial_backoff_ms: 200
  retryable_status_codes: [500, 502]
"""
        )
        args = types.SimpleNamespace(
            config=str(yaml_file),
            model=None,
            prefill=None,
            decode=None,
            port=8000,
            generator_on_p_node=False,
            roundrobin=False,
            log_level="warning",
            wait_timeout_seconds=600,
            probe_interval_seconds=10,
        )

        import config as config_mod

        proxy_cfg = config_mod.ProxyConfig.from_args(args)
        assert proxy_cfg.retry.enabled is True
        assert proxy_cfg.retry.max_retries == 3
        assert proxy_cfg.retry.initial_backoff_ms == 200
        assert proxy_cfg.retry.retryable_status_codes == [500, 502]

    def test_unknown_retry_key_rejected(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            """\
model: test-model
decode:
  - "127.0.0.1:8000"
retry:
  enabled: true
  bogus_key: 42
"""
        )
        args = types.SimpleNamespace(
            config=str(yaml_file),
            model=None,
            prefill=None,
            decode=None,
            port=8000,
            generator_on_p_node=False,
            roundrobin=False,
            log_level="warning",
            wait_timeout_seconds=600,
            probe_interval_seconds=10,
        )

        import config as config_mod

        with pytest.raises(Exception, match="bogus_key"):
            config_mod.ProxyConfig.from_args(args)
