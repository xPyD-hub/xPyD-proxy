# SPDX-License-Identifier: Apache-2.0
"""Tests for per-instance circuit breaker state machine."""

from __future__ import annotations

import pytest

from xpyd.circuit_breaker import CircuitBreaker, CircuitBreakerState

# ── helpers ──────────────────────────────────────────────────────────


class FakeClock:
    """Deterministic clock for testing time-dependent behaviour."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _make_cb(
    clock: FakeClock | None = None, **kwargs
) -> tuple[CircuitBreaker, FakeClock]:
    if clock is None:
        clock = FakeClock()
    cb = CircuitBreaker(clock=clock, **kwargs)
    return cb, clock


# ── initial state ────────────────────────────────────────────────────


class TestInitialState:
    def test_starts_closed(self):
        cb, _ = _make_cb()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_allows_requests_when_closed(self):
        cb, _ = _make_cb()
        assert cb.allow_request() is True


# ── CLOSED → OPEN ───────────────────────────────────────────────────


class TestClosedToOpen:
    def test_opens_after_failure_threshold(self):
        cb, clock = _make_cb(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

    def test_stays_closed_below_threshold(self):
        cb, clock = _make_cb(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_denies_requests_when_open(self):
        cb, clock = _make_cb(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.allow_request() is False

    def test_success_does_not_clear_failure_window(self):
        """A success in CLOSED state must not clear the failure window.

        The sliding window naturally expires old failures.  Clearing on
        success would mask intermittent-failure patterns (e.g. 4 failures,
        1 success, 4 more failures within the window never trips despite
        8/9 requests failing).
        """
        cb, clock = _make_cb(failure_threshold=5, window_duration_seconds=60)
        for _ in range(4):
            cb.record_failure()
        cb.record_success()  # should NOT clear the 4 failures
        cb.record_failure()  # 5th failure in window → trip
        assert cb.state == CircuitBreakerState.OPEN

    def test_failures_outside_window_are_ignored(self):
        cb, clock = _make_cb(failure_threshold=3, window_duration_seconds=10)
        cb.record_failure()
        clock.advance(6)
        cb.record_failure()
        clock.advance(6)
        # First failure is now 12s old (outside 10s window)
        cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED


# ── OPEN → HALF-OPEN ────────────────────────────────────────────────


class TestOpenToHalfOpen:
    def test_transitions_after_timeout(self):
        cb, clock = _make_cb(failure_threshold=2, timeout_duration_seconds=30)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

        clock.advance(30)
        assert cb.state == CircuitBreakerState.HALF_OPEN

    def test_stays_open_before_timeout(self):
        cb, clock = _make_cb(failure_threshold=2, timeout_duration_seconds=30)
        cb.record_failure()
        cb.record_failure()
        clock.advance(29)
        assert cb.state == CircuitBreakerState.OPEN

    def test_allows_probe_in_half_open(self):
        cb, clock = _make_cb(failure_threshold=2, timeout_duration_seconds=10)
        cb.record_failure()
        cb.record_failure()
        clock.advance(10)
        assert cb.allow_request() is True

    def test_denies_concurrent_probes_in_half_open(self):
        """HALF-OPEN must allow exactly one probe at a time."""
        cb, clock = _make_cb(failure_threshold=2, timeout_duration_seconds=10)
        cb.record_failure()
        cb.record_failure()
        clock.advance(10)

        assert cb.allow_request() is True  # first probe allowed
        assert cb.allow_request() is False  # second probe rejected
        assert cb.allow_request() is False  # still rejected

    def test_probe_slot_resets_after_failure(self):
        """After a probe failure the breaker re-opens and the probe slot
        is freed, so the next HALF-OPEN cycle can issue a new probe."""
        cb, clock = _make_cb(failure_threshold=2, timeout_duration_seconds=10)
        cb.record_failure()
        cb.record_failure()
        clock.advance(10)
        assert cb.allow_request() is True  # probe issued
        cb.record_failure()  # probe fails → OPEN
        assert cb.state == CircuitBreakerState.OPEN

        clock.advance(10)  # back to HALF-OPEN
        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert cb.allow_request() is True  # new probe allowed

    def test_probe_slot_resets_after_success_close(self):
        """After successful probes close the circuit, new requests are
        allowed normally."""
        cb, clock = _make_cb(
            failure_threshold=2,
            success_threshold=1,
            timeout_duration_seconds=10,
        )
        cb.record_failure()
        cb.record_failure()
        clock.advance(10)
        assert cb.allow_request() is True  # probe
        cb.record_success()  # close circuit
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.allow_request() is True  # normal traffic


# ── HALF-OPEN → CLOSED ──────────────────────────────────────────────


class TestHalfOpenToClosed:
    def test_closes_after_success_threshold(self):
        cb, clock = _make_cb(
            failure_threshold=2,
            success_threshold=2,
            timeout_duration_seconds=10,
        )
        cb.record_failure()
        cb.record_failure()
        clock.advance(10)
        assert cb.state == CircuitBreakerState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitBreakerState.HALF_OPEN  # not yet
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_interleaved_allow_request_and_success(self):
        """Realistic flow: allow_request() → record_success() per probe.

        When success_threshold > 1, the pending flag must be reset after
        each successful probe so that the next allow_request() is admitted.
        """
        cb, clock = _make_cb(
            failure_threshold=2,
            success_threshold=3,
            timeout_duration_seconds=10,
        )
        cb.record_failure()
        cb.record_failure()
        clock.advance(10)
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Probe 1
        assert cb.allow_request() is True
        cb.record_success()
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Probe 2 — must be allowed (pending flag was reset)
        assert cb.allow_request() is True
        cb.record_success()
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Probe 3 — final probe closes the circuit
        assert cb.allow_request() is True
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_allows_requests_after_closing(self):
        cb, clock = _make_cb(
            failure_threshold=2,
            success_threshold=1,
            timeout_duration_seconds=5,
        )
        cb.record_failure()
        cb.record_failure()
        clock.advance(5)
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.allow_request() is True


# ── HALF-OPEN → OPEN ────────────────────────────────────────────────


class TestHalfOpenToOpen:
    def test_reopens_on_probe_failure(self):
        cb, clock = _make_cb(failure_threshold=2, timeout_duration_seconds=10)
        cb.record_failure()
        cb.record_failure()
        clock.advance(10)
        assert cb.state == CircuitBreakerState.HALF_OPEN

        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

    def test_reopened_circuit_needs_timeout_again(self):
        cb, clock = _make_cb(failure_threshold=2, timeout_duration_seconds=10)
        # Trip → half-open → fail → re-open
        cb.record_failure()
        cb.record_failure()
        clock.advance(10)
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

        # Must wait another timeout
        clock.advance(9)
        assert cb.state == CircuitBreakerState.OPEN
        clock.advance(1)
        assert cb.state == CircuitBreakerState.HALF_OPEN


# ── Full cycle ──────────────────────────────────────────────────────


class TestFullCycle:
    def test_closed_open_half_open_closed(self):
        cb, clock = _make_cb(
            failure_threshold=3,
            success_threshold=2,
            timeout_duration_seconds=30,
        )
        assert cb.state == CircuitBreakerState.CLOSED

        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

        clock.advance(30)
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Realistic probe flow: allow_request() → record_success()
        assert cb.allow_request() is True
        cb.record_success()
        assert cb.state == CircuitBreakerState.HALF_OPEN

        assert cb.allow_request() is True
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED


# ── Config integration ──────────────────────────────────────────────


class TestCircuitBreakerConfig:
    def test_default_config(self):
        from xpyd.config import ProxyConfig

        cfg = ProxyConfig(model="m", decode=["127.0.0.1:8000"])
        assert cfg.circuit_breaker.enabled is False
        assert cfg.circuit_breaker.failure_threshold == 5
        assert cfg.circuit_breaker.success_threshold == 2
        assert cfg.circuit_breaker.timeout_duration_seconds == 30
        assert cfg.circuit_breaker.window_duration_seconds == 60

    def test_config_from_dict(self):
        from xpyd.config import CircuitBreakerConfig

        cb_cfg = CircuitBreakerConfig(enabled=True, failure_threshold=10)
        assert cb_cfg.enabled is True
        assert cb_cfg.failure_threshold == 10

    def test_config_rejects_unknown_fields(self):
        from pydantic import ValidationError

        from xpyd.config import CircuitBreakerConfig

        with pytest.raises(ValidationError, match="bogus"):
            CircuitBreakerConfig(enabled=True, bogus=42)
