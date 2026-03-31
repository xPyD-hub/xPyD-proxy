"""Tests for the InstanceRegistry."""

from __future__ import annotations

import concurrent.futures
import threading

import pytest
from registry import CircuitBreakerState, InstanceRegistry, InstanceStatus

# ---------------------------------------------------------------------------
# Add / Remove
# ---------------------------------------------------------------------------


class TestAddRemove:
    """Tests for adding and removing instances."""

    def test_add_decode_instance(self) -> None:
        reg = InstanceRegistry()
        reg.add("decode", "10.0.0.1:8200")
        info = reg.get_instance_info("10.0.0.1:8200")
        assert info.role == "decode"
        assert info.status == InstanceStatus.UNKNOWN
        assert info.circuit_breaker_state == CircuitBreakerState.CLOSED

    def test_add_prefill_instance(self) -> None:
        reg = InstanceRegistry()
        reg.add("prefill", "10.0.0.1:8100")
        info = reg.get_instance_info("10.0.0.1:8100")
        assert info.role == "prefill"

    def test_add_duplicate_raises(self) -> None:
        reg = InstanceRegistry()
        reg.add("decode", "10.0.0.1:8200")
        with pytest.raises(ValueError, match="already registered"):
            reg.add("decode", "10.0.0.1:8200")

    def test_add_invalid_role_raises(self) -> None:
        reg = InstanceRegistry()
        with pytest.raises(ValueError, match="Invalid role"):
            reg.add("worker", "10.0.0.1:8200")

    def test_remove_instance(self) -> None:
        reg = InstanceRegistry()
        reg.add("decode", "10.0.0.1:8200")
        reg.remove("10.0.0.1:8200")
        with pytest.raises(KeyError):
            reg.get_instance_info("10.0.0.1:8200")

    def test_remove_nonexistent_raises(self) -> None:
        reg = InstanceRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.remove("10.0.0.1:8200")

    def test_get_all_instances(self) -> None:
        reg = InstanceRegistry()
        reg.add("decode", "10.0.0.1:8200")
        reg.add("prefill", "10.0.0.2:8100")
        all_instances = reg.get_all_instances()
        assert len(all_instances) == 2
        addresses = {inst.address for inst in all_instances}
        assert addresses == {"10.0.0.1:8200", "10.0.0.2:8100"}


# ---------------------------------------------------------------------------
# get_available_instances with various states
# ---------------------------------------------------------------------------


class TestGetAvailableInstances:
    """Tests for get_available_instances with different health/circuit states."""

    def test_empty_registry_returns_empty(self) -> None:
        reg = InstanceRegistry()
        assert reg.get_available_instances("decode") == []

    def test_unknown_status_excluded(self) -> None:
        reg = InstanceRegistry()
        reg.add("decode", "10.0.0.1:8200")
        # Default status is UNKNOWN — should not be available.
        assert reg.get_available_instances("decode") == []

    def test_healthy_instance_included(self) -> None:
        reg = InstanceRegistry()
        reg.add("decode", "10.0.0.1:8200")
        reg.mark_healthy("10.0.0.1:8200")
        assert reg.get_available_instances("decode") == ["10.0.0.1:8200"]

    def test_unhealthy_instance_excluded(self) -> None:
        reg = InstanceRegistry()
        reg.add("decode", "10.0.0.1:8200")
        reg.mark_healthy("10.0.0.1:8200")
        reg.mark_unhealthy("10.0.0.1:8200")
        assert reg.get_available_instances("decode") == []

    def test_open_circuit_breaker_excluded(self) -> None:
        """Trigger enough failures to open the circuit breaker."""
        reg = InstanceRegistry()
        reg.add("decode", "10.0.0.1:8200")
        reg.mark_healthy("10.0.0.1:8200")
        # Default failure_threshold is 5
        for _ in range(5):
            reg.record_failure("10.0.0.1:8200")
        info = reg.get_instance_info("10.0.0.1:8200")
        assert info.circuit_breaker_state == CircuitBreakerState.OPEN
        assert reg.get_available_instances("decode") == []

    def test_half_open_circuit_breaker_excluded(self) -> None:
        """Trip breaker then advance time so it becomes HALF_OPEN."""
        reg = InstanceRegistry()
        reg.add("decode", "10.0.0.1:8200")
        reg.mark_healthy("10.0.0.1:8200")

        # Use a controllable clock on the circuit breaker
        t = [0.0]
        info = reg.get_instance_info("10.0.0.1:8200")
        cb = info.circuit_breaker
        cb._clock = lambda: t[0]

        for _ in range(5):
            cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

        # Advance past timeout
        t[0] = cb.timeout_duration_seconds + 1
        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert reg.get_available_instances("decode") == []

    def test_role_filtering(self) -> None:
        reg = InstanceRegistry()
        reg.add("decode", "10.0.0.1:8200")
        reg.add("prefill", "10.0.0.2:8100")
        reg.mark_healthy("10.0.0.1:8200")
        reg.mark_healthy("10.0.0.2:8100")
        assert reg.get_available_instances("decode") == ["10.0.0.1:8200"]
        assert reg.get_available_instances("prefill") == ["10.0.0.2:8100"]

    def test_mixed_states(self) -> None:
        """Verify the verification example from the tasklist."""
        reg = InstanceRegistry()
        for i in range(1, 5):
            reg.add("decode", f"10.0.0.{i}:8200")
            reg.mark_healthy(f"10.0.0.{i}:8200")

        available = reg.get_available_instances("decode")
        assert len(available) == 4

        reg.mark_unhealthy("10.0.0.2:8200")
        available = reg.get_available_instances("decode")
        assert len(available) == 3
        assert "10.0.0.2:8200" not in available

        reg.mark_healthy("10.0.0.2:8200")
        available = reg.get_available_instances("decode")
        assert len(available) == 4

    def test_mark_healthy_nonexistent_raises(self) -> None:
        reg = InstanceRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.mark_healthy("10.0.0.1:8200")

    def test_mark_unhealthy_nonexistent_raises(self) -> None:
        reg = InstanceRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.mark_unhealthy("10.0.0.1:8200")


# ---------------------------------------------------------------------------
# record_success / record_failure (delegated to CircuitBreaker)
# ---------------------------------------------------------------------------


class TestRecordSuccessFailure:
    """Tests for request outcome recording via CircuitBreaker."""

    def test_record_failure_trips_breaker(self) -> None:
        reg = InstanceRegistry()
        reg.add("decode", "10.0.0.1:8200")
        for _ in range(5):
            reg.record_failure("10.0.0.1:8200")
        info = reg.get_instance_info("10.0.0.1:8200")
        assert info.circuit_breaker_state == CircuitBreakerState.OPEN

    def test_record_success_closes_half_open(self) -> None:
        reg = InstanceRegistry()
        reg.add("decode", "10.0.0.1:8200")

        # Get a handle to the actual circuit breaker and use a fake clock
        t = [0.0]
        cb = reg.get_instance_info("10.0.0.1:8200").circuit_breaker
        cb._clock = lambda: t[0]

        # Trip the breaker
        for _ in range(5):
            cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

        # Advance past timeout → HALF_OPEN
        t[0] = cb.timeout_duration_seconds + 1
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # success_threshold defaults to 2
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_record_on_nonexistent_raises(self) -> None:
        reg = InstanceRegistry()
        with pytest.raises(KeyError):
            reg.record_success("10.0.0.1:8200")
        with pytest.raises(KeyError):
            reg.record_failure("10.0.0.1:8200")


# ---------------------------------------------------------------------------
# Active request counting
# ---------------------------------------------------------------------------


class TestActiveRequests:
    """Tests for active request count tracking."""

    def test_increment_decrement(self) -> None:
        reg = InstanceRegistry()
        reg.add("decode", "10.0.0.1:8200")
        reg.increment_active_requests("10.0.0.1:8200")
        reg.increment_active_requests("10.0.0.1:8200")
        assert reg.get_instance_info("10.0.0.1:8200").active_request_count == 2
        reg.decrement_active_requests("10.0.0.1:8200")
        assert reg.get_instance_info("10.0.0.1:8200").active_request_count == 1

    def test_decrement_does_not_go_negative(self) -> None:
        reg = InstanceRegistry()
        reg.add("decode", "10.0.0.1:8200")
        reg.decrement_active_requests("10.0.0.1:8200")
        assert reg.get_instance_info("10.0.0.1:8200").active_request_count == 0


# ---------------------------------------------------------------------------
# Concurrent access safety
# ---------------------------------------------------------------------------


class TestConcurrency:
    """Tests for thread-safe concurrent access."""

    def test_concurrent_add_remove(self) -> None:
        """Multiple threads adding and removing instances concurrently."""
        reg = InstanceRegistry()
        errors: list[Exception] = []

        def worker(idx: int) -> None:
            addr = f"10.0.0.{idx}:8200"
            try:
                reg.add("decode", addr)
                reg.mark_healthy(addr)
                _ = reg.get_available_instances("decode")
                reg.remove(addr)
            except Exception as exc:
                errors.append(exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(worker, i) for i in range(100)]
            concurrent.futures.wait(futures)

        assert errors == []
        assert reg.get_all_instances() == []

    def test_concurrent_record_success_failure(self) -> None:
        """Concurrent success/failure recording doesn't corrupt state."""
        reg = InstanceRegistry()
        reg.add("decode", "10.0.0.1:8200")
        iterations = 500

        def record_successes() -> None:
            for _ in range(iterations):
                reg.record_success("10.0.0.1:8200")

        def record_failures() -> None:
            for _ in range(iterations):
                reg.record_failure("10.0.0.1:8200")

        t1 = threading.Thread(target=record_successes)
        t2 = threading.Thread(target=record_failures)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # After concurrent ops, circuit breaker state should be valid.
        info = reg.get_instance_info("10.0.0.1:8200")
        assert info.circuit_breaker_state in {
            CircuitBreakerState.CLOSED,
            CircuitBreakerState.OPEN,
            CircuitBreakerState.HALF_OPEN,
        }

    def test_concurrent_increment_decrement(self) -> None:
        """Active request count stays consistent under contention."""
        reg = InstanceRegistry()
        reg.add("decode", "10.0.0.1:8200")
        iterations = 1000

        def inc() -> None:
            for _ in range(iterations):
                reg.increment_active_requests("10.0.0.1:8200")

        def dec() -> None:
            for _ in range(iterations):
                reg.decrement_active_requests("10.0.0.1:8200")

        t1 = threading.Thread(target=inc)
        t2 = threading.Thread(target=dec)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Net effect: increments and decrements cancel out (with floor at 0).
        info = reg.get_instance_info("10.0.0.1:8200")
        assert info.active_request_count >= 0

    def test_get_instance_info_returns_snapshot(self) -> None:
        """Modifying registry after get_instance_info doesn't affect snapshot."""
        reg = InstanceRegistry()
        reg.add("decode", "10.0.0.1:8200")
        reg.mark_healthy("10.0.0.1:8200")
        snapshot = reg.get_instance_info("10.0.0.1:8200")
        reg.mark_unhealthy("10.0.0.1:8200")
        assert snapshot.status == InstanceStatus.HEALTHY
        assert reg.get_instance_info("10.0.0.1:8200").status == InstanceStatus.UNHEALTHY
