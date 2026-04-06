"""Centralized Instance Registry for tracking instance state.

Provides a single source of truth for all prefill/decode instance metadata
including health status, circuit breaker state, and request counters.
Thread-safe for concurrent access from multiple async tasks.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from xpyd.circuit_breaker import CircuitBreaker, CircuitBreakerState


class InstanceStatus(str, Enum):
    """Health status of an instance."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class InstanceInfo:
    """Read-only snapshot of per-instance state returned by the registry.

    This is a pure data snapshot — it does **not** hold a reference to any
    mutable internal object such as a ``CircuitBreaker``.
    """

    address: str
    role: str  # "prefill" or "decode"
    model: str = ""  # model name this instance serves
    status: InstanceStatus = InstanceStatus.UNKNOWN
    last_health_check: Optional[float] = None
    active_request_count: int = 0
    circuit_breaker_state: CircuitBreakerState = CircuitBreakerState.CLOSED


@dataclass
class _InstanceRecord:
    """Internal mutable record stored inside the registry."""

    address: str
    role: str
    model: str = ""  # model name this instance serves
    status: InstanceStatus = InstanceStatus.UNKNOWN
    last_health_check: Optional[float] = None
    active_request_count: int = 0
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)


class InstanceRegistry:
    """Centralized registry tracking every prefill/decode instance's state.

    All access is protected by a reentrant lock for thread safety.
    """

    def __init__(
        self,
        clock: Optional[Callable[[], float]] = None,
        cb_enabled: bool = False,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_duration_seconds: float = 30,
        window_duration_seconds: float = 60,
    ) -> None:
        self._lock = threading.RLock()
        self._instances: Dict[str, _InstanceRecord] = {}
        self._clock = clock
        self._cb_enabled = cb_enabled
        self._cb_failure_threshold = failure_threshold
        self._cb_success_threshold = success_threshold
        self._cb_timeout_duration_seconds = timeout_duration_seconds
        self._cb_window_duration_seconds = window_duration_seconds

    def add(self, role: str, address: str, model: str = "") -> None:
        """Register an instance with the given role, address, and model.

        Args:
            role: Instance role, either "prefill" or "decode".
            address: Instance address (e.g. "10.0.0.1:8200").
            model: Model name this instance serves. Empty string means
                unspecified (backward compatible).

        Raises:
            ValueError: If role is not "prefill" or "decode".
            ValueError: If address is already registered.
        """
        if role not in ("prefill", "decode", "dual"):
            raise ValueError(f"Invalid role: {role!r}. Must be 'prefill', 'decode', or 'dual'.")
        with self._lock:
            if address in self._instances:
                raise ValueError(f"Instance {address!r} is already registered.")
            self._instances[address] = _InstanceRecord(
                address=address,
                role=role,
                model=model,
                circuit_breaker=CircuitBreaker(
                    failure_threshold=self._cb_failure_threshold,
                    success_threshold=self._cb_success_threshold,
                    timeout_duration_seconds=self._cb_timeout_duration_seconds,
                    window_duration_seconds=self._cb_window_duration_seconds,
                    clock=self._clock,
                ),
            )

    def remove(self, address: str) -> None:
        """Remove an instance from the registry.

        Args:
            address: Instance address to remove.

        Raises:
            KeyError: If address is not registered.
        """
        with self._lock:
            if address not in self._instances:
                raise KeyError(f"Instance {address!r} is not registered.")
            del self._instances[address]

    def get_available_instances(self, role: str, model: str = "") -> List[str]:
        """Return addresses of healthy instances with closed circuit breakers.

        Args:
            role: Filter by role ("prefill" or "decode").
            model: Filter by model name. When empty string (default),
                no model filtering is applied (backward compat).

        Returns:
            List of instance addresses that are healthy and have a closed
            circuit breaker.  HALF_OPEN instances are reserved for the
            circuit-breaker probe mechanism and excluded here.
        """
        with self._lock:
            results = []
            for instance in self._instances.values():
                if instance.role != role:
                    continue
                if model and instance.model != model:
                    continue
                if instance.status != InstanceStatus.HEALTHY:
                    continue
                if (
                    self._cb_enabled
                    and instance.circuit_breaker.state != CircuitBreakerState.CLOSED
                ):
                    continue
                results.append(instance.address)
            return results

    def get_dual_instances(self, model: str = "") -> List[str]:
        """Return available dual instances, optionally filtered by model.

        Uses the same availability criteria as get_available_instances:
        only healthy instances with closed circuit breakers are returned.
        HALF_OPEN instances are reserved for the circuit-breaker probe
        mechanism (via the health monitor) and excluded from scheduling.
        """
        with self._lock:
            results = []
            for instance in self._instances.values():
                if instance.role != "dual":
                    continue
                if model and instance.model != model:
                    continue
                if instance.status != InstanceStatus.HEALTHY:
                    continue
                if (
                    self._cb_enabled
                    and instance.circuit_breaker.state != CircuitBreakerState.CLOSED
                ):
                    continue
                results.append(instance.address)
            return results

    def get_registered_models(self) -> List[str]:
        """Return unique model names across all registered instances.

        Returns:
            Sorted list of unique non-empty model names.
        """
        with self._lock:
            models = {
                inst.model
                for inst in self._instances.values()
                if inst.model
            }
            return sorted(models)

    def update_model(self, address: str, model: str) -> None:
        """Update the model name for a registered instance.

        Args:
            address: Instance address.
            model: New model name.

        Raises:
            KeyError: If address is not registered.
        """
        with self._lock:
            instance = self._get_instance(address)
            instance.model = model

    def mark_healthy(self, address: str) -> None:
        """Mark an instance as healthy (called by health monitor).

        Args:
            address: Instance address.

        Raises:
            KeyError: If address is not registered.
        """
        with self._lock:
            instance = self._get_instance(address)
            instance.status = InstanceStatus.HEALTHY
            instance.last_health_check = time.monotonic()

    def mark_unhealthy(self, address: str) -> None:
        """Mark an instance as unhealthy (called by health monitor).

        Args:
            address: Instance address.

        Raises:
            KeyError: If address is not registered.
        """
        with self._lock:
            instance = self._get_instance(address)
            instance.status = InstanceStatus.UNHEALTHY
            instance.last_health_check = time.monotonic()

    def record_success(self, address: str) -> None:
        """Record a successful request to an instance.

        Delegates to the instance's CircuitBreaker.

        Args:
            address: Instance address.

        Raises:
            KeyError: If address is not registered.
        """
        with self._lock:
            instance = self._get_instance(address)
            instance.circuit_breaker.record_success()

    def record_failure(self, address: str) -> None:
        """Record a failed request to an instance.

        Delegates to the instance's CircuitBreaker.

        Args:
            address: Instance address.

        Raises:
            KeyError: If address is not registered.
        """
        with self._lock:
            instance = self._get_instance(address)
            instance.circuit_breaker.record_failure()

    def get_instance_info(self, address: str) -> InstanceInfo:
        """Return a snapshot of instance info.

        Args:
            address: Instance address.

        Returns:
            Copy of the InstanceInfo for the given address.

        Raises:
            KeyError: If address is not registered.
        """
        with self._lock:
            instance = self._get_instance(address)
            return InstanceInfo(
                address=instance.address,
                role=instance.role,
                model=instance.model,
                status=instance.status,
                last_health_check=instance.last_health_check,
                active_request_count=instance.active_request_count,
                circuit_breaker_state=instance.circuit_breaker.state,
            )

    def get_all_instances(self) -> List[InstanceInfo]:
        """Return a snapshot of all registered instances.

        Returns:
            List of InstanceInfo copies.
        """
        with self._lock:
            return [self.get_instance_info(addr) for addr in self._instances]

    def increment_active_requests(self, address: str) -> None:
        """Increment the active request count for an instance.

        Args:
            address: Instance address.

        Raises:
            KeyError: If address is not registered.
        """
        with self._lock:
            instance = self._get_instance(address)
            instance.active_request_count += 1

    def decrement_active_requests(self, address: str) -> None:
        """Decrement the active request count for an instance.

        Args:
            address: Instance address.

        Raises:
            KeyError: If address is not registered.
        """
        with self._lock:
            instance = self._get_instance(address)
            instance.active_request_count = max(0, instance.active_request_count - 1)

    def get_active_requests(self, address: str) -> int:
        """Return the active request count for an instance.

        Args:
            address: Instance address.

        Returns:
            Active request count, or 0 if not registered.
        """
        with self._lock:
            try:
                return self._get_instance(address).active_request_count
            except KeyError:
                return 0

    def _get_instance(self, address: str) -> _InstanceRecord:
        """Get instance by address or raise KeyError. Must hold lock."""
        try:
            return self._instances[address]
        except KeyError:
            raise KeyError(f"Instance {address!r} is not registered.") from None
