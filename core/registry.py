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
from typing import Dict, List, Optional

try:
    from circuit_breaker import CircuitBreaker, CircuitBreakerState
except ImportError:
    from .circuit_breaker import CircuitBreaker, CircuitBreakerState


class InstanceStatus(str, Enum):
    """Health status of an instance."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class InstanceInfo:
    """Per-instance state tracked by the registry."""

    address: str
    role: str  # "prefill" or "decode"
    status: InstanceStatus = InstanceStatus.UNKNOWN
    last_health_check: Optional[float] = None
    active_request_count: int = 0
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    @property
    def circuit_breaker_state(self) -> CircuitBreakerState:
        """Current circuit breaker state, delegated to the CircuitBreaker."""
        return self.circuit_breaker.state


class InstanceRegistry:
    """Centralized registry tracking every prefill/decode instance's state.

    All access is protected by a reentrant lock for thread safety.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._instances: Dict[str, InstanceInfo] = {}

    def add(self, role: str, address: str) -> None:
        """Register an instance with the given role and address.

        Args:
            role: Instance role, either "prefill" or "decode".
            address: Instance address (e.g. "10.0.0.1:8200").

        Raises:
            ValueError: If role is not "prefill" or "decode".
            ValueError: If address is already registered.
        """
        if role not in ("prefill", "decode"):
            raise ValueError(f"Invalid role: {role!r}. Must be 'prefill' or 'decode'.")
        with self._lock:
            if address in self._instances:
                raise ValueError(f"Instance {address!r} is already registered.")
            self._instances[address] = InstanceInfo(
                address=address,
                role=role,
                circuit_breaker=CircuitBreaker(),
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

    def get_available_instances(self, role: str) -> List[str]:
        """Return addresses of healthy instances with closed circuit breakers.

        Args:
            role: Filter by role ("prefill" or "decode").

        Returns:
            List of instance addresses that are healthy and have a closed
            circuit breaker.  HALF_OPEN instances are reserved for the
            circuit-breaker probe mechanism and excluded here.
        """
        with self._lock:
            return [
                instance.address
                for instance in self._instances.values()
                if instance.role == role
                and instance.status == InstanceStatus.HEALTHY
                and instance.circuit_breaker.state == CircuitBreakerState.CLOSED
            ]

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
                status=instance.status,
                last_health_check=instance.last_health_check,
                active_request_count=instance.active_request_count,
                circuit_breaker=instance.circuit_breaker,
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

    def _get_instance(self, address: str) -> InstanceInfo:
        """Get instance by address or raise KeyError. Must hold lock."""
        try:
            return self._instances[address]
        except KeyError:
            raise KeyError(f"Instance {address!r} is not registered.") from None
