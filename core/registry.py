"""Centralized Instance Registry for tracking node state.

Provides a single source of truth for all prefill/decode node metadata
including health status, circuit breaker state, and request counters.
Thread-safe for concurrent access from multiple async tasks.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class InstanceStatus(str, Enum):
    """Health status of a node."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitBreakerState(str, Enum):
    """Circuit breaker state machine states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class InstanceInfo:
    """Per-node state tracked by the registry."""

    address: str
    role: str  # "prefill" or "decode"
    status: InstanceStatus = InstanceStatus.UNKNOWN
    circuit_breaker_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    last_health_check: Optional[float] = None
    active_request_count: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class InstanceRegistry:
    """Centralized registry tracking every prefill/decode node's state.

    All access is protected by a reentrant lock for thread safety.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._nodes: Dict[str, InstanceInfo] = {}

    def add(self, role: str, address: str) -> None:
        """Register a node with the given role and address.

        Args:
            role: Node role, either "prefill" or "decode".
            address: Node address (e.g. "10.0.0.1:8200").

        Raises:
            ValueError: If role is not "prefill" or "decode".
            ValueError: If address is already registered.
        """
        if role not in ("prefill", "decode"):
            raise ValueError(f"Invalid role: {role!r}. Must be 'prefill' or 'decode'.")
        with self._lock:
            if address in self._nodes:
                raise ValueError(f"Node {address!r} is already registered.")
            self._nodes[address] = InstanceInfo(address=address, role=role)

    def remove(self, address: str) -> None:
        """Remove a node from the registry.

        Args:
            address: Node address to remove.

        Raises:
            KeyError: If address is not registered.
        """
        with self._lock:
            if address not in self._nodes:
                raise KeyError(f"Node {address!r} is not registered.")
            del self._nodes[address]

    def get_available_nodes(self, role: str) -> List[str]:
        """Return addresses of healthy nodes with closed circuit breakers.

        Args:
            role: Filter by role ("prefill" or "decode").

        Returns:
            List of node addresses that are healthy and have a closed
            circuit breaker.  HALF_OPEN nodes are reserved for the
            circuit-breaker probe mechanism and excluded here.
        """
        with self._lock:
            return [
                node.address
                for node in self._nodes.values()
                if node.role == role
                and node.status == InstanceStatus.HEALTHY
                and node.circuit_breaker_state == CircuitBreakerState.CLOSED
            ]

    def mark_healthy(self, address: str) -> None:
        """Mark a node as healthy (called by health monitor).

        Args:
            address: Node address.

        Raises:
            KeyError: If address is not registered.
        """
        with self._lock:
            node = self._get_node(address)
            node.status = InstanceStatus.HEALTHY
            node.last_health_check = time.monotonic()

    def mark_unhealthy(self, address: str) -> None:
        """Mark a node as unhealthy (called by health monitor).

        Args:
            address: Node address.

        Raises:
            KeyError: If address is not registered.
        """
        with self._lock:
            node = self._get_node(address)
            node.status = InstanceStatus.UNHEALTHY
            node.last_health_check = time.monotonic()

    def record_success(self, address: str) -> None:
        """Record a successful request to a node.

        Resets consecutive failure count and increments success count.

        Args:
            address: Node address.

        Raises:
            KeyError: If address is not registered.
        """
        with self._lock:
            node = self._get_node(address)
            node.consecutive_failures = 0
            node.consecutive_successes += 1

    def record_failure(self, address: str) -> None:
        """Record a failed request to a node.

        Resets consecutive success count and increments failure count.

        Args:
            address: Node address.

        Raises:
            KeyError: If address is not registered.
        """
        with self._lock:
            node = self._get_node(address)
            node.consecutive_successes = 0
            node.consecutive_failures += 1

    def get_node_info(self, address: str) -> InstanceInfo:
        """Return a snapshot of node info.

        Args:
            address: Node address.

        Returns:
            Copy of the InstanceInfo for the given address.

        Raises:
            KeyError: If address is not registered.
        """
        with self._lock:
            node = self._get_node(address)
            return InstanceInfo(
                address=node.address,
                role=node.role,
                status=node.status,
                circuit_breaker_state=node.circuit_breaker_state,
                last_health_check=node.last_health_check,
                active_request_count=node.active_request_count,
                consecutive_failures=node.consecutive_failures,
                consecutive_successes=node.consecutive_successes,
            )

    def get_all_nodes(self) -> List[InstanceInfo]:
        """Return a snapshot of all registered nodes.

        Returns:
            List of InstanceInfo copies.
        """
        with self._lock:
            return [self.get_node_info(addr) for addr in self._nodes]

    def set_circuit_breaker_state(
        self, address: str, state: CircuitBreakerState
    ) -> None:
        """Update the circuit breaker state for a node.

        Args:
            address: Node address.
            state: New circuit breaker state.

        Raises:
            KeyError: If address is not registered.
        """
        with self._lock:
            node = self._get_node(address)
            node.circuit_breaker_state = state

    def increment_active_requests(self, address: str) -> None:
        """Increment the active request count for a node.

        Args:
            address: Node address.

        Raises:
            KeyError: If address is not registered.
        """
        with self._lock:
            node = self._get_node(address)
            node.active_request_count += 1

    def decrement_active_requests(self, address: str) -> None:
        """Decrement the active request count for a node.

        Args:
            address: Node address.

        Raises:
            KeyError: If address is not registered.
        """
        with self._lock:
            node = self._get_node(address)
            node.active_request_count = max(0, node.active_request_count - 1)

    def _get_node(self, address: str) -> InstanceInfo:
        """Get node by address or raise KeyError. Must hold lock."""
        try:
            return self._nodes[address]
        except KeyError:
            raise KeyError(f"Node {address!r} is not registered.") from None
