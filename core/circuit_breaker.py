# SPDX-License-Identifier: Apache-2.0
"""Per-node circuit breaker with CLOSED / OPEN / HALF-OPEN state machine.

The circuit breaker prevents sending requests to a node that is
consistently failing and gradually recovers when the node comes back.

State transitions::

    CLOSED  ──failure_threshold reached──►  OPEN
    OPEN    ──timeout expires──────────►  HALF-OPEN
    HALF-OPEN ──success_threshold──────►  CLOSED
    HALF-OPEN ──probe fails────────────►  OPEN
"""

from __future__ import annotations

import enum
import time
from collections import deque
from typing import Callable, Optional


class CircuitState(str, enum.Enum):
    """Possible states for a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"


class CircuitBreaker:
    """Per-node circuit breaker.

    Parameters
    ----------
    failure_threshold:
        Number of failures within *window_duration_seconds* to trip the
        breaker (transition CLOSED → OPEN).
    success_threshold:
        Consecutive successes in HALF-OPEN state required to close the
        circuit again.
    timeout_duration_seconds:
        How long the circuit stays OPEN before moving to HALF-OPEN.
    window_duration_seconds:
        Sliding window length (seconds) for counting failures in CLOSED
        state.
    clock:
        Optional callable returning the current monotonic time.  Defaults
        to ``time.monotonic``.  Useful for deterministic testing.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_duration_seconds: float = 30,
        window_duration_seconds: float = 60,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_duration_seconds = timeout_duration_seconds
        self.window_duration_seconds = window_duration_seconds

        self._clock = clock or time.monotonic
        self._state = CircuitState.CLOSED
        self._failure_timestamps: deque[float] = deque()
        self._opened_at: float = 0.0
        self._half_open_successes: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Return the current circuit state.

        Automatically transitions OPEN → HALF-OPEN when the timeout has
        elapsed so that callers always see an up-to-date state.
        """
        if self._state == CircuitState.OPEN:
            if self._clock() - self._opened_at >= self.timeout_duration_seconds:
                self._state = CircuitState.HALF_OPEN
                self._half_open_successes = 0
        return self._state

    def allow_request(self) -> bool:
        """Return ``True`` if a request should be sent through this node.

        * CLOSED  → always allow
        * OPEN    → deny (caller should route elsewhere)
        * HALF-OPEN → allow (probe request)
        """
        current = self.state  # triggers timeout check
        if current == CircuitState.CLOSED:
            return True
        if current == CircuitState.OPEN:
            return False
        # HALF-OPEN: allow probe
        return True

    def record_failure(self) -> None:
        """Record a failed request."""
        now = self._clock()
        current = self.state

        if current == CircuitState.HALF_OPEN:
            # Probe failed → reopen
            self._open(now)
            return

        if current == CircuitState.CLOSED:
            self._failure_timestamps.append(now)
            self._purge_old_failures(now)
            if len(self._failure_timestamps) >= self.failure_threshold:
                self._open(now)

    def record_success(self) -> None:
        """Record a successful request."""
        current = self.state

        if current == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self.success_threshold:
                self._close()
            return

        if current == CircuitState.CLOSED:
            # A success in CLOSED state clears the failure window
            self._failure_timestamps.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open(self, now: float) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = now
        self._failure_timestamps.clear()
        self._half_open_successes = 0

    def _close(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_timestamps.clear()
        self._half_open_successes = 0

    def _purge_old_failures(self, now: float) -> None:
        cutoff = now - self.window_duration_seconds
        while self._failure_timestamps and self._failure_timestamps[0] < cutoff:
            self._failure_timestamps.popleft()
