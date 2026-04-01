# SPDX-License-Identifier: Apache-2.0
"""Power of Two Choices scheduling policy."""

import itertools
import logging
import random
from typing import Optional

try:
    from .scheduler_base import SchedulingPolicy
except ImportError:
    from scheduler_base import SchedulingPolicy

logger = logging.getLogger(__name__)


class PowerOfTwoPolicy(SchedulingPolicy):
    """Pick 2 random workers, forward to the one with fewer active requests.

    Simple yet effective load-aware scheduling that avoids the overhead of
    tracking all workers while still making informed decisions.

    YAML config::

        scheduling: power_of_two
    """

    def __init__(
        self,
        workers: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._workers: list[str] = list(workers) if workers else []
        self._load: dict[str, int] = {w: 0 for w in self._workers}
        self.last_pair: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Load tracking
    # ------------------------------------------------------------------

    def set_load(self, worker: str, load: int) -> None:
        """Set the current load (active request count) for a worker."""
        with self.lock:
            self._load[worker] = load

    def get_load(self, worker: str) -> int:
        """Return the current load for a worker."""
        with self.lock:
            return self._load.get(worker, 0)

    # ------------------------------------------------------------------
    # Worker management
    # ------------------------------------------------------------------

    def add_worker(self, addr: str) -> None:
        """Add a worker to the pool."""
        with self.lock:
            if addr not in self._load:
                self._workers.append(addr)
                self._load[addr] = 0

    def remove_worker(self, addr: str) -> None:
        """Remove a worker from the pool."""
        with self.lock:
            if addr in self._load:
                self._workers = [w for w in self._workers if w != addr]
                del self._load[addr]

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self) -> Optional[str]:
        """Select a worker using the power-of-two-choices algorithm.

        Returns ``None`` when no workers are available.
        """
        with self.lock:
            if not self._workers:
                return None
            if len(self._workers) == 1:
                self.last_pair = (self._workers[0],)
                return self._workers[0]

            pair = random.sample(self._workers, 2)
            self.last_pair = tuple(pair)

            # Pick the one with fewer active requests
            if self._load.get(pair[0], 0) <= self._load.get(pair[1], 0):
                return pair[0]
            return pair[1]

    # ------------------------------------------------------------------
    # SchedulingPolicy interface
    # ------------------------------------------------------------------

    def schedule(
        self,
        cycler: itertools.cycle,
        is_prompt: Optional[bool] = None,
        request_len: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """Schedule using power-of-two-choices.

        If a registry is attached, refreshes the worker list and load
        from available instances before selecting.
        """
        if self._registry is not None:
            available = self._registry.get_available("decode")
            with self.lock:
                self._workers = list(available)
                # Sync load from registry where possible
                for w in self._workers:
                    if w not in self._load:
                        self._load[w] = 0
                # Remove stale entries
                current = set(self._workers)
                for w in list(self._load):
                    if w not in current:
                        del self._load[w]

        return self.select()

    def schedule_completion(
        self,
        prefill_instance: Optional[str] = None,
        decode_instance: Optional[str] = None,
        req_len: Optional[int] = None,
    ) -> None:
        """Decrement load counter when a request completes."""
        addr = decode_instance or prefill_instance
        if addr:
            with self.lock:
                if addr in self._load and self._load[addr] > 0:
                    self._load[addr] -= 1
