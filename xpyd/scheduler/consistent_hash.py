# SPDX-License-Identifier: Apache-2.0
"""Consistent Hash scheduling policy with virtual nodes."""

import hashlib
import logging
from bisect import bisect_left, bisect_right, insort
from typing import Optional, Sequence

from xpyd.scheduler.scheduler_base import Candidate, SchedulingContext, SchedulingPolicy

logger = logging.getLogger(__name__)

DEFAULT_VIRTUAL_NODES = 160


class ConsistentHashPolicy(SchedulingPolicy):
    """Route requests to workers using a consistent hash ring.

    Uses virtual nodes (default 160 per worker) for even distribution.
    Hash key priority: header (X-Session-ID) > user > client_ip > session_id.
    When workers are added or removed, only keys in the affected range
    are redistributed (minimal disruption).
    """

    def __init__(
        self,
        workers: Optional[list[str]] = None,
        virtual_nodes: int = DEFAULT_VIRTUAL_NODES,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._virtual_nodes = virtual_nodes
        # Sorted list of hash values; _ring_map maps hash → worker address.
        self._ring_keys: list[int] = []
        self._ring_map: dict[int, str] = {}
        self._workers: set[str] = set()
        if workers:
            for w in workers:
                self._add_worker_unlocked(w)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(key: str, index: int) -> int:
        """Compute a deterministic hash for a virtual node."""
        data = f"{key}#{index}".encode()
        return int(hashlib.sha256(data).hexdigest(), 16)

    def _add_worker_unlocked(self, addr: str) -> None:
        if addr in self._workers:
            return
        self._workers.add(addr)
        for i in range(self._virtual_nodes):
            h = self._hash(addr, i)
            if h in self._ring_map:
                logger.debug(
                    "Hash collision at vnode %s#%d (hash=%x), skipping",
                    addr,
                    i,
                    h,
                )
                continue
            self._ring_map[h] = addr
            insort(self._ring_keys, h)

    def _remove_worker_unlocked(self, addr: str) -> None:
        if addr not in self._workers:
            return
        self._workers.discard(addr)
        for i in range(self._virtual_nodes):
            h = self._hash(addr, i)
            if self._ring_map.get(h) == addr:
                del self._ring_map[h]
                # O(log n) removal via bisect instead of O(n) list.remove
                idx = bisect_left(self._ring_keys, h)
                if idx < len(self._ring_keys) and self._ring_keys[idx] == h:
                    del self._ring_keys[idx]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_instance_added(self, role, address, max_model_len):
        """Add a worker to the hash ring."""
        with self.lock:
            self._add_worker_unlocked(address)

    def on_instance_removed(self, role, address, index):
        """Remove a worker from the hash ring."""
        with self.lock:
            self._remove_worker_unlocked(address)

    def select_node(
        self, context: SchedulingContext, candidates: Sequence[Candidate]
    ) -> Optional[str]:
        """Walk clockwise to the first eligible worker on the ring."""
        addresses = {candidate.address for candidate in candidates}
        key = context.header or context.user or context.client_ip or context.session_id
        if key is None:
            key = "__default__"

        with self.lock:
            if not self._ring_keys or not candidates:
                return None
            h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
            start = bisect_right(self._ring_keys, h) % len(self._ring_keys)
            n = len(self._ring_keys)
            for i in range(n):
                idx = (start + i) % n
                worker = self._ring_map[self._ring_keys[idx]]
                if worker in addresses:
                    return worker
            return None
