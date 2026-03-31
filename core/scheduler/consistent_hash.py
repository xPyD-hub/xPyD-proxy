# SPDX-License-Identifier: Apache-2.0
"""Consistent Hash scheduling policy with virtual nodes."""

import hashlib
import itertools
import logging
from bisect import bisect_right, insort
from typing import Optional

try:
    from .scheduler_base import SchedulingPolicy
except ImportError:
    from scheduler_base import SchedulingPolicy

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
        header_name: str = "X-Session-ID",
    ):
        super().__init__()
        self._virtual_nodes = virtual_nodes
        self._header_name = header_name
        # Sorted list of (hash_value, worker_addr)
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
        return int(hashlib.md5(data).hexdigest(), 16)  # noqa: S324

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
            self._ring_map.pop(h, None)
            try:
                self._ring_keys.remove(h)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_worker(self, addr: str) -> None:
        """Add a worker to the hash ring."""
        with self.lock:
            self._add_worker_unlocked(addr)

    def remove_worker(self, addr: str) -> None:
        """Remove a worker from the hash ring."""
        with self.lock:
            self._remove_worker_unlocked(addr)

    def select(
        self,
        *,
        header: Optional[str] = None,
        session_id: Optional[str] = None,
        user: Optional[str] = None,
        client_ip: Optional[str] = None,
    ) -> Optional[str]:
        """Select a worker for the given request context.

        Key priority: header > user > client_ip > session_id.
        Returns ``None`` when the ring is empty.
        """
        key = header or user or client_ip or session_id
        if key is None:
            key = "__default__"

        with self.lock:
            if not self._ring_keys:
                return None
            h = int(hashlib.md5(key.encode()).hexdigest(), 16)  # noqa: S324
            idx = bisect_right(self._ring_keys, h) % len(self._ring_keys)
            return self._ring_map[self._ring_keys[idx]]

    # ------------------------------------------------------------------
    # SchedulingPolicy interface (for integration with existing router)
    # ------------------------------------------------------------------

    def schedule(
        self,
        cycler: itertools.cycle,
        is_prompt: Optional[bool] = None,
        request_len: Optional[int] = None,
        max_tokens: Optional[int] = None,
        *,
        header: Optional[str] = None,
        session_id: Optional[str] = None,
        user: Optional[str] = None,
        client_ip: Optional[str] = None,
    ) -> Optional[str]:
        """Schedule using request context for consistent hashing.

        Key priority: header (X-Session-ID) > user > client_ip > session_id.
        Falls back to ``"__default__"`` only when no context is provided.
        """
        return self.select(
            header=header,
            session_id=session_id,
            user=user,
            client_ip=client_ip,
        )
