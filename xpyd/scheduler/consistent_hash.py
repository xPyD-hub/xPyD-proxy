# SPDX-License-Identifier: Apache-2.0
"""Consistent Hash scheduling policy with virtual nodes."""

import hashlib
import itertools
import logging
from bisect import bisect_left, bisect_right, insort
from typing import Optional

from xpyd.scheduler.scheduler_base import SchedulingPolicy

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
            if self._ring_map.get(h) == addr:
                del self._ring_map[h]
                # O(log n) removal via bisect instead of O(n) list.remove
                idx = bisect_left(self._ring_keys, h)
                if idx < len(self._ring_keys) and self._ring_keys[idx] == h:
                    del self._ring_keys[idx]

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
            logger.warning(
                "No routing context (header/user/client_ip/session_id) — "
                "all such requests will hit the same worker"
            )

        with self.lock:
            if not self._ring_keys:
                return None
            h = int(hashlib.md5(key.encode()).hexdigest(), 16)  # noqa: S324
            idx = bisect_right(self._ring_keys, h) % len(self._ring_keys)
            return self._ring_map[self._ring_keys[idx]]

    # ------------------------------------------------------------------
    # SchedulingPolicy interface (for integration with existing router)
    # ------------------------------------------------------------------

    def select_from(
        self,
        candidates: set[str],
        *,
        header: Optional[str] = None,
        session_id: Optional[str] = None,
        user: Optional[str] = None,
        client_ip: Optional[str] = None,
    ) -> Optional[str]:
        """Select a worker from *candidates* using consistent hashing.

        Walks the ring clockwise from the hash point and returns the
        first worker that belongs to *candidates*.  Returns ``None``
        when no candidate is found.
        """
        key = header or user or client_ip or session_id
        if key is None:
            key = "__default__"

        with self.lock:
            if not self._ring_keys or not candidates:
                return None
            h = int(hashlib.md5(key.encode()).hexdigest(), 16)  # noqa: S324
            start = bisect_right(self._ring_keys, h) % len(self._ring_keys)
            n = len(self._ring_keys)
            for i in range(n):
                idx = (start + i) % n
                worker = self._ring_map[self._ring_keys[idx]]
                if worker in candidates:
                    return worker
            return None

    def schedule(
        self,
        cycler: itertools.cycle,
        is_prompt: Optional[bool] = None,
        request_len: Optional[int] = None,
        max_tokens: Optional[int] = None,
        model: str = "",
        *,
        header: Optional[str] = None,
        session_id: Optional[str] = None,
        user: Optional[str] = None,
        client_ip: Optional[str] = None,
        **kwargs,
    ) -> Optional[str]:
        """Schedule using request context for consistent hashing.

        Key priority: header (X-Session-ID) > user > client_ip > session_id.
        Falls back to ``"__default__"`` only when no context is provided.

        Both prefill and decode requests are routed through the hash
        ring.  When a registry is attached, the ring contains all
        workers and results are filtered to the appropriate role.
        """
        if self._registry is not None:
            role = "prefill" if is_prompt else "decode"
            candidates = set(self._registry.get_available_instances(role))
            if candidates:
                return self.select_from(
                    candidates,
                    header=header,
                    session_id=session_id,
                    user=user,
                    client_ip=client_ip,
                )
            # No candidates for this role in registry – fall back to cycler
            return next(cycler)
        return self.select(
            header=header,
            session_id=session_id,
            user=user,
            client_ip=client_ip,
        )
