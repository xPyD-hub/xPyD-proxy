# SPDX-License-Identifier: Apache-2.0
"""Cache-Aware Routing scheduling policy.

Routes requests with similar prompt prefixes to the same worker,
optimizing for prefix cache hits on inference backends.

Uses a consistent hash ring (160 virtual nodes per worker) so that
adding/removing a worker only remaps ~1/N of the key space.
Prompt prefix tokenization prefers a real tokenizer when available,
falling back to whitespace splitting for compatibility.
"""

import bisect
import hashlib
import itertools
import logging
from typing import Any, Optional

try:
    from .scheduler_base import SchedulingPolicy
except ImportError:
    from scheduler_base import SchedulingPolicy

logger = logging.getLogger(__name__)

DEFAULT_PREFIX_LENGTH = 256
VIRTUAL_NODES_PER_WORKER = 160


class ConsistentHashRing:
    """A consistent hash ring using virtual nodes.

    Each worker is mapped to *vnodes* points on a 2**128 ring (MD5 space).
    Lookup finds the nearest clockwise virtual node via binary search.
    """

    def __init__(self, vnodes: int = VIRTUAL_NODES_PER_WORKER):
        self._vnodes = vnodes
        # Sorted list of (hash_value, worker_addr)
        self._ring_keys: list[int] = []
        self._ring_workers: list[str] = []
        self._workers: set[str] = set()

    @property
    def workers(self) -> set[str]:
        return set(self._workers)

    @staticmethod
    def _hash(key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)  # noqa: S324

    def add_worker(self, addr: str) -> None:
        if addr in self._workers:
            return
        self._workers.add(addr)
        for i in range(self._vnodes):
            h = self._hash(f"{addr}#{i}")
            idx = bisect.bisect_left(self._ring_keys, h)
            self._ring_keys.insert(idx, h)
            self._ring_workers.insert(idx, addr)

    def remove_worker(self, addr: str) -> None:
        if addr not in self._workers:
            return
        self._workers.discard(addr)
        # Rebuild without removed worker (simpler than in-place removal)
        new_keys: list[int] = []
        new_workers: list[str] = []
        for k, w in zip(self._ring_keys, self._ring_workers):
            if w != addr:
                new_keys.append(k)
                new_workers.append(w)
        self._ring_keys = new_keys
        self._ring_workers = new_workers

    def get(self, key: int) -> Optional[str]:
        """Find the worker for *key* (an integer hash) on the ring."""
        if not self._ring_keys:
            return None
        idx = bisect.bisect_right(self._ring_keys, key)
        if idx == len(self._ring_keys):
            idx = 0  # wrap around
        return self._ring_workers[idx]

    def __len__(self) -> int:
        return len(self._workers)


class CacheAwarePolicy(SchedulingPolicy):
    """Route requests to workers based on prompt prefix hash.

    Hashes the first *prefix_length* tokens of the prompt and looks up
    the result on a consistent hash ring, so requests sharing the same
    prefix hit the same backend and benefit from KV-cache reuse.

    Parameters
    ----------
    workers:
        Initial list of worker addresses.
    prefix_length:
        Number of tokens to consider for prefix hashing.
        Defaults to 256.
    tokenizer:
        An optional tokenizer instance (e.g. ``AutoTokenizer``).  When
        provided, ``tokenizer.encode(prompt)`` is used for tokenization.
        Falls back to whitespace splitting when *tokenizer* is ``None``
        or when ``encode()`` raises.
    """

    def __init__(
        self,
        workers: Optional[list[str]] = None,
        prefix_length: int = DEFAULT_PREFIX_LENGTH,
        tokenizer: Any = None,
    ):
        super().__init__()
        self._prefix_length = prefix_length
        self._tokenizer = tokenizer
        self._ring = ConsistentHashRing(vnodes=VIRTUAL_NODES_PER_WORKER)
        for addr in (workers or []):
            self._ring.add_worker(addr)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, prompt: str) -> list:
        """Tokenize *prompt*, falling back to whitespace split."""
        if self._tokenizer is not None:
            try:
                return self._tokenizer.encode(prompt)
            except Exception:
                logger.debug("Tokenizer failed, falling back to whitespace split")
        return prompt.split()

    def _prefix_hash(self, prompt: str) -> int:
        """Hash the first *prefix_length* tokens of *prompt*."""
        tokens = self._tokenize(prompt)
        prefix_tokens = tokens[: self._prefix_length]
        # Use a stable string representation for hashing
        prefix_str = " ".join(str(t) for t in prefix_tokens)
        return int(
            hashlib.md5(prefix_str.encode()).hexdigest(), 16  # noqa: S324
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_worker(self, addr: str) -> None:
        """Add a worker to the pool."""
        with self.lock:
            self._ring.add_worker(addr)

    def remove_worker(self, addr: str) -> None:
        """Remove a worker from the pool."""
        with self.lock:
            self._ring.remove_worker(addr)

    def select(
        self,
        *,
        prompt: Optional[str] = None,
    ) -> Optional[str]:
        """Select a worker based on prompt prefix hash.

        Returns ``None`` when no workers are available or prompt is
        ``None``.
        """
        with self.lock:
            if len(self._ring) == 0:
                return None
            if prompt is None:
                prompt = ""
            h = self._prefix_hash(prompt)
            return self._ring.get(h)

    # ------------------------------------------------------------------
    # SchedulingPolicy interface
    # ------------------------------------------------------------------

    def schedule(
        self,
        cycler: itertools.cycle,
        is_prompt: Optional[bool] = None,
        request_len: Optional[int] = None,
        max_tokens: Optional[int] = None,
        *,
        prompt: Optional[str] = None,
    ) -> Optional[str]:
        """Schedule using prompt prefix for cache-aware routing.

        If a registry is attached, refreshes the worker list from
        available instances before selecting.
        """
        if self._registry is not None:
            available = self._registry.get_available_instances("decode")
            with self.lock:
                current = self._ring.workers
                for addr in available:
                    if addr not in current:
                        self._ring.add_worker(addr)
                for addr in current:
                    if addr not in available:
                        self._ring.remove_worker(addr)
        return self.select(prompt=prompt)
