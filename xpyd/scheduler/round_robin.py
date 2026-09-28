# SPDX-License-Identifier: Apache-2.0
"""Round-robin scheduling policy."""

import itertools
import logging
from typing import Optional, Sequence

from xpyd.scheduler.scheduler_base import Candidate, SchedulingContext, SchedulingPolicy

logger = logging.getLogger(__name__)


class RoundRobinSchedulingPolicy(SchedulingPolicy):
    """Cycle through instances in order, ignoring load."""

    def __init__(self, registry=None):
        super().__init__(registry=registry)
        self._positions: dict[tuple[str, str], int] = {}
        logger.info("RoundRobinSchedulingPolicy initialized")

    @classmethod
    def from_config(
        cls,
        *,
        prefill_instances=(),
        decode_instances=(),
        workers=(),
        registry=None,
        tokenizer=None,
        **options,
    ):
        return cls(registry=registry, **options)

    def select_node(
        self, context: SchedulingContext, candidates: Sequence[Candidate]
    ) -> Optional[str]:
        if not candidates:
            return None
        with self.lock:
            key = (context.role, context.model)
            position = self._positions.get(key, 0) % len(candidates)
            self._positions[key] = position + 1
            return candidates[position].address

    def safe_next(self, cycler: itertools.cycle) -> str:
        """Thread-safe next() on cycler. Returns str since cycler wraps instance
        addresses."""
        with self.lock:
            return next(cycler)

    def schedule(
        self,
        cycler: itertools.cycle,
        is_prompt: Optional[bool] = None,
        request_len: Optional[int] = None,
        max_tokens: Optional[int] = None,
        model: str = "",
        **kwargs,
    ) -> Optional[str]:
        if self._registry is not None:
            role = "prefill" if is_prompt else "decode"
            available = set(self._registry.get_available_instances(role, model=model))
            if not available:
                return None
            with self.lock:
                # Advance cycler until we find an available instance or
                # complete a full cycle (all unique addresses seen once).
                seen_addrs: set[str] = set()
                while True:
                    instance = next(cycler)
                    if instance in available:
                        return instance
                    if instance in seen_addrs:
                        # Full cycle without finding available → give up
                        return None
                    seen_addrs.add(instance)
        return self.safe_next(cycler)
