# SPDX-License-Identifier: Apache-2.0
"""Round-robin scheduling policy."""

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
