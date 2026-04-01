# SPDX-License-Identifier: Apache-2.0
"""Round-robin scheduling policy."""

import itertools
import logging
from typing import Optional

try:
    from .scheduler_base import SchedulingPolicy
except ImportError:
    from scheduler_base import SchedulingPolicy

logger = logging.getLogger(__name__)


class RoundRobinSchedulingPolicy(SchedulingPolicy):
    """Cycle through instances in order, ignoring load."""

    def __init__(self, registry=None):
        super().__init__(registry=registry)
        logger.info("RoundRobinSchedulingPolicy initialized")

    def safe_next(self, cycler: itertools.cycle):
        with self.lock:
            return next(cycler)

    def schedule(
        self,
        cycler: itertools.cycle,
        is_prompt: Optional[bool] = None,
        request_len: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        if self._registry is not None:
            role = "prefill" if is_prompt else "decode"
            available = self._registry.get_available_instances(role)
            if not available:
                return None
            with self.lock:
                # Round-robin over available instances
                instance = next(cycler)
                # Try up to N times to find one that's available
                seen = 0
                total = len(available)
                while instance not in available:
                    instance = next(cycler)
                    seen += 1
                    if seen > total + len(available):
                        return None
                return instance
        return self.safe_next(cycler)
