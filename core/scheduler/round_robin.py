# SPDX-License-Identifier: Apache-2.0
"""Round-robin scheduling policy."""

import itertools
from typing import Optional

from core.scheduler.scheduler_base import SchedulingPolicy


class RoundRobinSchedulingPolicy(SchedulingPolicy):
    """Cycle through instances in order, ignoring load."""

    def __init__(self):
        print("RoundRobinSchedulingPolicy")
        super().__init__()

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
        return self.safe_next(cycler)
