# SPDX-License-Identifier: Apache-2.0
"""Power of Two Choices scheduling policy."""

import random
from typing import Optional, Sequence

from xpyd.scheduler.scheduler_base import Candidate, SchedulingContext, SchedulingPolicy


class PowerOfTwoPolicy(SchedulingPolicy):
    """Choose the less-loaded of two eligible candidates."""

    def __init__(self, workers=None, registry=None):
        super().__init__(registry=registry)
        self._last_pair: tuple[str, ...] = ()

    @property
    def last_pair(self) -> tuple[str, ...]:
        return self._last_pair

    def select_node(
        self, context: SchedulingContext, candidates: Sequence[Candidate]
    ) -> Optional[str]:
        with self.lock:
            ordered = sorted(candidates, key=lambda candidate: candidate.address)
            if not ordered:
                self._last_pair = ()
                return None
            addresses = [candidate.address for candidate in ordered]
            pair = random.sample(addresses, 2) if len(ordered) > 1 else addresses
            self._last_pair = tuple(pair)
            loads = {
                candidate.address: candidate.active_requests for candidate in ordered
            }
            return min(pair, key=loads.__getitem__)
