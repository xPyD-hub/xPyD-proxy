# SPDX-License-Identifier: Apache-2.0
"""Abstract base class for scheduling policies."""

import itertools
import threading
from abc import ABC, abstractmethod
from typing import Optional


class SchedulingPolicy(ABC):
    """Base class for all scheduling policies."""

    def __init__(self):
        self.lock = threading.Lock()

    @abstractmethod
    def schedule(
        self,
        cycler: itertools.cycle,
        is_prompt: Optional[bool] = None,
        request_len: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        raise NotImplementedError("Scheduling policy is not set.")

    def schedule_completion(
        self,
        prefill_instance: Optional[str] = None,
        decode_instance: Optional[str] = None,
        req_len: Optional[int] = None,
    ) -> None:
        """Called when a request finishes. Override to track load."""
