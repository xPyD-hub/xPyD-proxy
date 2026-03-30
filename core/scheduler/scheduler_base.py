# SPDX-License-Identifier: Apache-2.0
"""Abstract base class for scheduling policies."""

import itertools
import threading
from abc import ABC, abstractmethod
from typing import Optional


class SchedulingPolicy(ABC):
    """Base class for all scheduling policies.

    Subclasses must implement :meth:`schedule`.  Optionally override
    :meth:`schedule_completion` to track load after requests finish.
    """

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
        """Select the next instance to handle a request.

        Args:
            cycler: An ``itertools.cycle`` over the instance list.
            is_prompt: ``True`` for prefill scheduling, ``False`` for decode.
            request_len: Estimated prompt token length.
            max_tokens: Maximum tokens to generate.

        Returns:
            The selected ``host:port`` string, or ``None`` if no instance
            can handle the request.
        """
        raise NotImplementedError("Scheduling policy is not set.")

    def schedule_completion(
        self,
        prefill_instance: Optional[str] = None,
        decode_instance: Optional[str] = None,
        req_len: Optional[int] = None,
    ) -> None:
        """Called when a request finishes.  Override to track load."""
