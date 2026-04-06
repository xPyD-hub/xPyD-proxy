# SPDX-License-Identifier: Apache-2.0
"""Resilience module with retry, exponential backoff and jitter.

Provides :class:`ResilienceConfig` (Pydantic model for YAML configuration)
and :class:`ResilienceHandler` which executes a request function with automatic
retry on transient failures, selecting a different backend instance for
each attempt.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Callable, List, Optional

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class ResilienceConfig(BaseModel):
    """Configuration for resilience (retry) with exponential backoff + jitter."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    max_retries: int = 2
    initial_backoff_ms: int = 100
    max_backoff_ms: int = 10000
    backoff_multiplier: float = 2.0
    jitter_factor: float = 0.1
    retryable_status_codes: List[int] = [408, 429, 500, 502, 503, 504]


def compute_backoff(
    attempt: int,
    initial_ms: float,
    multiplier: float,
    max_ms: float,
    jitter_factor: float,
) -> float:
    """Compute backoff delay in milliseconds for the given attempt.

    Formula::

        delay = min(initial_ms * multiplier^attempt, max_ms)
        actual = delay * (1 + random(-jitter_factor, +jitter_factor))
    """
    delay = min(initial_ms * (multiplier ** attempt), max_ms)
    jitter = random.uniform(-jitter_factor, jitter_factor)  # noqa: S311
    return delay * (1.0 + jitter)


class ResilienceHandler:
    """Execute requests with resilience (retry, backoff, and instance re-selection).

    Uses dependency injection (callback functions) rather than importing
    the registry directly, keeping the module decoupled.
    """

    def __init__(self, config: ResilienceConfig) -> None:
        self.config = config

    def _should_retry(self, status_code: int, is_streaming: bool) -> bool:
        """Determine whether a failed request should be retried."""
        if is_streaming:
            return False
        return status_code in self.config.retryable_status_codes

    async def execute(
        self,
        request_fn: Callable[..., Any],
        select_instance_fn: Callable[..., Any],
        on_success: Optional[Callable[..., Any]] = None,
        on_failure: Optional[Callable[..., Any]] = None,
    ) -> Any:
        """Execute *request_fn* with retry logic.

        Parameters
        ----------
        request_fn:
            ``async (instance) -> response``.  The response object must
            expose a ``status_code`` attribute (int).
        select_instance_fn:
            ``(excluded: list[str] | None) -> str`` — picks an instance
            address, optionally excluding already-tried ones.
        on_success:
            Optional callback ``(instance, response) -> None``.
        on_failure:
            Optional callback ``(instance, response, attempt) -> None``.
        """
        # When retry is disabled, perform exactly one attempt.
        if not self.config.enabled:
            instance = select_instance_fn(excluded=None)
            response = await request_fn(instance)
            status_code = response.status_code
            if 200 <= status_code < 300:
                if on_success is not None:
                    on_success(instance, response)
            else:
                if on_failure is not None:
                    on_failure(instance, response, 0)
            return response

        tried: List[str] = []
        next_instance: Optional[str] = None

        for attempt in range(self.config.max_retries + 1):
            if next_instance is not None:
                instance = next_instance
                next_instance = None
            else:
                instance = select_instance_fn(
                    excluded=tried if tried else None
                )
            tried.append(instance)

            response = await request_fn(instance)

            status_code = response.status_code

            if 200 <= status_code < 300:
                if on_success is not None:
                    on_success(instance, response)
                return response

            is_streaming = getattr(response, "is_streaming", False)

            if attempt < self.config.max_retries and self._should_retry(
                status_code, is_streaming
            ):
                backoff_ms = compute_backoff(
                    attempt=attempt,
                    initial_ms=self.config.initial_backoff_ms,
                    multiplier=self.config.backoff_multiplier,
                    max_ms=self.config.max_backoff_ms,
                    jitter_factor=self.config.jitter_factor,
                )
                try:
                    next_instance = select_instance_fn(excluded=tried)
                except Exception:
                    logger.warning(
                        "[RETRY] No alternative instance available after %s "
                        "returned %d, returning last response",
                        instance,
                        status_code,
                    )
                    if on_failure is not None:
                        on_failure(instance, response, attempt)
                    return response
                if next_instance is None:
                    logger.warning(
                        "[RETRY] No alternative instance available after %s "
                        "returned %d, returning last response",
                        instance,
                        status_code,
                    )
                    if on_failure is not None:
                        on_failure(instance, response, attempt)
                    return response
                logger.warning(
                    "[RETRY] %s returned %d, attempt %d/%d, backoff %.0fms, "
                    "retrying on %s",
                    instance,
                    status_code,
                    attempt + 1,
                    self.config.max_retries,
                    backoff_ms,
                    next_instance,
                )
                if on_failure is not None:
                    on_failure(instance, response, attempt)
                await asyncio.sleep(backoff_ms / 1000.0)
                continue

            if on_failure is not None:
                on_failure(instance, response, attempt)
            return response

        return response  # pragma: no cover
