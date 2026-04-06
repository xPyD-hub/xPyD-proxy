# SPDX-License-Identifier: Apache-2.0
"""Background health monitor for MicroPDProxy backend nodes.

Periodically probes each configured node's ``/health`` endpoint and
invokes callbacks to report status changes.  Designed to integrate
with an ``InstanceRegistry`` or any component that accepts
``on_healthy(addr)`` / ``on_unhealthy(addr)`` callbacks.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Async background health prober."""

    def __init__(
        self,
        nodes: List[str],
        *,
        interval_seconds: float = 10.0,
        timeout_seconds: float = 3.0,
        on_healthy: Optional[Callable[[str], None]] = None,
        on_unhealthy: Optional[Callable[[str], None]] = None,
    ):
        self.nodes = nodes
        self.interval = interval_seconds
        self.timeout = timeout_seconds
        self._on_healthy = on_healthy or (lambda addr: None)
        self._on_unhealthy = on_unhealthy or (lambda addr: None)
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the background probe loop."""
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Cancel the background probe task."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def check_once(self) -> None:
        """Run a single probe cycle (useful for testing)."""
        await self._probe_all()

    async def _loop(self) -> None:
        """Probe loop that runs until cancelled."""
        while True:
            await self._probe_all()
            await asyncio.sleep(self.interval)

    async def _probe_all(self) -> None:
        """Probe every node concurrently."""
        client_timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            tasks = [self._probe_node(session, node) for node in self.nodes]
            await asyncio.gather(*tasks)

    async def _probe_node(
        self, session: aiohttp.ClientSession, addr: str
    ) -> None:
        """Probe a single node."""
        url = f"http://{addr}/health"
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    self._on_healthy(addr)
                else:
                    self._on_unhealthy(addr)
        except Exception:
            self._on_unhealthy(addr)
