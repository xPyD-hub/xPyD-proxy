# SPDX-License-Identifier: Apache-2.0
"""Startup node discovery for MicroPDProxy.

Probes configured prefill/decode nodes in the background and tracks
which ones are healthy. The proxy returns 503 until at least one
prefill and one decode node are ready.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Set

import aiohttp

logger = logging.getLogger(__name__)


class DiscoveryTimeout(Exception):
    """Raised when discovery fails to find minimum nodes within timeout."""


class NodeDiscovery:
    """Background node health prober.

    On startup, begins probing all configured nodes. Tracks which are
    healthy so the proxy can decide when to start accepting requests.
    """

    def __init__(
        self,
        prefill_instances: List[str],
        decode_instances: List[str],
        probe_interval: float = 10.0,
        wait_timeout: float = 600.0,
    ):
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        self.probe_interval = probe_interval
        self.wait_timeout = wait_timeout

        self.healthy_prefill: Set[str] = set()
        self.healthy_decode: Set[str] = set()
        self._ready = asyncio.Event()
        self._task: asyncio.Task | None = None

    @property
    def is_ready(self) -> bool:
        """True when at least 1 prefill + 1 decode node are healthy."""
        return len(self.healthy_prefill) >= 1 and len(self.healthy_decode) >= 1

    async def start(self):
        """Start the background probe loop."""
        self._task = asyncio.create_task(self._probe_loop())
        self._task.add_done_callback(self._on_probe_done)

    @staticmethod
    def _on_probe_done(task: asyncio.Task) -> None:
        """Stop the event loop if the probe loop ended with a timeout."""
        if task.cancelled():
            return
        exc = task.exception()
        if isinstance(exc, DiscoveryTimeout):
            logger.critical("Discovery failed: %s", exc)
            loop = asyncio.get_event_loop()
            loop.stop()

    async def stop(self):
        """Cancel the background probe task."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, DiscoveryTimeout):
                pass

    async def wait_until_ready(self) -> bool:
        """Block until ready or timeout. Returns True if ready."""
        try:
            await asyncio.wait_for(
                self._ready.wait(), timeout=self.wait_timeout
            )
            return True
        except asyncio.TimeoutError:
            return False

    async def _probe_loop(self):
        """Periodically probe all nodes."""
        start_time = time.monotonic()
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        ) as session:
            while True:
                await self._probe_all(session)

                if self.is_ready and not self._ready.is_set():
                    self._ready.set()
                    logger.info(
                        "Proxy ready: %d prefill, %d decode nodes available",
                        len(self.healthy_prefill),
                        len(self.healthy_decode),
                    )

                elapsed = time.monotonic() - start_time
                if elapsed >= self.wait_timeout and not self.is_ready:
                    logger.error(
                        "Timeout waiting for backend nodes after %.0fs", elapsed
                    )
                    raise DiscoveryTimeout(
                        f"No minimum nodes (1P+1D) after {elapsed:.0f}s"
                    )

                await asyncio.sleep(self.probe_interval)

    async def _probe_all(self, session: aiohttp.ClientSession):
        """Probe all prefill and decode nodes once."""
        tasks = []
        for inst in self.prefill_instances:
            tasks.append(self._probe_node(session, inst, "prefill"))
        for inst in self.decode_instances:
            tasks.append(self._probe_node(session, inst, "decode"))
        await asyncio.gather(*tasks)

    async def _probe_node(
        self, session: aiohttp.ClientSession, instance: str, role: str
    ):
        """Probe a single node's /health endpoint."""
        url = f"http://{instance}/health"
        healthy_set = (
            self.healthy_prefill if role == "prefill" else self.healthy_decode
        )
        all_instances = (
            self.prefill_instances
            if role == "prefill"
            else self.decode_instances
        )
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    if instance not in healthy_set:
                        healthy_set.add(instance)
                        logger.info(
                            "[%d/%d %s nodes ready] %s",
                            len(healthy_set),
                            len(all_instances),
                            role,
                            instance,
                        )
                else:
                    healthy_set.discard(instance)
        except Exception:
            healthy_set.discard(instance)
