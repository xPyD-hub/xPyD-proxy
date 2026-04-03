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
from typing import TYPE_CHECKING, List, Optional, Set

import aiohttp

if TYPE_CHECKING:
    from xpyd.registry import InstanceRegistry

logger = logging.getLogger(__name__)


class DiscoveryTimeout(Exception):
    """Raised when discovery fails to find minimum nodes within timeout."""


class NodeDiscovery:
    """Background node health prober.

    On startup, begins probing all configured nodes. Tracks which are
    healthy so the proxy can decide when to start accepting requests.
    Also probes ``/v1/models`` to auto-detect model names served by
    each instance and updates the registry accordingly.
    """

    def __init__(
        self,
        prefill_instances: List[str],
        decode_instances: List[str],
        probe_interval: float = 10.0,
        wait_timeout: float = 600.0,
        registry: Optional["InstanceRegistry"] = None,
        dual_instances: Optional[List[str]] = None,
    ):
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        self.dual_instances = dual_instances or []
        self.probe_interval = probe_interval
        self.wait_timeout = wait_timeout
        self.registry = registry

        self.healthy_prefill: Set[str] = set()
        self.healthy_decode: Set[str] = set()
        self.healthy_dual: Set[str] = set()
        self._ready = asyncio.Event()
        self._task: asyncio.Task | None = None

    @property
    def is_ready(self) -> bool:
        """True when at least 1 prefill + 1 decode node are healthy, or at least 1 dual node is healthy."""
        has_pd = len(self.healthy_prefill) >= 1 and len(self.healthy_decode) >= 1
        has_dual = len(self.healthy_dual) >= 1
        return has_pd or has_dual

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
                        "Proxy ready: %d prefill, %d decode, %d dual nodes available",
                        len(self.healthy_prefill),
                        len(self.healthy_decode),
                        len(self.healthy_dual),
                    )

                elapsed = time.monotonic() - start_time
                if elapsed >= self.wait_timeout and not self.is_ready:
                    logger.error(
                        "Timeout waiting for backend nodes after %.0fs", elapsed
                    )
                    raise DiscoveryTimeout(
                        f"No minimum nodes (1P+1D or 1 dual) after {elapsed:.0f}s"
                    )

                await asyncio.sleep(self.probe_interval)

    async def _probe_all(self, session: aiohttp.ClientSession):
        """Probe all prefill, decode, and dual nodes once."""
        tasks = []
        for inst in self.prefill_instances:
            tasks.append(self._probe_node(session, inst, "prefill"))
        for inst in self.decode_instances:
            tasks.append(self._probe_node(session, inst, "decode"))
        for inst in self.dual_instances:
            tasks.append(self._probe_node(session, inst, "dual"))
        await asyncio.gather(*tasks)

    async def _probe_node(
        self, session: aiohttp.ClientSession, instance: str, role: str
    ):
        """Probe a single node's /health endpoint and /v1/models for model auto-detection."""
        url = f"http://{instance}/health"
        if role == "prefill":
            healthy_set = self.healthy_prefill
            all_instances = self.prefill_instances
        elif role == "decode":
            healthy_set = self.healthy_decode
            all_instances = self.decode_instances
        else:
            healthy_set = self.healthy_dual
            all_instances = self.dual_instances
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
                    # Probe /v1/models for model auto-detection
                    await self._probe_models(session, instance)
                else:
                    healthy_set.discard(instance)
        except Exception:
            healthy_set.discard(instance)

    async def _probe_models(
        self, session: aiohttp.ClientSession, instance: str
    ):
        """Probe /v1/models on a healthy instance to auto-detect model name.

        Only probes when the registry model for this instance is unknown
        (empty string). Logs at info level only when a model is newly
        detected; uses debug level for routine probes.
        """
        if self.registry is None:
            return
        # Skip probe if the model is already known for this instance
        try:
            info = self.registry.get_instance_info(instance)
            if info.model:
                return  # model already set — no need to probe
        except KeyError:
            return  # instance not in registry
        try:
            models_url = f"http://{instance}/v1/models"
            async with session.get(models_url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models_data = data.get("data", [])
                    if models_data:
                        model_name = models_data[0].get("id", "")
                        if model_name:
                            self.registry.update_model(instance, model_name)
                            logger.info(
                                "Auto-detected model %r on %s",
                                model_name,
                                instance,
                            )
                        else:
                            logger.debug(
                                "Empty model id from /v1/models on %s",
                                instance,
                            )
                    else:
                        logger.debug(
                            "Empty data list from /v1/models on %s",
                            instance,
                        )
                else:
                    logger.debug(
                        "/v1/models returned %d on %s", resp.status, instance,
                    )
        except Exception:
            logger.debug(
                "Failed to probe /v1/models on %s", instance
            )
