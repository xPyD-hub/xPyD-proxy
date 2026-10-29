# SPDX-License-Identifier: Apache-2.0
"""FastAPI application assembly and background service lifecycle."""

import asyncio
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from xpyd.config import ProxyConfig
from xpyd.discovery import NodeDiscovery
from xpyd.errors import PROXY_ERROR, error_response
from xpyd.health_monitor import HealthMonitor

if TYPE_CHECKING:
    from xpyd.proxy import Proxy


def create_app(
    config: ProxyConfig,
    proxy: "Proxy",
    health_monitor: HealthMonitor | None = None,
) -> FastAPI:
    """Assemble an app around an existing proxy without starting a listener."""
    discovery = NodeDiscovery(
        prefill_instances=proxy.prefill_instances,
        decode_instances=proxy.decode_instances,
        probe_interval=config.probe_interval_seconds,
        wait_timeout=config.wait_timeout_seconds,
        heartbeat_interval=config.heartbeat_interval_seconds,
        registry=proxy.registry,
        aggregated_instances=[
            address
            for instances in proxy.aggregated_instances.values()
            for address in instances
        ],
        on_model_discovered=lambda model: asyncio.to_thread(
            proxy.ensure_tokenizer, model
        ),
    )
    proxy.discovery = discovery
    proxy.health_monitor = health_monitor

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _check_readiness(request: Request, call_next):
        # Operational endpoints must remain accessible before backends are ready.
        path = request.url.path
        if path in (
            "/health",
            "/ping",
            "/status",
            "/metrics",
            "/instances/add",
            "/instances/remove",
        ) or path.startswith("/status/"):
            return await call_next(request)
        if not discovery.is_ready:
            return error_response("Waiting for backend nodes", PROXY_ERROR, 503)
        return await call_next(request)

    @app.on_event("startup")
    async def _start_discovery():
        if config.disaggregated_mode == "zmq":
            from xpyd.zmq_notifications import ZmqNotificationListener

            zmq_config = config.zmq
            assert zmq_config is not None
            listener = ZmqNotificationListener(
                zmq_config.host,
                zmq_config.port,
                zmq_config.notification_timeout_seconds,
            )
            await listener.start()
            proxy.zmq_notifications = listener
        await discovery.start()
        if health_monitor:
            await health_monitor.start()

    @app.on_event("shutdown")
    async def _stop_discovery():
        await discovery.stop()
        if proxy.zmq_notifications is not None:
            await proxy.zmq_notifications.stop()
        if health_monitor:
            await health_monitor.stop()

    app.include_router(proxy.router)
    return app
