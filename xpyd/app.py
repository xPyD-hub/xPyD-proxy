# SPDX-License-Identifier: Apache-2.0
"""FastAPI application assembly and background service lifecycle."""

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from xpyd.discovery import NodeDiscovery
from xpyd.errors import PROXY_ERROR, error_response
from xpyd.health_monitor import HealthMonitor
from xpyd.zmq_notifications import ZmqNotificationListener


def create_app(
    router: APIRouter,
    *,
    discovery: NodeDiscovery,
    health_monitor: HealthMonitor | None = None,
    zmq_notifications: ZmqNotificationListener | None = None,
) -> FastAPI:
    """Mount a router and manage explicitly supplied background services."""
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
        if zmq_notifications is not None:
            await zmq_notifications.start()
        await discovery.start()
        if health_monitor:
            await health_monitor.start()

    @app.on_event("shutdown")
    async def _stop_discovery():
        await discovery.stop()
        if zmq_notifications is not None:
            await zmq_notifications.stop()
        if health_monitor:
            await health_monitor.stop()

    app.include_router(router)
    return app
