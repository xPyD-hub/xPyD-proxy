# SPDX-License-Identifier: Apache-2.0
"""Health, info, and metrics route handlers."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from xpyd.metrics import get_metrics


def register(router: APIRouter, server) -> None:
    """Register health/info routes on *router*."""

    async def get_health():
        return await server.get_from_instance("/health", is_full_instancelist=1)

    async def get_ping():
        return await server.get_from_instance("/ping", is_full_instancelist=1)

    async def get_models():
        return await server.get_from_instance("/v1/models")

    async def get_version():
        return await server.get_from_instance("/version")

    async def get_metrics_endpoint():
        return Response(
            content=get_metrics(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    router.get("/health", response_class=PlainTextResponse)(get_health)
    router.get("/ping", response_class=PlainTextResponse)(get_ping)
    router.post("/ping", response_class=PlainTextResponse)(get_ping)
    router.get("/v1/models", response_class=JSONResponse)(get_models)
    router.get("/version", response_class=JSONResponse)(get_version)
    router.get("/metrics")(get_metrics_endpoint)

    router.options("/health")(lambda: None)
    router.options("/ping")(lambda: None)
    router.options("/v1/models")(lambda: None)
    router.options("/version")(lambda: None)
