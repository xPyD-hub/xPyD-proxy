# SPDX-License-Identifier: Apache-2.0
"""The /health and /ping endpoints."""

from typing import TYPE_CHECKING

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

if TYPE_CHECKING:
    from xpyd.proxy import Proxy


def register(router: APIRouter, server: "Proxy") -> None:

    async def get_health():
        return await server.get_from_instance("/health", is_full_instancelist=1)

    async def get_ping():
        return await server.get_from_instance("/ping", is_full_instancelist=1)

    router.get("/health", response_class=PlainTextResponse)(get_health)
    router.get("/ping", response_class=PlainTextResponse)(get_ping)
    router.post("/ping", response_class=PlainTextResponse)(get_ping)
    router.options("/health")(lambda: None)
    router.options("/ping")(lambda: None)
