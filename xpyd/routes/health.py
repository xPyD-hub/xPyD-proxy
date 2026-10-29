# SPDX-License-Identifier: Apache-2.0
"""The /health and /ping endpoints."""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from xpyd.interfaces import GetFromInstance


def register(router: APIRouter, *, fetch: GetFromInstance) -> None:

    async def get_health():
        return await fetch("/health", is_full_instancelist=1)

    async def get_ping():
        return await fetch("/ping", is_full_instancelist=1)

    router.get("/health", response_class=PlainTextResponse)(get_health)
    router.get("/ping", response_class=PlainTextResponse)(get_ping)
    router.post("/ping", response_class=PlainTextResponse)(get_ping)
    router.options("/health")(lambda: None)
    router.options("/ping")(lambda: None)
