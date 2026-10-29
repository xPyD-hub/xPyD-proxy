# SPDX-License-Identifier: Apache-2.0
"""The /pooling endpoint."""

from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from xpyd.proxy import Proxy


def register(router: APIRouter, server: "Proxy") -> None:
    async def post_pooling(request: Request):
        return await server.post_to_instance(
            request, "/pooling", {"model": "", "messages": ""}
        )

    router.post("/pooling", response_class=JSONResponse)(post_pooling)
    router.options("/pooling")(lambda: None)
