# SPDX-License-Identifier: Apache-2.0
"""The /invocations endpoint."""

from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from xpyd.proxy import Proxy


def register(router: APIRouter, server: "Proxy") -> None:
    async def post_invocations(request: Request):
        return await server.post_to_instance(
            request, "/invocations", {"model": "", "prompt": ""}
        )

    router.post("/invocations", response_class=JSONResponse)(post_invocations)
    router.options("/invocations")(lambda: None)
