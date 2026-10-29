# SPDX-License-Identifier: Apache-2.0
"""The /tokenize endpoint."""

from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from xpyd.proxy import Proxy


def register(router: APIRouter, server: "Proxy") -> None:
    async def post_tokenize(request: Request):
        return await server.post_to_instance(
            request, "/tokenize", {"model": "", "prompt": ""}
        )

    router.post("/tokenize", response_class=JSONResponse)(post_tokenize)
    router.options("/tokenize")(lambda: None)
