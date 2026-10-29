# SPDX-License-Identifier: Apache-2.0
"""The /detokenize endpoint."""

from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from xpyd.proxy import Proxy


def register(router: APIRouter, server: "Proxy") -> None:
    async def post_detokenize(request: Request):
        return await server.post_to_instance(
            request, "/detokenize", {"model": "", "tokens": []}
        )

    router.post("/detokenize", response_class=JSONResponse)(post_detokenize)
    router.options("/detokenize")(lambda: None)
