# SPDX-License-Identifier: Apache-2.0
"""The /v1/embeddings endpoint."""

from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from xpyd.proxy import Proxy


def register(router: APIRouter, server: "Proxy") -> None:
    async def post_embeddings(request: Request):
        return await server.post_to_instance(
            request, "/v1/embeddings", {"model": "", "input": ""}
        )

    router.post("/v1/embeddings", response_class=JSONResponse)(post_embeddings)
    router.options("/v1/embeddings")(lambda: None)
