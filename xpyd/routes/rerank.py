# SPDX-License-Identifier: Apache-2.0
"""The /rerank, /v1/rerank and /v2/rerank endpoints."""

from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from xpyd.proxy import Proxy


def register(router: APIRouter, server: "Proxy") -> None:
    async def post_rerank(request: Request):
        return await server.post_to_instance(
            request, "/rerank", {"model": "", "query": "", "documents": ""}
        )

    async def post_rerankv1(request: Request):
        return await server.post_to_instance(
            request, "/v1/rerank", {"model": "", "query": "", "documents": ""}
        )

    async def post_rerankv2(request: Request):
        return await server.post_to_instance(
            request, "/v2/rerank", {"model": "", "query": "", "documents": ""}
        )

    router.post("/rerank", response_class=JSONResponse)(post_rerank)
    router.post("/v1/rerank", response_class=JSONResponse)(post_rerankv1)
    router.post("/v2/rerank", response_class=JSONResponse)(post_rerankv2)
    router.options("/rerank")(lambda: None)
    router.options("/v1/rerank")(lambda: None)
    router.options("/v2/rerank")(lambda: None)
