# SPDX-License-Identifier: Apache-2.0
"""The /v1/embeddings endpoint."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from xpyd.interfaces import PostToInstance


def register(router: APIRouter, *, forward: PostToInstance) -> None:
    async def post_embeddings(request: Request):
        return await forward(request, "/v1/embeddings", {"model": "", "input": ""})

    router.post("/v1/embeddings", response_class=JSONResponse)(post_embeddings)
    router.options("/v1/embeddings")(lambda: None)
