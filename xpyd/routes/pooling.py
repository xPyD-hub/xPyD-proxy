# SPDX-License-Identifier: Apache-2.0
"""The /pooling endpoint."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from xpyd.interfaces import PostToInstance


def register(router: APIRouter, *, forward: PostToInstance) -> None:
    async def post_pooling(request: Request):
        return await forward(request, "/pooling", {"model": "", "messages": ""})

    router.post("/pooling", response_class=JSONResponse)(post_pooling)
    router.options("/pooling")(lambda: None)
