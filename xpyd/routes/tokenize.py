# SPDX-License-Identifier: Apache-2.0
"""The /tokenize endpoint."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from xpyd.interfaces import PostToInstance


def register(router: APIRouter, *, forward: PostToInstance) -> None:
    async def post_tokenize(request: Request):
        return await forward(request, "/tokenize", {"model": "", "prompt": ""})

    router.post("/tokenize", response_class=JSONResponse)(post_tokenize)
    router.options("/tokenize")(lambda: None)
