# SPDX-License-Identifier: Apache-2.0
"""The /detokenize endpoint."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from xpyd.interfaces import PostToInstance


def register(router: APIRouter, *, forward: PostToInstance) -> None:
    async def post_detokenize(request: Request):
        return await forward(request, "/detokenize", {"model": "", "tokens": []})

    router.post("/detokenize", response_class=JSONResponse)(post_detokenize)
    router.options("/detokenize")(lambda: None)
