# SPDX-License-Identifier: Apache-2.0
"""The /invocations endpoint."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from xpyd.interfaces import PostToInstance


def register(router: APIRouter, *, forward: PostToInstance) -> None:
    async def post_invocations(request: Request):
        return await forward(request, "/invocations", {"model": "", "prompt": ""})

    router.post("/invocations", response_class=JSONResponse)(post_invocations)
    router.options("/invocations")(lambda: None)
