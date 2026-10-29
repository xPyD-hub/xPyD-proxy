# SPDX-License-Identifier: Apache-2.0
"""The /score and /v1/score endpoints."""

from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from xpyd.proxy import Proxy


def register(router: APIRouter, server: "Proxy") -> None:
    async def post_score(request: Request):
        return await server.post_to_instance(
            request,
            "/score",
            {"model": "", "text_1": "", "text_2": "", "predictions": ""},
        )

    async def post_scorev1(request: Request):
        return await server.post_to_instance(
            request,
            "/v1/score",
            {"model": "", "text_1": "", "text_2": "", "predictions": ""},
        )

    router.post("/score", response_class=JSONResponse)(post_score)
    router.post("/v1/score", response_class=JSONResponse)(post_scorev1)
    router.options("/score")(lambda: None)
    router.options("/v1/score")(lambda: None)
