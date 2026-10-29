# SPDX-License-Identifier: Apache-2.0
"""The /version endpoint."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from xpyd.interfaces import GetFromInstance


def register(router: APIRouter, *, fetch: GetFromInstance) -> None:
    async def get_version():
        return await fetch("/version")

    router.get("/version", response_class=JSONResponse)(get_version)
    router.options("/version")(lambda: None)
