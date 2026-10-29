# SPDX-License-Identifier: Apache-2.0
"""The /version endpoint."""

from typing import TYPE_CHECKING

from fastapi import APIRouter
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from xpyd.proxy import Proxy


def register(router: APIRouter, server: "Proxy") -> None:
    async def get_version():
        return await server.get_from_instance("/version")

    router.get("/version", response_class=JSONResponse)(get_version)
    router.options("/version")(lambda: None)
