# SPDX-License-Identifier: Apache-2.0
"""The /metrics endpoint."""

from typing import TYPE_CHECKING

from fastapi import APIRouter
from fastapi.responses import Response

from xpyd.metrics import get_metrics

if TYPE_CHECKING:
    from xpyd.proxy import Proxy


def register(router: APIRouter, server: "Proxy") -> None:
    async def get_metrics_endpoint():
        return Response(
            content=get_metrics(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    router.get("/metrics")(get_metrics_endpoint)
