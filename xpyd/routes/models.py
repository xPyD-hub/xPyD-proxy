# SPDX-License-Identifier: Apache-2.0
"""The /v1/models endpoint."""

from typing import TYPE_CHECKING

from fastapi import APIRouter
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from xpyd.proxy import Proxy

# Fixed epoch timestamp for model listing (avoids changing on every request).
_MODEL_CREATED_EPOCH = 0


def register(router: APIRouter, server: "Proxy") -> None:
    async def get_models():
        if server.registry is not None:
            models = server.registry.get_registered_models()
            data = [
                {
                    "id": model,
                    "object": "model",
                    "created": _MODEL_CREATED_EPOCH,
                    "owned_by": "system",
                }
                for model in models
            ]
            return JSONResponse(
                content={"object": "list", "data": data},
                status_code=200,
            )
        return await server.get_from_instance("/v1/models")

    router.get("/v1/models", response_class=JSONResponse)(get_models)
    router.options("/v1/models")(lambda: None)
