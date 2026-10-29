# SPDX-License-Identifier: Apache-2.0
"""The /v1/models endpoint."""

from collections.abc import Callable

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from xpyd.interfaces import GetFromInstance

# Fixed epoch timestamp for model listing (avoids changing on every request).
_MODEL_CREATED_EPOCH = 0


def register(
    router: APIRouter,
    *,
    fetch: GetFromInstance,
    get_registered_models: Callable[[], list[str]] | None = None,
) -> None:
    async def get_models():
        if get_registered_models is not None:
            models = get_registered_models()
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
        return await fetch("/v1/models")

    router.get("/v1/models", response_class=JSONResponse)(get_models)
    router.options("/v1/models")(lambda: None)
