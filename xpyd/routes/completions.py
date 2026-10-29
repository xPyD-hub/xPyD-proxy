# SPDX-License-Identifier: Apache-2.0
"""The /v1/completions endpoint."""

from fastapi import APIRouter, Depends, Request

from xpyd.completion_handler import validate_json_content_type
from xpyd.interfaces import AsyncRequestHandler, RequestHandler


def register(
    router: APIRouter,
    *,
    execute: AsyncRequestHandler,
    custom_handler: RequestHandler | None = None,
) -> None:
    async def create_completion(raw_request: Request):
        return await execute(raw_request)

    router.post(
        "/v1/completions",
        dependencies=[Depends(validate_json_content_type)],
    )(custom_handler or create_completion)
    router.options("/v1/completions")(lambda: None)
