# SPDX-License-Identifier: Apache-2.0
"""The /v1/chat/completions endpoint."""

from fastapi import APIRouter, Depends, Request

from xpyd.completion_handler import validate_json_content_type
from xpyd.interfaces import AsyncRequestHandler, RequestHandler


def register(
    router: APIRouter,
    *,
    execute: AsyncRequestHandler,
    custom_handler: RequestHandler | None = None,
) -> None:
    async def create_chat_completion(raw_request: Request):
        return await execute(raw_request)

    router.post(
        "/v1/chat/completions",
        dependencies=[Depends(validate_json_content_type)],
    )(custom_handler or create_chat_completion)
    router.options("/v1/chat/completions")(lambda: None)
