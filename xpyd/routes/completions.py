# SPDX-License-Identifier: Apache-2.0
"""The /v1/completions endpoint."""

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Request

from xpyd.completion_handler import handle_completion, validate_json_content_type

if TYPE_CHECKING:
    from xpyd.proxy import Proxy


def register(router: APIRouter, server: "Proxy") -> None:
    async def create_completion(raw_request: Request):
        return await handle_completion(
            "/v1/completions", raw_request, server, is_chat=False
        )

    router.post(
        "/v1/completions",
        dependencies=[Depends(validate_json_content_type)],
    )(server.custom_create_completion or create_completion)
    router.options("/v1/completions")(lambda: None)
