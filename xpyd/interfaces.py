# SPDX-License-Identifier: Apache-2.0
"""Narrow callable contracts used by HTTP endpoints and request execution."""

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol

from fastapi import Request
from fastapi.responses import JSONResponse, Response

AsyncRequestHandler = Callable[[Request], Awaitable[Response]]
RequestHandler = Callable[[Request], Response | Awaitable[Response]]
PostToInstance = Callable[[Request, str, dict[str, Any]], Awaitable[JSONResponse]]
CountTokens = Callable[[Any, str], int]
TokenizeOnBackend = Callable[[dict[str, Any], bool], Awaitable[list[int]]]


class GetFromInstance(Protocol):
    async def __call__(self, path: str, is_full_instancelist: int = 0) -> JSONResponse:
        pass


class ForwardRequest(Protocol):
    def __call__(
        self,
        url: str,
        data: dict[str, Any],
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[bytes]:
        pass


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]:
        pass

    def apply_chat_template(self, messages: list[dict], **kwargs: Any) -> Any:
        pass


class Notifications(Protocol):
    async def register(self, request_id: str) -> None:
        pass

    async def wait(self, request_id: str, expected: int) -> None:
        pass

    async def discard(self, request_id: str) -> None:
        pass
