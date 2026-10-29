# SPDX-License-Identifier: Apache-2.0
"""Completion entry points share execution without coupling route modules."""

from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from xpyd.routes import chat_completions, completions


@pytest.fixture(
    params=[
        (completions, "/v1/completions", False),
        (
            chat_completions,
            "/v1/chat/completions",
            True,
        ),
    ],
    ids=["completions", "chat-completions"],
)
def endpoint(request):
    return request.param


def test_default_entry_delegates_to_shared_handler(endpoint):
    module, path, is_chat = endpoint
    handler = AsyncMock(return_value=JSONResponse({"shared": True}, status_code=202))
    router = APIRouter()
    module.register(router, execute=handler)
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    response = client.post(path, json={"model": "example"})

    assert response.status_code == 202
    assert response.json() == {"shared": True}
    handler.assert_awaited_once()
    assert len(handler.call_args.args) == 1
    assert isinstance(handler.call_args.args[0], Request)
    assert handler.call_args.args[0].url.path == path
    assert handler.call_args.kwargs == {}
    assert client.options(path).status_code == 200
    other_path = "/v1/completions" if is_chat else "/v1/chat/completions"
    assert client.post(other_path, json={}).status_code == 404


@pytest.mark.parametrize("synchronous", [False, True])
def test_custom_handler_keeps_content_type_validation(endpoint, synchronous):
    module, path, _ = endpoint
    calls = []

    async def custom(request: Request):
        calls.append(await request.json())
        return JSONResponse({"custom": True})

    def sync_custom(request: Request):
        calls.append({"model": "example"})
        return JSONResponse({"custom": True})

    execute = AsyncMock()
    router = APIRouter()
    module.register(
        router, execute=execute, custom_handler=sync_custom if synchronous else custom
    )
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    assert (
        client.post(
            path, content="{}", headers={"content-type": "text/plain"}
        ).status_code
        == 415
    )
    assert calls == []
    response = client.post(path, json={"model": "example"})
    assert response.status_code == 200
    assert response.json() == {"custom": True}
    assert calls == [{"model": "example"}]
    execute.assert_not_called()
