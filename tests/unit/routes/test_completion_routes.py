# SPDX-License-Identifier: Apache-2.0
"""Completion entry points share execution without coupling route modules."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from xpyd.routes import chat_completions, completions

from ._helpers import build_app


@pytest.fixture(
    params=[
        (completions, "/v1/completions", False, "custom_create_completion"),
        (
            chat_completions,
            "/v1/chat/completions",
            True,
            "custom_create_chat_completion",
        ),
    ],
    ids=["completions", "chat-completions"],
)
def endpoint(request):
    return request.param


def test_default_entry_delegates_to_shared_handler(endpoint):
    module, path, is_chat, _ = endpoint
    server = SimpleNamespace(
        custom_create_completion=None,
        custom_create_chat_completion=None,
    )
    handler = AsyncMock(return_value=JSONResponse({"shared": True}, status_code=202))
    with patch.object(module, "handle_completion", handler):
        client = TestClient(build_app(module.register, server))
        response = client.post(path, json={"model": "example"})

    assert response.status_code == 202
    assert response.json() == {"shared": True}
    handler.assert_awaited_once()
    assert handler.call_args.args[0] == path
    assert isinstance(handler.call_args.args[1], Request)
    assert handler.call_args.args[2] is server
    assert handler.call_args.kwargs == {"is_chat": is_chat}
    assert client.options(path).status_code == 200
    other_path = "/v1/completions" if is_chat else "/v1/chat/completions"
    assert client.post(other_path, json={}).status_code == 404


def test_custom_handler_keeps_content_type_validation(endpoint):
    module, path, _, custom_attribute = endpoint
    calls = []

    async def custom(request: Request):
        calls.append(await request.json())
        return JSONResponse({"custom": True})

    server = SimpleNamespace(
        custom_create_completion=None,
        custom_create_chat_completion=None,
    )
    setattr(server, custom_attribute, custom)
    client = TestClient(build_app(module.register, server))

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
