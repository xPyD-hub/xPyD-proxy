# SPDX-License-Identifier: Apache-2.0
"""Tests for the passthrough (forward-to-instance) routes."""

# Standard
from typing import Any

# Third Party
import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

# First Party
from xpyd.routes import (
    detokenize,
    embeddings,
    invocations,
    pooling,
    rerank,
    score,
    tokenize,
)

from ._helpers import build_app


def register(router, server):
    for module in (
        tokenize,
        detokenize,
        embeddings,
        pooling,
        score,
        rerank,
        invocations,
    ):
        module.register(router, forward=server.post_to_instance)


# (route, forwarded path, expected json template, a valid request body)
ENDPOINTS = [
    (
        "/tokenize",
        "/tokenize",
        {"model": "", "prompt": ""},
        {"model": "opt", "prompt": "hello"},
    ),
    (
        "/detokenize",
        "/detokenize",
        {"model": "", "tokens": []},
        {"model": "opt", "tokens": [1, 2]},
    ),
    (
        "/v1/embeddings",
        "/v1/embeddings",
        {"model": "", "input": ""},
        {"model": "opt", "input": "hello"},
    ),
    (
        "/pooling",
        "/pooling",
        {"model": "", "messages": ""},
        {"model": "opt", "messages": "hello"},
    ),
    (
        "/score",
        "/score",
        {"model": "", "text_1": "", "text_2": "", "predictions": ""},
        {"model": "opt", "text_1": "a", "text_2": "b", "predictions": ""},
    ),
    (
        "/v1/score",
        "/v1/score",
        {"model": "", "text_1": "", "text_2": "", "predictions": ""},
        {"model": "opt", "text_1": "a", "text_2": "b", "predictions": ""},
    ),
    (
        "/rerank",
        "/rerank",
        {"model": "", "query": "", "documents": ""},
        {"model": "opt", "query": "q", "documents": ["d"]},
    ),
    (
        "/v1/rerank",
        "/v1/rerank",
        {"model": "", "query": "", "documents": ""},
        {"model": "opt", "query": "q", "documents": ["d"]},
    ),
    (
        "/v2/rerank",
        "/v2/rerank",
        {"model": "", "query": "", "documents": ""},
        {"model": "opt", "query": "q", "documents": ["d"]},
    ),
    (
        "/invocations",
        "/invocations",
        {"model": "", "prompt": ""},
        {"model": "opt", "prompt": "hello"},
    ),
]

ROUTE_IDS = [route for route, _, _, _ in ENDPOINTS]


class _RecordingServer:
    """Captures the arguments the routes hand to ``post_to_instance``."""

    def __init__(self, response: Any = None) -> None:
        self.calls: list[tuple[str, dict, dict]] = []
        self._response = response

    async def post_to_instance(self, request, path, json_template):
        body = await request.json()
        self.calls.append((path, json_template, body))
        if self._response is not None:
            return self._response
        return JSONResponse({"ok": True})


@pytest.mark.parametrize(
    "route,forward_path,template,body",
    ENDPOINTS,
    ids=ROUTE_IDS,
)
def test_route_forwards_with_expected_template(
    route: str,
    forward_path: str,
    template: dict,
    body: dict,
) -> None:
    """Each passthrough route forwards to its backend path and template."""
    server = _RecordingServer()
    client = TestClient(build_app(register, server))

    response = client.post(route, json=body)

    assert response.status_code == 200
    assert server.calls == [(forward_path, template, body)]


@pytest.mark.parametrize("route", ROUTE_IDS)
def test_route_options_is_allowed(route: str) -> None:
    """CORS preflight is answered for every passthrough route."""
    server = _RecordingServer()
    client = TestClient(build_app(register, server))

    assert client.options(route).status_code == 200
    assert server.calls == []


@pytest.mark.parametrize("status_code", [200, 400, 503])
def test_backend_status_and_body_are_passed_through(status_code: int) -> None:
    """The backend status code and payload reach the caller unchanged."""
    payload = {"detail": "from-backend", "status": status_code}
    server = _RecordingServer(JSONResponse(payload, status_code=status_code))
    client = TestClient(build_app(register, server))

    response = client.post("/tokenize", json={"model": "opt", "prompt": "hi"})

    assert response.status_code == status_code
    assert response.json() == payload
