"""Tests for the dummy prefill node."""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from dummy_nodes.prefill_node import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


CHAT_PAYLOAD = {
    "model": "dummy",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 5,
    "stream": False,
}


@pytest.mark.anyio
async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["node_type"] == "prefill"


@pytest.mark.anyio
async def test_non_streaming(client: AsyncClient):
    resp = await client.post("/v1/chat/completions", json=CHAT_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()

    # Structure checks
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert len(data["choices"][0]["message"]["content"]) > 0

    # Usage
    assert data["usage"]["completion_tokens"] == 5
    assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"] + 5


@pytest.mark.anyio
async def test_streaming(client: AsyncClient):
    payload = {**CHAT_PAYLOAD, "stream": True}
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = resp.text.strip().split("\n")
    data_lines = [l for l in lines if l.startswith("data: ")]

    # Should have: 1 role chunk + 5 content chunks + 1 finish chunk + 1 [DONE]
    assert len(data_lines) == 8  # 1 + 5 + 1 + 1

    # Last data line should be [DONE]
    assert data_lines[-1] == "data: [DONE]"

    # First data chunk should contain role
    first = json.loads(data_lines[0].removeprefix("data: "))
    assert first["choices"][0]["delta"]["role"] == "assistant"

    # Content chunks
    content = ""
    for line in data_lines[1:-2]:  # skip role, finish, and DONE
        chunk = json.loads(line.removeprefix("data: "))
        content += chunk["choices"][0]["delta"]["content"]
    assert len(content) > 0


@pytest.mark.anyio
async def test_max_tokens_respected(client: AsyncClient):
    payload = {**CHAT_PAYLOAD, "max_tokens": 3, "stream": False}
    resp = await client.post("/v1/chat/completions", json=payload)
    data = resp.json()
    assert data["usage"]["completion_tokens"] == 3


@pytest.mark.anyio
async def test_max_tokens_not_specified(client: AsyncClient):
    payload = {
        "model": "dummy",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    data = resp.json()
    # Default max_tokens is 16
    assert data["usage"]["completion_tokens"] == 16
