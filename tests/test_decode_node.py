"""Tests for the dummy decode node."""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from dummy_nodes.decode_node import app


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
    assert data["node_type"] == "decode"


@pytest.mark.anyio
async def test_non_streaming(client: AsyncClient):
    resp = await client.post("/v1/chat/completions", json=CHAT_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()

    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert len(data["choices"][0]["message"]["content"]) > 0

    assert data["usage"]["completion_tokens"] == 5
    assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"] + 5


@pytest.mark.anyio
async def test_streaming(client: AsyncClient):
    payload = {**CHAT_PAYLOAD, "stream": True}
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = resp.text.strip().split("\n")
    data_lines = [line for line in lines if line.startswith("data: ")]

    # 1 role + 5 content + 1 finish + 1 [DONE] = 8
    assert len(data_lines) == 8

    assert data_lines[-1] == "data: [DONE]"

    first = json.loads(data_lines[0].removeprefix("data: "))
    assert first["choices"][0]["delta"]["role"] == "assistant"

    content = ""
    for line in data_lines[1:-2]:
        chunk = json.loads(line.removeprefix("data: "))
        content += chunk["choices"][0]["delta"]["content"]
    assert len(content) > 0


@pytest.mark.anyio
async def test_max_tokens_respected(client: AsyncClient):
    payload = {**CHAT_PAYLOAD, "max_tokens": 10, "stream": False}
    resp = await client.post("/v1/chat/completions", json=payload)
    data = resp.json()
    assert data["usage"]["completion_tokens"] == 10


@pytest.mark.anyio
async def test_streaming_token_count(client: AsyncClient):
    """Verify the number of content tokens in streaming matches max_tokens."""
    payload = {**CHAT_PAYLOAD, "max_tokens": 7, "stream": True}
    resp = await client.post("/v1/chat/completions", json=payload)

    lines = resp.text.strip().split("\n")
    data_lines = [
        line for line in lines if line.startswith("data: ") and line != "data: [DONE]"
    ]

    # Count content chunks (exclude role-only and finish-only chunks)
    content_chunks = 0
    for line in data_lines:
        chunk = json.loads(line.removeprefix("data: "))
        delta = chunk["choices"][0]["delta"]
        if delta.get("content") is not None:
            content_chunks += 1

    assert content_chunks == 7
