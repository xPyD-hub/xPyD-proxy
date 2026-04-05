"""Tests for sim_adapter node apps (replaces test_prefill_node.py + test_decode_node.py)."""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from sim_adapter import make_sim_app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def prefill_client():
    app = make_sim_app(mode="prefill")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
async def decode_client():
    app = make_sim_app(mode="decode")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


CHAT_PAYLOAD = {
    "model": "dummy",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 5,
    "stream": False,
}


# --- Prefill node tests ---


@pytest.mark.anyio
async def test_prefill_health(prefill_client):
    resp = await prefill_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.anyio
async def test_prefill_non_streaming(prefill_client):
    resp = await prefill_client.post("/v1/chat/completions", json=CHAT_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert len(data["choices"][0]["message"]["content"]) > 0


@pytest.mark.anyio
async def test_prefill_streaming(prefill_client):
    payload = {**CHAT_PAYLOAD, "stream": True}
    resp = await prefill_client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    lines = resp.text.strip().split("\n")
    data_lines = [line for line in lines if line.startswith("data: ")]
    assert data_lines[-1] == "data: [DONE]"
    first = json.loads(data_lines[0].removeprefix("data: "))
    assert first["choices"][0]["delta"]["role"] == "assistant"


@pytest.mark.anyio
async def test_prefill_max_tokens(prefill_client):
    payload = {**CHAT_PAYLOAD, "max_tokens": 3}
    resp = await prefill_client.post("/v1/chat/completions", json=payload)
    assert resp.json()["usage"]["completion_tokens"] == 3


@pytest.mark.anyio
async def test_prefill_default_max_tokens(prefill_client):
    payload = {"model": "dummy", "messages": [{"role": "user", "content": "Hi"}]}
    resp = await prefill_client.post("/v1/chat/completions", json=payload)
    assert resp.json()["usage"]["completion_tokens"] == 16


# --- Decode node tests ---


@pytest.mark.anyio
async def test_decode_health(decode_client):
    resp = await decode_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.anyio
async def test_decode_non_streaming(decode_client):
    resp = await decode_client.post("/v1/chat/completions", json=CHAT_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"


@pytest.mark.anyio
async def test_decode_streaming(decode_client):
    payload = {**CHAT_PAYLOAD, "stream": True}
    resp = await decode_client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "data: [DONE]" in resp.text
    first_data = [line for line in resp.text.split("\n") if line.startswith("data: ") and line != "data: [DONE]"][0]
    first = json.loads(first_data.removeprefix("data: "))
    assert first["choices"][0]["delta"]["role"] == "assistant"


@pytest.mark.anyio
async def test_decode_max_tokens(decode_client):
    payload = {**CHAT_PAYLOAD, "max_tokens": 10}
    resp = await decode_client.post("/v1/chat/completions", json=payload)
    assert resp.json()["usage"]["completion_tokens"] == 10


@pytest.mark.anyio
async def test_decode_streaming_has_content(decode_client):
    payload = {**CHAT_PAYLOAD, "max_tokens": 7, "stream": True}
    resp = await decode_client.post("/v1/chat/completions", json=payload)
    data_lines = [line for line in resp.text.strip().split("\n")
                  if line.startswith("data: ") and line != "data: [DONE]"]
    content_chunks = sum(1 for line in data_lines
                        if json.loads(line.removeprefix("data: "))["choices"][0]["delta"].get("content"))
    assert content_chunks >= 1
