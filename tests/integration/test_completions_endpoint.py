"""Tests for /v1/completions endpoint."""

import pytest
from httpx import AsyncClient

COMPLETION_PAYLOAD = {
    "model": "dummy",
    "prompt": "Once upon a time",
    "max_tokens": 5,
    "stream": False,
}


@pytest.mark.anyio
async def test_non_streaming_completion(client: AsyncClient):
    resp = await client.post("/v1/completions", json=COMPLETION_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "text_completion"
    assert len(data["choices"]) >= 1
    assert data["choices"][0]["finish_reason"] in ("stop", "length")
    assert len(data["choices"][0]["text"]) > 0


@pytest.mark.anyio
async def test_streaming_completion(client: AsyncClient):
    payload = {**COMPLETION_PAYLOAD, "stream": True}
    resp = await client.post("/v1/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = resp.text.strip().split("\n")
    data_lines = [line for line in lines if line.startswith("data: ")]
    assert len(data_lines) >= 2
    assert data_lines[-1] == "data: [DONE]"


@pytest.mark.anyio
async def test_completion_max_tokens(client: AsyncClient):
    payload = {**COMPLETION_PAYLOAD, "max_tokens": 3, "stream": False}
    resp = await client.post("/v1/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["usage"]["completion_tokens"] == 3
