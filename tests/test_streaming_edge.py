"""Tests for streaming edge cases."""

import json
import os
import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from httpx import ASGITransport, AsyncClient
from MicroPDProxyServer import Proxy, RoundRobinSchedulingPolicy

from dummy_nodes.decode_node import app as decode_app
from dummy_nodes.prefill_node import app as prefill_app

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOKENIZER_PATH = os.path.join(_REPO_ROOT, "tokenizers", "DeepSeek-R1")


def _free_port():
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


_PREFILL_PORT = _free_port()
_DECODE_PORT = _free_port()


def _run_server(app, port):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    uvicorn.Server(config).run()


threading.Thread(target=_run_server, args=(prefill_app, _PREFILL_PORT), daemon=True).start()
threading.Thread(target=_run_server, args=(decode_app, _DECODE_PORT), daemon=True).start()
time.sleep(2)


def _make_proxy_app():
    proxy = Proxy(
        prefill_instances=[f"127.0.0.1:{_PREFILL_PORT}"],
        decode_instances=[f"127.0.0.1:{_DECODE_PORT}"],
        model=_TOKENIZER_PATH,
        scheduling_policy=RoundRobinSchedulingPolicy(),
        generator_on_p_node=False,
    )
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(proxy.router)
    return app


CHAT_PAYLOAD = {
    "model": "dummy",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 5,
    "stream": True,
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    app = _make_proxy_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as cli:
        yield cli


@pytest.mark.anyio
async def test_streaming_chunk_structure(client: AsyncClient):
    """Each SSE chunk (except [DONE]) should be valid JSON with expected fields."""
    resp = await client.post("/v1/chat/completions", json=CHAT_PAYLOAD)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = resp.text.strip().split("\n")
    data_lines = [line for line in lines if line.startswith("data: ") and line != "data: [DONE]"]
    assert len(data_lines) >= 1, "Expected at least one data chunk before [DONE]"

    chunk_ids = set()
    for line in data_lines:
        payload = line.removeprefix("data: ")
        chunk = json.loads(payload)
        assert "id" in chunk
        assert chunk["object"] == "chat.completion.chunk"
        assert "choices" in chunk
        assert len(chunk["choices"]) >= 1
        assert "delta" in chunk["choices"][0]
        chunk_ids.add(chunk["id"])

    # All chunks in a single response should share the same id
    assert len(chunk_ids) == 1, f"Expected one unique id across chunks, got {chunk_ids}"


@pytest.mark.anyio
async def test_streaming_max_tokens_one(client: AsyncClient):
    """Streaming with max_tokens=1 should produce exactly one content chunk + [DONE]."""
    payload = {
        "model": "dummy",
        "messages": [{"role": "user", "content": "Say something"}],
        "max_tokens": 1,
        "stream": True,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = resp.text.strip().split("\n")
    data_lines = [line for line in lines if line.startswith("data: ")]
    assert data_lines[-1] == "data: [DONE]"

    content_chunks = [
        line for line in data_lines
        if line != "data: [DONE]"
    ]
    # With max_tokens=1, expect exactly 1 content chunk
    assert len(content_chunks) >= 1

    # Verify the chunk is valid
    chunk = json.loads(content_chunks[0].removeprefix("data: "))
    assert chunk["object"] == "chat.completion.chunk"
    assert "delta" in chunk["choices"][0]
