"""Tests for large payloads and edge cases."""

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

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
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


threading.Thread(
    target=_run_server, args=(prefill_app, _PREFILL_PORT), daemon=True
).start()
threading.Thread(
    target=_run_server, args=(decode_app, _DECODE_PORT), daemon=True
).start()
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
async def test_large_prompt(client: AsyncClient):
    """A 10K+ token prompt should still be handled without crashing."""
    # ~12K tokens: each "word_NNNN " is roughly 2 tokens
    long_content = " ".join(f"word_{idx}" for idx in range(6000))
    payload = {
        "model": "dummy",
        "messages": [{"role": "user", "content": long_content}],
        "max_tokens": 5,
        "stream": False,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert len(data["choices"]) >= 1


@pytest.mark.anyio
async def test_max_tokens_zero(client: AsyncClient):
    """max_tokens=0 → 200 with 0 completion tokens."""
    payload = {
        "model": "dummy",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 0,
        "stream": False,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["usage"]["completion_tokens"] == 0


@pytest.mark.anyio
async def test_max_tokens_negative(client: AsyncClient):
    """Negative max_tokens is passed through (proxy does not validate)."""
    payload = {
        "model": "dummy",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": -1,
        "stream": False,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    # Proxy does not validate max_tokens; backend handles it
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_max_tokens_very_large(client: AsyncClient):
    """Very large max_tokens should succeed (dummy backend caps output)."""
    payload = {
        "model": "dummy",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 999999999,
        "stream": False,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data


@pytest.mark.anyio
async def test_missing_messages_returns_400(client: AsyncClient):
    """POST /v1/chat/completions without 'messages' should return 400."""
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "dummy", "max_tokens": 5},
    )
    assert resp.status_code == 400
    assert "messages" in resp.json()["error"]["message"].lower()


@pytest.mark.anyio
async def test_missing_prompt_returns_400(client: AsyncClient):
    """POST /v1/completions without 'prompt' should return 400."""
    resp = await client.post(
        "/v1/completions",
        json={"model": "dummy", "max_tokens": 5},
    )
    assert resp.status_code == 400
    assert "prompt" in resp.json()["error"]["message"].lower()


@pytest.mark.anyio
async def test_invalid_messages_type_returns_400(client: AsyncClient):
    """POST /v1/chat/completions with non-list 'messages' should return 400."""
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "dummy", "messages": "not a list", "max_tokens": 5},
    )
    assert resp.status_code == 400
    assert "list" in resp.json()["error"]["message"].lower()
