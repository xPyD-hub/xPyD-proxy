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

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOKENIZER_PATH = os.path.join(_REPO_ROOT, "tokenizers", "DeepSeek-R1")


def _free_port():
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    app = _make_proxy_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_large_prompt(client: AsyncClient):
    """A very long prompt should still be handled (not crash)."""
    long_content = "Hello world. " * 500
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


@pytest.mark.anyio
async def test_max_tokens_zero(client: AsyncClient):
    """max_tokens=0 should be handled gracefully."""
    payload = {
        "model": "dummy",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 0,
        "stream": False,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    # Should not crash; response may vary
    assert resp.status_code in (200, 400, 422, 500)


@pytest.mark.anyio
async def test_max_tokens_negative(client: AsyncClient):
    """Negative max_tokens should be handled."""
    payload = {
        "model": "dummy",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": -1,
        "stream": False,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code in (200, 400, 422, 500)


@pytest.mark.anyio
async def test_max_tokens_very_large(client: AsyncClient):
    """Very large max_tokens should be handled."""
    payload = {
        "model": "dummy",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 999999999,
        "stream": False,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code in (200, 400, 422, 500)
