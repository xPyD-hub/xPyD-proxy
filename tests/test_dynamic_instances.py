"""Tests for dynamic instance management (add/remove)."""

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
async def test_add_prefill_instance(client: AsyncClient):
    """Adding an instance without ADMIN_API_KEY set should fail (500)."""
    resp = await client.post(
        "/instances/add",
        json={"type": "prefill", "instance": "127.0.0.1:9999"},
        headers={"x-api-key": "test-key"},
    )
    # Without ADMIN_API_KEY env var set, should get 500
    assert resp.status_code == 500


@pytest.mark.anyio
async def test_add_decode_instance(client: AsyncClient):
    """Adding a decode instance without proper auth should fail."""
    resp = await client.post(
        "/instances/add",
        json={"type": "decode", "instance": "127.0.0.1:9999"},
        headers={"x-api-key": "test-key"},
    )
    assert resp.status_code == 500


@pytest.mark.anyio
async def test_remove_instance(client: AsyncClient):
    """Proxy.remove_instance_endpoint is a no-op currently; verify it exists."""
    from MicroPDProxyServer import Proxy
    proxy = Proxy(
        prefill_instances=["p:1"],
        decode_instances=["d:1"],
        model=_TOKENIZER_PATH,
        scheduling_policy=RoundRobinSchedulingPolicy(),
        generator_on_p_node=False,
    )
    # Should not raise
    result = proxy.remove_instance_endpoint("prefill", "p:1")
    assert result is None
