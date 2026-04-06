"""Shared test fixtures and utilities."""

import os
import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from httpx import ASGITransport, AsyncClient

from sim_adapter import make_sim_app
from xpyd.proxy import Proxy, RoundRobinSchedulingPolicy

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOKENIZER_PATH = os.path.join(_REPO_ROOT, "tokenizers", "DeepSeek-R1")

# Create apps with the correct model name (must match proxy config)
_prefill_app = make_sim_app(mode="prefill")
_decode_app = make_sim_app(mode="decode")


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


threading.Thread(target=_run_server, args=(_prefill_app, _PREFILL_PORT), daemon=True).start()
threading.Thread(target=_run_server, args=(_decode_app, _DECODE_PORT), daemon=True).start()

# Wait for readiness
import httpx as _httpx  # noqa: E402

for _port in (_PREFILL_PORT, _DECODE_PORT):
    for _ in range(50):
        try:
            if _httpx.get(f"http://127.0.0.1:{_port}/health", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.2)
    else:
        raise RuntimeError(f"Server on port {_port} failed to start")


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
        CORSMiddleware, allow_origins=["*"], allow_credentials=False,
        allow_methods=["*"], allow_headers=["*"],
    )
    app.include_router(proxy.router)
    return app


@pytest.fixture
def dummy_ports():
    return _PREFILL_PORT, _DECODE_PORT


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    app = _make_proxy_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as cli:
        yield cli
