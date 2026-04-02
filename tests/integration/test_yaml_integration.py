"""Integration test: start proxy from a YAML config with dummy nodes."""

from __future__ import annotations

import argparse
import socket
import textwrap
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from httpx import ASGITransport, AsyncClient

from dummy_nodes.decode_node import app as decode_app
from dummy_nodes.prefill_node import app as prefill_app
from xpyd.config import ProxyConfig
from xpyd.server import Proxy, RoundRobinSchedulingPolicy

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOKENIZER_PATH = str(_REPO_ROOT / "tokenizers" / "DeepSeek-R1")


def _free_port():
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# Start dedicated dummy nodes for YAML integration tests.
_YAML_PREFILL_PORT = _free_port()
_YAML_DECODE_PORT = _free_port()


def _run_server(app, port):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    uvicorn.Server(config).run()


threading.Thread(
    target=_run_server, args=(prefill_app, _YAML_PREFILL_PORT), daemon=True
).start()
threading.Thread(
    target=_run_server, args=(decode_app, _YAML_DECODE_PORT), daemon=True
).start()
time.sleep(2)


def _make_proxy_from_yaml(yaml_content: str, tmp_path: Path) -> Proxy:
    """Write YAML to a temp file, parse it, and build a Proxy instance."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(textwrap.dedent(yaml_content))

    args = argparse.Namespace(
        config=str(config_file),
        model=None,
        prefill=None,
        decode=None,
        port=8000,
        generator_on_p_node=False,
        roundrobin=False,
        log_level="warning",
    )
    config = ProxyConfig.from_args(args)

    return Proxy(
        prefill_instances=config.prefill,
        decode_instances=config.decode,
        model=config.model,
        scheduling_policy=RoundRobinSchedulingPolicy(),
        generator_on_p_node=config.generator_on_p_node,
    )


def _make_app(proxy: Proxy) -> FastAPI:
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
async def yaml_client(tmp_path):
    """Async HTTP client wired to a proxy started from YAML config."""
    yaml_content = f"""\
    model: {_TOKENIZER_PATH}
    prefill:
      - "127.0.0.1:{_YAML_PREFILL_PORT}"
    decode:
      - "127.0.0.1:{_YAML_DECODE_PORT}"
    scheduling: roundrobin
    """
    proxy = _make_proxy_from_yaml(yaml_content, tmp_path)
    app = _make_app(proxy)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as cli:
        yield cli


@pytest.fixture
async def yaml_topology_client(tmp_path):
    """Client from YAML config using topology-style node definition."""
    yaml_content = f"""\
    model: {_TOKENIZER_PATH}
    prefill:
      nodes:
        - "127.0.0.1:{_YAML_PREFILL_PORT}"
      tp_size: 1
      dp_size: 1
      world_size_per_node: 1
    decode:
      nodes:
        - "127.0.0.1:{_YAML_DECODE_PORT}"
      tp_size: 1
      dp_size: 1
      world_size_per_node: 1
    scheduling: roundrobin
    """
    proxy = _make_proxy_from_yaml(yaml_content, tmp_path)
    app = _make_app(proxy)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as cli:
        yield cli


@pytest.mark.anyio
async def test_yaml_config_health(yaml_client):
    """Proxy started from YAML config should respond to /health."""
    resp = await yaml_client.get("/health")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_yaml_config_non_streaming(yaml_client):
    """Non-streaming completion through YAML-configured proxy."""
    resp = await yaml_client.post(
        "/v1/completions",
        json={
            "model": "test",
            "prompt": "Hello",
            "max_tokens": 5,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data


@pytest.mark.anyio
async def test_yaml_config_streaming(yaml_client):
    """Streaming completion through YAML-configured proxy."""
    resp = await yaml_client.post(
        "/v1/completions",
        json={
            "model": "test",
            "prompt": "Hello",
            "max_tokens": 5,
            "stream": True,
        },
    )
    assert resp.status_code == 200
    body = resp.text
    assert "data:" in body


@pytest.mark.anyio
async def test_yaml_config_chat_completion(yaml_client):
    """Chat completion through YAML-configured proxy."""
    resp = await yaml_client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data


@pytest.mark.anyio
async def test_yaml_topology_health(yaml_topology_client):
    """Proxy from topology-style YAML config responds to /health."""
    resp = await yaml_topology_client.get("/health")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_yaml_topology_completion(yaml_topology_client):
    """Completion through topology-style YAML-configured proxy."""
    resp = await yaml_topology_client.post(
        "/v1/completions",
        json={
            "model": "test",
            "prompt": "Hello",
            "max_tokens": 5,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
