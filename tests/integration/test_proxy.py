"""Tests for MicroPDProxyServer."""

import itertools
import json
import os
import socket
import threading
import time
from unittest.mock import patch

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from httpx import ASGITransport, AsyncClient

from dummy_nodes.decode_node import app as decode_app
from dummy_nodes.prefill_node import app as prefill_app
from xpyd.proxy import LoadBalancedScheduler, Proxy, RoundRobinSchedulingPolicy

# ---------------------------------------------------------------------------
# Use local tokenizer from repo to avoid network dependency in CI
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_TOKENIZER_PATH = os.path.join(_REPO_ROOT, "tokenizers", "DeepSeek-R1")


def _make_proxy_app(
    prefill_instances: list | None = None,
    decode_instances: list | None = None,
) -> FastAPI:
    """Create a FastAPI app with a Proxy router for testing."""
    proxy = Proxy(
        prefill_instances=prefill_instances or [],
        decode_instances=decode_instances or [],
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


# ---------------------------------------------------------------------------
# Helpers – start real dummy-node servers for the proxy to talk to
# ---------------------------------------------------------------------------


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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
time.sleep(1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    proxy_app = _make_proxy_app(
        prefill_instances=[f"127.0.0.1:{_PREFILL_PORT}"],
        decode_instances=[f"127.0.0.1:{_DECODE_PORT}"],
    )
    transport = ASGITransport(app=proxy_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


CHAT_PAYLOAD = {
    "model": "dummy",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 5,
    "stream": False,
}


# ---------------------------------------------------------------------------
# Proxy endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    # Proxy /health returns per-instance results keyed by host:port
    assert len(data) > 0
    for _inst, info in data.items():
        assert info["status"] == 200
        assert info["data"]["status"] == "ok"


@pytest.mark.anyio
async def test_status(client: AsyncClient):
    resp = await client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["prefill_node_count"] == 1
    assert data["decode_node_count"] == 1
    assert len(data["prefill_nodes"]) == 1
    assert len(data["decode_nodes"]) == 1


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
    payload = {**CHAT_PAYLOAD, "max_tokens": 3, "stream": False}
    resp = await client.post("/v1/chat/completions", json=payload)
    data = resp.json()
    assert data["usage"]["completion_tokens"] == 3


@pytest.mark.anyio
async def test_streaming_token_count(client: AsyncClient):
    """Verify the number of content tokens in streaming matches max_tokens."""
    payload = {**CHAT_PAYLOAD, "max_tokens": 7, "stream": True}
    resp = await client.post("/v1/chat/completions", json=payload)

    lines = resp.text.strip().split("\n")
    data_lines = [
        line for line in lines if line.startswith("data: ") and line != "data: [DONE]"
    ]

    content_chunks = 0
    for line in data_lines:
        chunk = json.loads(line.removeprefix("data: "))
        delta = chunk["choices"][0]["delta"]
        if delta.get("content") is not None:
            content_chunks += 1

    assert content_chunks == 7


# ---------------------------------------------------------------------------
# Unit tests – scheduling policies
# ---------------------------------------------------------------------------


def test_round_robin_scheduling():
    policy = RoundRobinSchedulingPolicy()
    instances = ["a:1", "b:2", "c:3"]
    cycler = itertools.cycle(instances)
    results = [policy.schedule(cycler) for _ in range(6)]
    assert results == ["a:1", "b:2", "c:3", "a:1", "b:2", "c:3"]


def test_round_robin_schedule_with_full_signature():
    """Verify RoundRobin.schedule() accepts the same args Proxy.schedule() passes."""
    policy = RoundRobinSchedulingPolicy()
    instances = ["a:1", "b:2"]
    cycler = itertools.cycle(instances)

    r1 = policy.schedule(cycler, True, 100, 1)
    r2 = policy.schedule(cycler, False, 100, 50)
    assert r1 == "a:1"
    assert r2 == "b:2"


def test_round_robin_schedule_completion_exists():
    """Verify RoundRobin has schedule_completion() (no-op from base class)."""
    policy = RoundRobinSchedulingPolicy()
    policy.schedule_completion(
        prefill_instance="a:1", decode_instance=None, req_len=100
    )
    policy.schedule_completion(prefill_instance=None, decode_instance="b:2", req_len=50)


@patch(
    "xpyd.scheduler.load_balanced.query_instance_model_len",
    return_value=[131072, 131072],
)
def test_load_balanced_scheduling(mock_query):
    """Test LoadBalancedScheduler distributes requests across instances."""
    prefill = ["p1:1", "p2:2"]
    decode = ["d1:1", "d2:2"]
    policy = LoadBalancedScheduler(prefill, decode)

    p_cycler = itertools.cycle(prefill)
    d_cycler = itertools.cycle(decode)

    r1 = policy.schedule(p_cycler, is_prompt=True, request_len=100, max_tokens=50)
    assert r1 in prefill

    r2 = policy.schedule(p_cycler, is_prompt=True, request_len=100, max_tokens=50)
    assert r2 in prefill
    assert r2 != r1

    d1 = policy.schedule(d_cycler, is_prompt=False, request_len=50, max_tokens=50)
    assert d1 in decode
    d2 = policy.schedule(d_cycler, is_prompt=False, request_len=50, max_tokens=50)
    assert d2 in decode
    assert d2 != d1
