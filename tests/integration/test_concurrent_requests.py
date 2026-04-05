"""Tests for concurrent requests."""

import asyncio
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

from sim_adapter import make_sim_app
from xpyd.proxy import Proxy, RoundRobinSchedulingPolicy

prefill_app = make_sim_app(mode='prefill')
decode_app = make_sim_app(mode='decode')

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


CHAT_PAYLOAD = {
    "model": "dummy",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 5,
    "stream": False,
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
async def test_concurrent_chat_completions(client: AsyncClient):
    """15 concurrent non-streaming requests should all succeed with unique ids."""
    concurrency = 15
    payloads = [
        {**CHAT_PAYLOAD, "messages": [{"role": "user", "content": f"Hello {idx}"}]}
        for idx in range(concurrency)
    ]

    tasks = [client.post("/v1/chat/completions", json=p) for p in payloads]
    responses = await asyncio.gather(*tasks)

    ids = set()
    for resp in responses:
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) >= 1
        ids.add(data["id"])

    # Every response must have a unique id
    assert len(ids) == concurrency


@pytest.mark.anyio
async def test_concurrent_streaming(client: AsyncClient):
    """10 concurrent streaming requests should all produce valid SSE."""
    concurrency = 10
    payload = {**CHAT_PAYLOAD, "stream": True}

    tasks = [
        client.post("/v1/chat/completions", json=payload) for _ in range(concurrency)
    ]
    responses = await asyncio.gather(*tasks)

    for resp in responses:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        lines = resp.text.strip().split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]
        assert len(data_lines) >= 2
        assert data_lines[-1] == "data: [DONE]"


@pytest.mark.anyio
async def test_mixed_concurrent_streaming_and_non_streaming(client: AsyncClient):
    """Streaming and non-streaming requests running concurrently."""
    non_stream_tasks = [
        client.post(
            "/v1/chat/completions",
            json={
                **CHAT_PAYLOAD,
                "messages": [{"role": "user", "content": f"ns-{idx}"}],
            },
        )
        for idx in range(8)
    ]
    stream_tasks = [
        client.post(
            "/v1/chat/completions",
            json={
                **CHAT_PAYLOAD,
                "stream": True,
                "messages": [{"role": "user", "content": f"s-{idx}"}],
            },
        )
        for idx in range(8)
    ]

    responses = await asyncio.gather(*non_stream_tasks, *stream_tasks)
    non_stream_responses = responses[:8]
    stream_responses = responses[8:]

    ns_ids = set()
    for resp in non_stream_responses:
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["usage"]["completion_tokens"] == CHAT_PAYLOAD["max_tokens"]
        ns_ids.add(data["id"])

    assert len(ns_ids) == 8, "Non-streaming response ids must be unique"

    for resp in stream_responses:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        lines = resp.text.strip().split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]
        assert data_lines[-1] == "data: [DONE]"
        # Verify each chunk is valid JSON
        for line in data_lines[:-1]:
            chunk = json.loads(line.removeprefix("data: "))
            assert chunk["object"] == "chat.completion.chunk"
