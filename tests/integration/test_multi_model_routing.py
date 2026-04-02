# SPDX-License-Identifier: Apache-2.0
"""Integration tests for multi-model routing with 4P+4D dummy nodes, 3 models."""

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

from xpyd.proxy import Proxy, RoundRobinSchedulingPolicy
from xpyd.registry import InstanceRegistry

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_TOKENIZER_PATH = os.path.join(_REPO_ROOT, "tokenizers", "DeepSeek-R1")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port():
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _make_dummy_app(model_id: str):
    """Create a minimal dummy node app serving a given model_id."""
    from dummy_nodes.common import (
        ChatCompletionRequest,
        ChatCompletionResponse,
        Choice,
        ChoiceMessage,
        CompletionChoice,
        CompletionRequest,
        CompletionResponse,
        ModelCard,
        ModelListResponse,
        UsageInfo,
        build_models_response,
        count_prompt_tokens_from_messages,
        count_prompt_tokens_from_prompt,
        generate_id,
        get_effective_max_tokens,
        now_ts,
        render_dummy_text,
    )

    app = FastAPI(title=f"Dummy Node ({model_id})")

    @app.get("/v1/models")
    async def models():
        return ModelListResponse(
            data=[ModelCard(id=model_id, created=now_ts(), max_model_len=131072)]
        )

    @app.get("/health")
    async def health():
        return "OK"

    @app.post("/v1/chat/completions")
    async def chat(request: ChatCompletionRequest):
        prompt_tokens = count_prompt_tokens_from_messages(request.messages)
        max_tokens = get_effective_max_tokens(
            request.max_completion_tokens, request.max_tokens,
        )
        text = render_dummy_text(max_tokens)
        return ChatCompletionResponse(
            id=generate_id(),
            created=now_ts(),
            model=model_id,
            choices=[Choice(message=ChoiceMessage(content=text))],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=len(text),
                total_tokens=prompt_tokens + len(text),
            ),
        )

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        prompt_tokens = count_prompt_tokens_from_prompt(request.prompt)
        max_tokens = get_effective_max_tokens(request.max_tokens)
        text = render_dummy_text(max_tokens)
        return CompletionResponse(
            id=generate_id("cmpl"),
            created=now_ts(),
            model=model_id,
            choices=[CompletionChoice(text=text)],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=len(text),
                total_tokens=prompt_tokens + len(text),
            ),
        )

    return app


def _run(app, port):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    uvicorn.Server(config).run()


# ---------------------------------------------------------------------------
# Topology: 4P + 4D, 3 models
#   llama-3:     p1, p2, d1, d2
#   deepseek-r1: p3, d3
#   qwen-2:      p4, d4
# ---------------------------------------------------------------------------

_PORTS = {k: _free_port() for k in [
    "p1", "p2", "p3", "p4", "d1", "d2", "d3", "d4",
]}

_MODEL_MAP = {
    "p1": "llama-3", "p2": "llama-3", "p3": "deepseek-r1", "p4": "qwen-2",
    "d1": "llama-3", "d2": "llama-3", "d3": "deepseek-r1", "d4": "qwen-2",
}

# Start all nodes
for name, port in _PORTS.items():
    model = _MODEL_MAP[name]
    app = _make_dummy_app(model)
    threading.Thread(target=_run, args=(app, port), daemon=True).start()

time.sleep(2)


def _addr(name):
    return f"127.0.0.1:{_PORTS[name]}"


def _make_multi_model_proxy_app():
    """Build a proxy with multi-model registry."""
    all_prefill = [_addr("p1"), _addr("p2"), _addr("p3"), _addr("p4")]
    all_decode = [_addr("d1"), _addr("d2"), _addr("d3"), _addr("d4")]

    reg = InstanceRegistry()
    for name in ["p1", "p2", "p3", "p4"]:
        reg.add("prefill", _addr(name), model=_MODEL_MAP[name])
    for name in ["d1", "d2", "d3", "d4"]:
        reg.add("decode", _addr(name), model=_MODEL_MAP[name])
    for name in _PORTS:
        reg.mark_healthy(_addr(name))

    sched = RoundRobinSchedulingPolicy(registry=reg)

    proxy = Proxy(
        prefill_instances=all_prefill,
        decode_instances=all_decode,
        model=_TOKENIZER_PATH,
        scheduling_policy=sched,
        generator_on_p_node=False,
        registry=reg,
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
    return app, reg


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def multi_model_client():
    app, _ = _make_multi_model_proxy_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as cli:
        yield cli


@pytest.fixture
async def multi_model_client_and_registry():
    app, reg = _make_multi_model_proxy_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as cli:
        yield cli, reg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_multi_model_routing_correct(multi_model_client: AsyncClient):
    """Send request with model=llama-3, verify response model matches."""
    resp = await multi_model_client.post(
        "/v1/chat/completions",
        json={
            "model": "llama-3",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "llama-3"


@pytest.mark.anyio
async def test_multi_model_unknown_model_503(multi_model_client: AsyncClient):
    """Request with unknown model should return 503."""
    resp = await multi_model_client.post(
        "/v1/chat/completions",
        json={
            "model": "nonexistent",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
        },
    )
    assert resp.status_code == 503
    data = resp.json()
    assert "error" in data


@pytest.mark.anyio
async def test_models_endpoint_lists_all(
    multi_model_client_and_registry,
):
    """GET /v1/models returns all 3 models."""
    cli, reg = multi_model_client_and_registry
    resp = await cli.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    model_ids = sorted(m["id"] for m in data["data"])
    assert model_ids == ["deepseek-r1", "llama-3", "qwen-2"]


@pytest.mark.anyio
async def test_models_endpoint_format(
    multi_model_client_and_registry,
):
    """Response format matches OpenAI spec."""
    cli, reg = multi_model_client_and_registry
    resp = await cli.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "object" in data
    assert "data" in data
    for model in data["data"]:
        assert "id" in model
        assert model["object"] == "model"
        assert "created" in model
        assert "owned_by" in model
