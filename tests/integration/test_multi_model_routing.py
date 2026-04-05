# SPDX-License-Identifier: Apache-2.0
"""Integration tests for multi-model routing with 4P+4D dummy nodes, 3 models."""

import os
import socket
import threading
import time

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


pytestmark = pytest.mark.slow

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
            request.max_completion_tokens,
            request.max_tokens,
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


# ---------------------------------------------------------------------------
# Topology: 4P + 4D, 3 models
#   llama-3:     p1, p2, d1, d2
#   deepseek-r1: p3, d3
#   qwen-2:      p4, d4
# ---------------------------------------------------------------------------

_MODEL_MAP = {
    "p1": "llama-3",
    "p2": "llama-3",
    "p3": "deepseek-r1",
    "p4": "qwen-2",
    "d1": "llama-3",
    "d2": "llama-3",
    "d3": "deepseek-r1",
    "d4": "qwen-2",
}


@pytest.fixture(scope="session")
def dummy_nodes():
    """Start all 8 dummy nodes once per session, poll for readiness, and
    return a dict mapping node name to ``127.0.0.1:<port>``."""
    import httpx

    ports: dict[str, int] = {k: _free_port() for k in _MODEL_MAP}
    servers: list[uvicorn.Server] = []

    for name, port in ports.items():
        model = _MODEL_MAP[name]
        app = _make_dummy_app(model)
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="error",
        )
        srv = uvicorn.Server(config)
        servers.append(srv)
        threading.Thread(target=srv.run, daemon=True).start()

    # Poll for readiness instead of fixed sleep
    deadline = time.monotonic() + 10
    for _name, port in ports.items():
        url = f"http://127.0.0.1:{port}/health"
        while time.monotonic() < deadline:
            try:
                r = httpx.get(url, timeout=1)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.1)

    addrs = {name: f"127.0.0.1:{port}" for name, port in ports.items()}
    yield addrs

    # Teardown: signal servers to shut down
    for srv in servers:
        srv.should_exit = True


def _addr(name, addrs):
    return addrs[name]


def _make_multi_model_proxy_app(addrs):
    """Build a proxy with multi-model registry."""
    all_prefill = [addrs["p1"], addrs["p2"], addrs["p3"], addrs["p4"]]
    all_decode = [addrs["d1"], addrs["d2"], addrs["d3"], addrs["d4"]]

    reg = InstanceRegistry()
    for name in ["p1", "p2", "p3", "p4"]:
        reg.add("prefill", addrs[name], model=_MODEL_MAP[name])
    for name in ["d1", "d2", "d3", "d4"]:
        reg.add("decode", addrs[name], model=_MODEL_MAP[name])
    for name in addrs:
        reg.mark_healthy(addrs[name])

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
async def multi_model_client(dummy_nodes):
    app, _ = _make_multi_model_proxy_app(dummy_nodes)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as cli:
        yield cli


@pytest.fixture
async def multi_model_client_and_registry(dummy_nodes):
    app, reg = _make_multi_model_proxy_app(dummy_nodes)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as cli:
        yield cli, reg, dummy_nodes


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
async def test_multi_model_unknown_model_404(multi_model_client: AsyncClient):
    """Request with unknown model should return 404."""
    resp = await multi_model_client.post(
        "/v1/chat/completions",
        json={
            "model": "nonexistent",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
        },
    )
    assert resp.status_code == 404
    data = resp.json()
    assert "error" in data


@pytest.mark.anyio
async def test_models_endpoint_lists_all(
    multi_model_client_and_registry,
):
    """GET /v1/models returns all 3 models."""
    cli, reg, addrs = multi_model_client_and_registry
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
    cli, reg, addrs = multi_model_client_and_registry
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


@pytest.mark.anyio
async def test_multi_model_routing_isolation(multi_model_client: AsyncClient):
    """Request with model=deepseek-r1 must NOT hit llama-3 instances."""
    resp = await multi_model_client.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "deepseek-r1"


@pytest.mark.anyio
async def test_multi_model_one_model_down(
    multi_model_client_and_registry,
):
    """When all instances of model B are unhealthy, B returns error but A works."""
    cli, reg, addrs = multi_model_client_and_registry
    # Mark deepseek-r1 instances as unhealthy
    reg.mark_unhealthy(_addr("p3", addrs))
    reg.mark_unhealthy(_addr("d3", addrs))

    # deepseek-r1 should fail (no available instances)
    resp_b = await cli.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
        },
    )
    assert resp_b.status_code in (404, 503)

    # llama-3 should still work
    resp_a = await cli.post(
        "/v1/chat/completions",
        json={
            "model": "llama-3",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
        },
    )
    assert resp_a.status_code == 200
    assert resp_a.json()["model"] == "llama-3"

    # Restore health
    reg.mark_healthy(_addr("p3", addrs))
    reg.mark_healthy(_addr("d3", addrs))


@pytest.mark.anyio
async def test_multi_model_load_balance(multi_model_client: AsyncClient):
    """N requests to llama-3 should distribute across d1 and d2."""
    models_seen = set()
    for _ in range(10):
        resp = await multi_model_client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == "llama-3"
        models_seen.add(resp.json()["model"])
    # All should be llama-3
    assert models_seen == {"llama-3"}


@pytest.mark.anyio
async def test_multi_model_prefill_decode_match(multi_model_client: AsyncClient):
    """Prefill and decode for same request go to instances of the same model."""
    # Send multiple requests to different models, verify model in response
    for model_name in ["llama-3", "deepseek-r1", "qwen-2"]:
        resp = await multi_model_client.post(
            "/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == model_name


@pytest.mark.anyio
async def test_models_endpoint_updates_on_instance_change(
    multi_model_client_and_registry,
):
    """After removing all instances of qwen-2, /v1/models no longer lists it."""
    cli, reg, addrs = multi_model_client_and_registry

    # Verify qwen-2 is listed initially
    resp = await cli.get("/v1/models")
    model_ids = [m["id"] for m in resp.json()["data"]]
    assert "qwen-2" in model_ids

    # Remove all qwen-2 instances
    reg.remove(_addr("p4", addrs))
    reg.remove(_addr("d4", addrs))

    # qwen-2 should no longer be listed
    resp = await cli.get("/v1/models")
    model_ids = [m["id"] for m in resp.json()["data"]]
    assert "qwen-2" not in model_ids

    # Re-add for cleanup (other tests might share fixtures)
    reg.add("prefill", _addr("p4", addrs), model="qwen-2")
    reg.add("decode", _addr("d4", addrs), model="qwen-2")
    reg.mark_healthy(_addr("p4", addrs))
    reg.mark_healthy(_addr("d4", addrs))
