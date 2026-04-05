# SPDX-License-Identifier: Apache-2.0
"""Integration tests for dual-role instances with mock servers."""

import os
import random
import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from httpx import ASGITransport, AsyncClient

from xpyd.proxy import Proxy
from xpyd.registry import InstanceRegistry
from xpyd.scheduler import RoundRobinSchedulingPolicy

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_TOKENIZER_PATH = os.path.join(_REPO_ROOT, "tokenizers", "DeepSeek-R1")


def _free_port():
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _make_dummy_app(model_id: str):
    from sim_adapter import make_sim_app
    return make_sim_app(model_id)



# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dual_nodes():
    """Start 8 dummy nodes for dual/PD tests."""
    import httpx

    node_models = {
        "dual1": "qwen-2",
        "dual2": "qwen-2",
        "p1": "llama-3",
        "p2": "llama-3",
        "d1": "llama-3",
        "d2": "llama-3",
        "dual3": "deepseek-r1",
        "dual4": "deepseek-r1",
    }
    ports = {k: _free_port() for k in node_models}
    servers = []

    for name, port in ports.items():
        model = node_models[name]
        app = _make_dummy_app(model)
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        srv = uvicorn.Server(config)
        servers.append(srv)
        threading.Thread(target=srv.run, daemon=True).start()

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
    yield addrs, node_models

    for srv in servers:
        srv.should_exit = True


def _make_dual_proxy_app(addrs, node_models):
    """Build a proxy with dual + P/D models."""
    reg = InstanceRegistry()
    dual_instances = {}

    # Register dual instances
    for name in ["dual1", "dual2", "dual3", "dual4"]:
        model = node_models[name]
        reg.add("dual", addrs[name], model=model)
        dual_instances.setdefault(model, []).append(addrs[name])

    # Register P/D instances
    all_prefill = []
    all_decode = []
    for name in ["p1", "p2"]:
        reg.add("prefill", addrs[name], model=node_models[name])
        all_prefill.append(addrs[name])
    for name in ["d1", "d2"]:
        reg.add("decode", addrs[name], model=node_models[name])
        all_decode.append(addrs[name])

    for addr in addrs.values():
        reg.mark_healthy(addr)

    sched = RoundRobinSchedulingPolicy(registry=reg)
    proxy = Proxy(
        prefill_instances=all_prefill,
        decode_instances=all_decode,
        model=_TOKENIZER_PATH,
        scheduling_policy=sched,
        generator_on_p_node=False,
        registry=reg,
        dual_instances=dual_instances,
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
async def dual_client(dual_nodes):
    addrs, node_models = dual_nodes
    app, reg = _make_dual_proxy_app(addrs, node_models)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as cli:
        yield cli


@pytest.fixture
async def dual_client_and_registry(dual_nodes):
    addrs, node_models = dual_nodes
    app, reg = _make_dual_proxy_app(addrs, node_models)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as cli:
        yield cli, reg


# ---------------------------------------------------------------------------
# End-to-end dual tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_dual_chat_completion(dual_client: AsyncClient):
    """Dual model request returns correct response."""
    resp = await dual_client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen-2",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "qwen-2"


@pytest.mark.anyio
async def test_dual_completions_endpoint(dual_client: AsyncClient):
    """Dual model /v1/completions returns correct response."""
    resp = await dual_client.post(
        "/v1/completions",
        json={
            "model": "qwen-2",
            "prompt": "hello world",
            "max_tokens": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "qwen-2"


@pytest.mark.anyio
async def test_pd_model_still_works(dual_client: AsyncClient):
    """P/D model request still goes through two-phase flow."""
    resp = await dual_client.post(
        "/v1/chat/completions",
        json={
            "model": "llama-3",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "llama-3"


@pytest.mark.anyio
async def test_dual_unknown_model_error(dual_client: AsyncClient):
    """Unknown model returns error."""
    resp = await dual_client.post(
        "/v1/chat/completions",
        json={
            "model": "nonexistent",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
        },
    )
    assert resp.status_code in (404, 503)


@pytest.mark.anyio
async def test_dual_all_down_503(dual_client_and_registry):
    """All dual instances down returns 503."""
    cli, reg = dual_client_and_registry
    # Mark deepseek-r1 dual instances as unhealthy
    for info in reg.get_all_instances():
        if info.model == "deepseek-r1" and info.role == "dual":
            reg.mark_unhealthy(info.address)

    resp = await cli.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
        },
    )
    assert resp.status_code == 503

    # Restore
    for info in reg.get_all_instances():
        if info.model == "deepseek-r1":
            reg.mark_healthy(info.address)


@pytest.mark.anyio
async def test_models_lists_dual_models(dual_client: AsyncClient):
    """GET /v1/models includes dual models."""
    resp = await dual_client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    model_ids = sorted(m["id"] for m in data["data"])
    assert "qwen-2" in model_ids
    assert "deepseek-r1" in model_ids
    assert "llama-3" in model_ids


# ---------------------------------------------------------------------------
# Fixed boundary tests (8 instances)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_all_dual_single_model(dual_nodes):
    """8 dual instances serving one model."""
    addrs, _ = dual_nodes
    all_addrs = list(addrs.values())

    reg = InstanceRegistry()
    dual_map = {"test-model": all_addrs}
    for addr in all_addrs:
        reg.add("dual", addr, model="test-model")
        reg.mark_healthy(addr)

    sched = RoundRobinSchedulingPolicy(registry=reg)
    proxy = Proxy(
        prefill_instances=[],
        decode_instances=[],
        model=_TOKENIZER_PATH,
        scheduling_policy=sched,
        generator_on_p_node=False,
        registry=reg,
        dual_instances=dual_map,
    )

    app = FastAPI()
    app.include_router(proxy.router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as cli:
        resp = await cli.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
            },
        )
        assert resp.status_code == 200


@pytest.mark.anyio
async def test_three_models_mixed(dual_client: AsyncClient):
    """3 models: qwen-2 (dual) + llama-3 (P/D) + deepseek-r1 (dual)."""
    for model_name in ["qwen-2", "llama-3", "deepseek-r1"]:
        resp = await dual_client.post(
            "/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == model_name


# ---------------------------------------------------------------------------
# Randomized mixed deployment (20 seeds)
# ---------------------------------------------------------------------------


def _generate_random_deployment(addrs, seed):
    """Generate a random valid deployment from 8 addresses."""
    rng = random.Random(seed)
    pool = list(addrs.values())
    rng.shuffle(pool)

    num_models = rng.randint(1, 3)
    models = []

    for i in range(num_models):
        remaining = num_models - i
        max_for_this = len(pool) - (remaining - 1) * 2
        if max_for_this < 2:
            max_for_this = 2
        count = rng.randint(2, min(max_for_this, len(pool)))
        assigned = pool[:count]
        pool = pool[count:]

        mode = rng.choice(["dual", "pd"])
        name = f"model-{i}"

        if mode == "dual":
            models.append({"name": name, "mode": "dual", "instances": assigned})
        else:
            split = rng.randint(1, len(assigned) - 1)
            models.append(
                {
                    "name": name,
                    "mode": "pd",
                    "prefill": assigned[:split],
                    "decode": assigned[split:],
                }
            )

    return models


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.anyio
async def test_randomized_deployment(dual_nodes, seed):
    """Randomized deployment with seed-based reproducibility."""
    addrs, _ = dual_nodes
    deployment = _generate_random_deployment(addrs, seed)

    reg = InstanceRegistry()
    dual_map = {}
    all_prefill = []
    all_decode = []

    for model_cfg in deployment:
        name = model_cfg["name"]
        if model_cfg["mode"] == "dual":
            dual_map[name] = model_cfg["instances"]
            for addr in model_cfg["instances"]:
                if addr not in [i.address for i in reg.get_all_instances()]:
                    reg.add("dual", addr, model=name)
        else:
            for addr in model_cfg["prefill"]:
                if addr not in [i.address for i in reg.get_all_instances()]:
                    reg.add("prefill", addr, model=name)
                    all_prefill.append(addr)
            for addr in model_cfg["decode"]:
                if addr not in [i.address for i in reg.get_all_instances()]:
                    reg.add("decode", addr, model=name)
                    all_decode.append(addr)

    for info in reg.get_all_instances():
        reg.mark_healthy(info.address)

    sched = RoundRobinSchedulingPolicy(registry=reg)
    proxy = Proxy(
        prefill_instances=all_prefill,
        decode_instances=all_decode,
        model=_TOKENIZER_PATH,
        scheduling_policy=sched,
        generator_on_p_node=False,
        registry=reg,
        dual_instances=dual_map,
    )

    app = FastAPI()
    app.include_router(proxy.router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as cli:
        for model_cfg in deployment:
            name = model_cfg["name"]
            # The dummy nodes return model IDs based on their actual model,
            # not the name we assigned. For routing verification, just check
            # we get a 200 (correct routing to a live server).
            resp = await cli.post(
                "/v1/chat/completions",
                json={
                    "model": name,
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 5,
                },
            )
            assert resp.status_code == 200, (
                f"seed={seed}, model={name}, mode={model_cfg['mode']}, "
                f"status={resp.status_code}"
            )
