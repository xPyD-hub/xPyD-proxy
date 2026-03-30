"""Tests for API key authentication on admin endpoints."""

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


INSTANCE_PAYLOAD = {
    "type": "prefill",
    "instance": f"127.0.0.1:{_PREFILL_PORT}",
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    app = _make_proxy_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Dev mode: ADMIN_API_KEY not set → auth is skipped
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_no_api_key_env_allows_access(client: AsyncClient, monkeypatch):
    """When ADMIN_API_KEY is not set, admin endpoints are accessible without key."""
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    # /instances/add should not return 401/403 (auth skipped)
    # May fail on validation (instance already exists), that's fine
    resp = await client.post("/instances/add", json=INSTANCE_PAYLOAD)
    assert resp.status_code not in (401, 403)


@pytest.mark.anyio
async def test_no_api_key_env_remove_accessible(client: AsyncClient, monkeypatch):
    """When ADMIN_API_KEY is not set, /instances/remove is accessible."""
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    resp = await client.post(
        "/instances/remove",
        json={"type": "prefill", "instance": "1.2.3.4:9999"},
    )
    # Auth skipped; will get 404 (instance not found) — not 401/403
    assert resp.status_code not in (401, 403)


# ---------------------------------------------------------------------------
# With ADMIN_API_KEY set: missing key → 401
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_missing_key_returns_401(client: AsyncClient, monkeypatch):
    """Request without x-api-key header returns 401 when key is required."""
    monkeypatch.setenv("ADMIN_API_KEY", "secret-123")
    resp = await client.post("/instances/add", json=INSTANCE_PAYLOAD)
    assert resp.status_code == 401
    assert "Missing" in resp.json()["detail"]


@pytest.mark.anyio
async def test_missing_key_remove_returns_401(client: AsyncClient, monkeypatch):
    """Same for /instances/remove."""
    monkeypatch.setenv("ADMIN_API_KEY", "secret-123")
    resp = await client.post(
        "/instances/remove",
        json={"type": "decode", "instance": "1.2.3.4:8000"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Wrong key → 403
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_wrong_key_returns_403(client: AsyncClient, monkeypatch):
    """Request with incorrect x-api-key returns 403."""
    monkeypatch.setenv("ADMIN_API_KEY", "correct-key")
    resp = await client.post(
        "/instances/add",
        json=INSTANCE_PAYLOAD,
        headers={"x-api-key": "wrong-key"},
    )
    assert resp.status_code == 403
    assert "Forbidden" in resp.json()["detail"]


@pytest.mark.anyio
async def test_wrong_key_remove_returns_403(client: AsyncClient, monkeypatch):
    """Same for /instances/remove."""
    monkeypatch.setenv("ADMIN_API_KEY", "correct-key")
    resp = await client.post(
        "/instances/remove",
        json={"type": "prefill", "instance": "1.2.3.4:8000"},
        headers={"x-api-key": "wrong-key"},
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Correct key → auth passes (endpoint logic takes over)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_correct_key_passes_auth(client: AsyncClient, monkeypatch):
    """Request with correct key passes auth. Endpoint returns 200 or 400."""
    monkeypatch.setenv("ADMIN_API_KEY", "my-secret")
    resp = await client.post(
        "/instances/add",
        json=INSTANCE_PAYLOAD,
        headers={"x-api-key": "my-secret"},
    )
    # Auth passes → endpoint logic runs (may fail on duplicate instance)
    assert resp.status_code in (200, 400)
    assert resp.status_code != 401
    assert resp.status_code != 403


@pytest.mark.anyio
async def test_correct_key_remove_passes_auth(client: AsyncClient, monkeypatch):
    """Correct key on /instances/remove passes auth."""
    monkeypatch.setenv("ADMIN_API_KEY", "my-secret")
    resp = await client.post(
        "/instances/remove",
        json={"type": "decode", "instance": "1.2.3.4:9999"},
        headers={"x-api-key": "my-secret"},
    )
    # Auth passes → 404 because instance doesn't exist
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /instances/remove functional tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_remove_nonexistent_instance(client: AsyncClient, monkeypatch):
    """Removing an instance that doesn't exist returns 404."""
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    resp = await client.post(
        "/instances/remove",
        json={"type": "prefill", "instance": "1.2.3.4:9999"},
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_remove_invalid_type(client: AsyncClient, monkeypatch):
    """Invalid instance type returns 400."""
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    resp = await client.post(
        "/instances/remove",
        json={"type": "invalid", "instance": "1.2.3.4:8000"},
    )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_remove_invalid_format(client: AsyncClient, monkeypatch):
    """Invalid instance format returns 400."""
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    resp = await client.post(
        "/instances/remove",
        json={"type": "prefill", "instance": "no-port"},
    )
    assert resp.status_code == 400
