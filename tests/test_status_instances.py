"""Tests for the /status/instances endpoint."""

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient
from registry import InstanceRegistry


def _build_status_app(registry: InstanceRegistry) -> FastAPI:
    """Build a minimal FastAPI app with only the /status/instances endpoint.

    This mirrors the endpoint defined in ``MicroPDProxyServer.run`` so we can
    test it in isolation without needing a full server configuration.
    """
    app = FastAPI()

    @app.get("/status/instances")
    async def _instance_status():
        result = {"prefill_instances": [], "decode_instances": []}
        all_instances = registry.get_all_instances()
        for role in ("prefill", "decode"):
            for info in all_instances:
                if info.role != role:
                    continue
                result[f"{role}_instances"].append(
                    {
                        "address": info.address,
                        "status": info.status.value,
                        "circuit": info.circuit_breaker_state.value,
                        "active_requests": info.active_request_count,
                        "last_check": info.last_health_check,
                    }
                )
        return JSONResponse(result)

    return app


@pytest.fixture
def registry() -> InstanceRegistry:
    """Create a registry with known prefill and decode instances."""
    reg = InstanceRegistry()
    reg.add("prefill", "10.0.0.1:8000")
    reg.add("prefill", "10.0.0.2:8000")
    reg.add("decode", "10.0.0.3:8000")
    # Mark one prefill healthy so we get a mix of statuses
    reg.mark_healthy("10.0.0.1:8000")
    return reg


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_status_instances_returns_all_instances(registry):
    """GET /status/instances should list every registered instance."""
    app = _build_status_app(registry)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/status/instances")

    assert resp.status_code == 200
    data = resp.json()

    # Both keys must be present
    assert "prefill_instances" in data
    assert "decode_instances" in data

    assert len(data["prefill_instances"]) == 2
    assert len(data["decode_instances"]) == 1

    # Verify required fields on every instance entry
    required_fields = {"address", "status", "circuit", "active_requests", "last_check"}
    for section in ("prefill_instances", "decode_instances"):
        for entry in data[section]:
            assert required_fields.issubset(entry.keys()), (
                f"Missing fields in {section} entry: "
                f"{required_fields - entry.keys()}"
            )

    # Check specific addresses
    prefill_addrs = {e["address"] for e in data["prefill_instances"]}
    assert prefill_addrs == {"10.0.0.1:8000", "10.0.0.2:8000"}

    decode_addrs = {e["address"] for e in data["decode_instances"]}
    assert decode_addrs == {"10.0.0.3:8000"}


@pytest.mark.anyio
async def test_status_instances_reflects_health(registry):
    """Healthy instance should report status=healthy; others stay unknown."""
    app = _build_status_app(registry)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/status/instances")

    data = resp.json()
    by_addr = {e["address"]: e for e in data["prefill_instances"]}

    assert by_addr["10.0.0.1:8000"]["status"] == "healthy"
    assert by_addr["10.0.0.2:8000"]["status"] == "unknown"


@pytest.mark.anyio
async def test_status_instances_empty_registry():
    """An empty registry should return empty lists."""
    reg = InstanceRegistry()
    app = _build_status_app(reg)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/status/instances")

    data = resp.json()
    assert data == {"prefill_instances": [], "decode_instances": []}
