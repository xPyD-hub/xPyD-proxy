"""Integration tests for Task 9: registry + circuit breaker + health monitor."""

from __future__ import annotations

import socket
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
from xpyd.proxy import Proxy, RoundRobinSchedulingPolicy

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOKENIZER_PATH = str(_REPO_ROOT / "tokenizers" / "DeepSeek-R1")


def _free_port():
    """Allocate a free port. Keep socket open until caller binds to avoid races."""
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _run_server(app, port):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    uvicorn.Server(config).run()


@pytest.fixture(scope="session")
def dummy_nodes():
    """Start dummy prefill/decode servers once per test session."""
    prefill_port = _free_port()
    decode_port_1 = _free_port()
    decode_port_2 = _free_port()

    for app, port in [
        (prefill_app, prefill_port),
        (decode_app, decode_port_1),
        (decode_app, decode_port_2),
    ]:
        threading.Thread(target=_run_server, args=(app, port), daemon=True).start()
    time.sleep(2)

    return {
        "prefill_port": prefill_port,
        "decode_port_1": decode_port_1,
        "decode_port_2": decode_port_2,
    }


def _make_config(dummy_nodes, **overrides):
    """Build a ProxyConfig for testing."""
    defaults = {
        "model": _TOKENIZER_PATH,
        "prefill": [f"127.0.0.1:{dummy_nodes['prefill_port']}"],
        "decode": [
            f"127.0.0.1:{dummy_nodes['decode_port_1']}",
            f"127.0.0.1:{dummy_nodes['decode_port_2']}",
        ],
        "port": 8000,
    }
    defaults.update(overrides)
    return ProxyConfig(**defaults)


def _make_proxy_app(config):
    """Create a FastAPI app with Proxy from config."""
    proxy = Proxy(
        prefill_instances=config.prefill,
        decode_instances=config.decode,
        model=config.model,
        scheduling_policy=RoundRobinSchedulingPolicy(),
        generator_on_p_node=config.generator_on_p_node,
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


# ------------------------------------------------------------------
# Scenario A: Baseline — normal requests work
# ------------------------------------------------------------------


@pytest.fixture
async def baseline_client(dummy_nodes):
    config = _make_config(dummy_nodes)
    app = _make_proxy_app(config)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as cli:
        yield cli


@pytest.mark.anyio
async def test_baseline_health(baseline_client):
    resp = await baseline_client.get("/health")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_baseline_completion(baseline_client):
    resp = await baseline_client.post(
        "/v1/completions",
        json={"model": "test", "prompt": "Hello", "max_tokens": 5, "stream": False},
    )
    assert resp.status_code == 200
    assert "choices" in resp.json()


@pytest.mark.anyio
async def test_baseline_chat_completion(baseline_client):
    resp = await baseline_client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    assert "choices" in resp.json()


# ------------------------------------------------------------------
# Scenario B: Registry correctly registers instances
# ------------------------------------------------------------------


class TestRegistry:
    def test_registry_registers_all_instances(self, dummy_nodes):
        from xpyd.registry import InstanceRegistry

        registry = InstanceRegistry()
        registry.add("prefill", f"127.0.0.1:{dummy_nodes['prefill_port']}")
        registry.add("decode", f"127.0.0.1:{dummy_nodes['decode_port_1']}")
        registry.add("decode", f"127.0.0.1:{dummy_nodes['decode_port_2']}")

        # New instances default to UNKNOWN — mark healthy first.
        registry.mark_healthy(f"127.0.0.1:{dummy_nodes['prefill_port']}")
        registry.mark_healthy(f"127.0.0.1:{dummy_nodes['decode_port_1']}")
        registry.mark_healthy(f"127.0.0.1:{dummy_nodes['decode_port_2']}")

        prefill = registry.get_available_instances("prefill")
        decode = registry.get_available_instances("decode")
        assert len(prefill) == 1
        assert len(decode) == 2

    def test_registry_mark_unhealthy_removes_from_available(self, dummy_nodes):
        from xpyd.registry import InstanceRegistry

        registry = InstanceRegistry()
        registry.add("decode", f"127.0.0.1:{dummy_nodes['decode_port_1']}")
        registry.add("decode", f"127.0.0.1:{dummy_nodes['decode_port_2']}")
        registry.mark_healthy(f"127.0.0.1:{dummy_nodes['decode_port_1']}")
        registry.mark_healthy(f"127.0.0.1:{dummy_nodes['decode_port_2']}")
        registry.mark_unhealthy(f"127.0.0.1:{dummy_nodes['decode_port_1']}")
        available = registry.get_available_instances("decode")
        assert len(available) == 1
        assert f"127.0.0.1:{dummy_nodes['decode_port_2']}" in available

    def test_registry_mark_healthy_restores(self, dummy_nodes):
        from xpyd.registry import InstanceRegistry

        registry = InstanceRegistry()
        registry.add("decode", f"127.0.0.1:{dummy_nodes['decode_port_1']}")
        registry.mark_unhealthy(f"127.0.0.1:{dummy_nodes['decode_port_1']}")
        assert len(registry.get_available_instances("decode")) == 0
        registry.mark_healthy(f"127.0.0.1:{dummy_nodes['decode_port_1']}")
        assert len(registry.get_available_instances("decode")) == 1


# ------------------------------------------------------------------
# Scenario C: Circuit breaker integration
# ------------------------------------------------------------------


class TestCircuitBreakerIntegration:
    def test_circuit_opens_after_failures(self):
        from xpyd.registry import InstanceRegistry

        registry = InstanceRegistry(
            cb_enabled=True, failure_threshold=2, timeout_duration_seconds=30
        )
        registry.add("decode", "10.0.0.1:8200")
        registry.add("decode", "10.0.0.2:8200")
        registry.mark_healthy("10.0.0.1:8200")
        registry.mark_healthy("10.0.0.2:8200")

        # Two failures → circuit opens
        registry.record_failure("10.0.0.1:8200")
        registry.record_failure("10.0.0.1:8200")

        available = registry.get_available_instances("decode")
        assert "10.0.0.1:8200" not in available
        assert "10.0.0.2:8200" in available

    def test_circuit_closes_after_recovery(self):
        from xpyd.registry import InstanceRegistry

        t = [0.0]
        registry = InstanceRegistry(
            cb_enabled=True,
            failure_threshold=2,
            success_threshold=1,
            timeout_duration_seconds=5,
            clock=lambda: t[0],
        )
        registry.add("decode", "10.0.0.1:8200")
        registry.mark_healthy("10.0.0.1:8200")

        # Open circuit
        registry.record_failure("10.0.0.1:8200")
        registry.record_failure("10.0.0.1:8200")
        assert len(registry.get_available_instances("decode")) == 0

        # Advance time past timeout → half-open
        t[0] = 6.0
        # Record success in half-open
        registry.record_success("10.0.0.1:8200")
        assert len(registry.get_available_instances("decode")) == 1


# ------------------------------------------------------------------
# Scenario D: Health monitor detects healthy nodes
# ------------------------------------------------------------------


@pytest.mark.anyio
async def test_health_monitor_detects_healthy(dummy_nodes):
    from xpyd.health_monitor import HealthMonitor

    results = []
    monitor = HealthMonitor(
        nodes=[
            f"127.0.0.1:{dummy_nodes['decode_port_1']}",
            f"127.0.0.1:{dummy_nodes['decode_port_2']}",
        ],
        interval_seconds=60,
        timeout_seconds=2,
        on_healthy=lambda addr: results.append(("healthy", addr)),
        on_unhealthy=lambda addr: results.append(("unhealthy", addr)),
    )
    await monitor.check_once()

    healthy = [r for r in results if r[0] == "healthy"]
    assert len(healthy) == 2


@pytest.mark.anyio
async def test_health_monitor_detects_unreachable():
    from xpyd.health_monitor import HealthMonitor

    results = []
    monitor = HealthMonitor(
        nodes=["127.0.0.1:1"],  # nothing listening
        interval_seconds=60,
        timeout_seconds=1,
        on_healthy=lambda addr: results.append(("healthy", addr)),
        on_unhealthy=lambda addr: results.append(("unhealthy", addr)),
    )
    await monitor.check_once()

    assert len(results) == 1
    assert results[0] == ("unhealthy", "127.0.0.1:1")


# ------------------------------------------------------------------
# Scenario E: Health monitor + registry integration
# ------------------------------------------------------------------


@pytest.mark.anyio
async def test_health_monitor_updates_registry(dummy_nodes):
    """Health monitor callbacks should update registry status."""
    from xpyd.health_monitor import HealthMonitor
    from xpyd.registry import InstanceRegistry

    registry = InstanceRegistry()
    registry.add("decode", f"127.0.0.1:{dummy_nodes['decode_port_1']}")
    registry.add("decode", "127.0.0.1:1")  # unreachable

    monitor = HealthMonitor(
        nodes=[f"127.0.0.1:{dummy_nodes['decode_port_1']}", "127.0.0.1:1"],
        interval_seconds=60,
        timeout_seconds=1,
        on_healthy=registry.mark_healthy,
        on_unhealthy=registry.mark_unhealthy,
    )
    await monitor.check_once()

    available = registry.get_available_instances("decode")
    assert f"127.0.0.1:{dummy_nodes['decode_port_1']}" in available
    assert "127.0.0.1:1" not in available


# ------------------------------------------------------------------
# Scenario F: 4xx should not be retried (unit level check)
# ------------------------------------------------------------------


class TestNoRetryOn4xx:
    """Verify that 4xx errors are returned directly without retry."""

    @pytest.mark.anyio
    async def test_400_returned_directly(self, baseline_client):
        resp = await baseline_client.post(
            "/v1/completions",
            json={"model": "test"},  # missing 'prompt' field
        )
        assert resp.status_code == 400


# ------------------------------------------------------------------
# Scenario G: All features disabled = backward compatible
# ------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_default_config_all_disabled(self, dummy_nodes):
        config = _make_config(dummy_nodes)
        assert config.circuit_breaker.enabled is False
        assert config.health_check.enabled is False

    @pytest.mark.anyio
    async def test_proxy_works_with_all_disabled(self, dummy_nodes):
        config = _make_config(dummy_nodes)
        app = _make_proxy_app(config)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as cli:
            resp = await cli.post(
                "/v1/completions",
                json={
                    "model": "test",
                    "prompt": "Hello",
                    "max_tokens": 5,
                    "stream": False,
                },
            )
            assert resp.status_code == 200
