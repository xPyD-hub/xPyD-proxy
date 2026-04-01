"""Tests for the background Health Monitor."""

from __future__ import annotations

import socket
import textwrap
import threading
import time

import pytest
import uvicorn
from config import HealthCheckConfig, ProxyConfig
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from health_monitor import HealthMonitor


def _free_port():
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_health_server(port: int, healthy: bool = True):
    """Start a tiny HTTP server that responds on /health."""
    app = FastAPI()

    @app.get("/health")
    async def health():
        if healthy:
            return PlainTextResponse("ok")
        return PlainTextResponse("down", status_code=503)

    def _run():
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        uvicorn.Server(config).run()

    threading.Thread(target=_run, daemon=True).start()
    time.sleep(1)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_detects_healthy_node():
    """Monitor should call on_healthy for a responding node."""
    port = _free_port()
    _start_health_server(port, healthy=True)

    results = []
    monitor = HealthMonitor(
        nodes=[f"127.0.0.1:{port}"],
        interval_seconds=60,
        timeout_seconds=2,
        on_healthy=lambda addr: results.append(("healthy", addr)),
        on_unhealthy=lambda addr: results.append(("unhealthy", addr)),
    )
    await monitor.check_once()

    assert len(results) == 1
    assert results[0] == ("healthy", f"127.0.0.1:{port}")


@pytest.mark.anyio
async def test_detects_unhealthy_node():
    """Monitor should call on_unhealthy for unreachable node."""
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


@pytest.mark.anyio
async def test_detects_503_as_unhealthy():
    """Monitor should call on_unhealthy when /health returns 503."""
    port = _free_port()
    _start_health_server(port, healthy=False)

    results = []
    monitor = HealthMonitor(
        nodes=[f"127.0.0.1:{port}"],
        interval_seconds=60,
        timeout_seconds=2,
        on_healthy=lambda addr: results.append(("healthy", addr)),
        on_unhealthy=lambda addr: results.append(("unhealthy", addr)),
    )
    await monitor.check_once()

    assert len(results) == 1
    assert results[0] == ("unhealthy", f"127.0.0.1:{port}")


@pytest.mark.anyio
async def test_multiple_nodes_mixed():
    """Monitor should handle a mix of healthy and unhealthy nodes."""
    port_ok = _free_port()
    _start_health_server(port_ok, healthy=True)

    results = []
    monitor = HealthMonitor(
        nodes=[f"127.0.0.1:{port_ok}", "127.0.0.1:1"],
        interval_seconds=60,
        timeout_seconds=1,
        on_healthy=lambda addr: results.append(("healthy", addr)),
        on_unhealthy=lambda addr: results.append(("unhealthy", addr)),
    )
    await monitor.check_once()

    assert len(results) == 2
    healthy = [r for r in results if r[0] == "healthy"]
    unhealthy = [r for r in results if r[0] == "unhealthy"]
    assert len(healthy) == 1
    assert len(unhealthy) == 1


@pytest.mark.anyio
async def test_start_and_stop():
    """Monitor start/stop should not raise."""
    monitor = HealthMonitor(
        nodes=["127.0.0.1:1"],
        interval_seconds=60,
        timeout_seconds=1,
    )
    await monitor.start()
    await monitor.stop()


# ------------------------------------------------------------------
# Config integration
# ------------------------------------------------------------------


class TestHealthCheckConfig:
    def test_defaults(self):
        cfg = HealthCheckConfig()
        assert cfg.enabled is False
        assert cfg.interval_seconds == 10.0
        assert cfg.timeout_seconds == 3.0

    def test_custom(self):
        cfg = HealthCheckConfig(enabled=True, interval_seconds=5, timeout_seconds=2)
        assert cfg.enabled is True

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError):
            HealthCheckConfig(enabled=True, bad_key=1)

    def test_yaml_integration(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text(
            textwrap.dedent(
                """\
            model: /m
            decode:
              - "10.0.0.1:8000"
            health_check:
              enabled: true
              interval_seconds: 5
              timeout_seconds: 2
            """
            )
        )
        import argparse

        args = argparse.Namespace(
            config=str(p),
            model=None,
            prefill=None,
            decode=None,
            port=8000,
            generator_on_p_node=False,
            roundrobin=False,
            log_level="warning",
            wait_timeout_seconds=600,
            probe_interval_seconds=10,
        )
        cfg = ProxyConfig.from_args(args)
        assert cfg.health_check.enabled is True
        assert cfg.health_check.interval_seconds == 5
        assert cfg.health_check.timeout_seconds == 2

    def test_yaml_default(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text(
            textwrap.dedent(
                """\
            model: /m
            decode:
              - "10.0.0.1:8000"
            """
            )
        )
        import argparse

        args = argparse.Namespace(
            config=str(p),
            model=None,
            prefill=None,
            decode=None,
            port=8000,
            generator_on_p_node=False,
            roundrobin=False,
            log_level="warning",
            wait_timeout_seconds=600,
            probe_interval_seconds=10,
        )
        cfg = ProxyConfig.from_args(args)
        assert cfg.health_check.enabled is False
