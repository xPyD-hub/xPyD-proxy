"""Tests for Task 8: CLI packaging, --validate-config, startup discovery."""

from __future__ import annotations

import argparse
import os
import socket
import textwrap
import threading
import time
from unittest.mock import patch

import pytest
import uvicorn
from config import ProxyConfig
from discovery import DiscoveryTimeout, NodeDiscovery
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from httpx import ASGITransport, AsyncClient
from MicroPDProxyServer import _build_parser, _resolve_config_path

# ------------------------------------------------------------------
# CLI argument parsing
# ------------------------------------------------------------------


class TestCLIParsing:
    def test_version_flag(self, capsys):
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_config_arg(self):
        parser = _build_parser()
        args = parser.parse_args(["--config", "test.yaml"])
        assert args.config == "test.yaml"

    def test_validate_config_arg(self):
        parser = _build_parser()
        args = parser.parse_args(["--validate-config", "test.yaml"])
        assert args.validate_config == "test.yaml"

    def test_log_level_arg(self):
        parser = _build_parser()
        args = parser.parse_args(["--log-level", "debug"])
        assert args.log_level == "debug"

    def test_all_args_together(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--config",
                "c.yaml",
                "--model",
                "/m",
                "--prefill",
                "10.0.0.1:8000",
                "--decode",
                "10.0.0.2:8000",
                "--port",
                "9000",
                "--roundrobin",
                "--log-level",
                "info",
            ]
        )
        assert args.config == "c.yaml"
        assert args.model == "/m"
        assert args.port == 9000
        assert args.roundrobin is True
        assert args.log_level == "info"


# ------------------------------------------------------------------
# Config resolution: --config > XPYD_CONFIG > ./xpyd.yaml
# ------------------------------------------------------------------


class TestConfigResolution:
    def test_cli_config_wins(self):
        args = argparse.Namespace(config="cli.yaml")
        assert _resolve_config_path(args) == "cli.yaml"

    def test_env_var_fallback(self):
        args = argparse.Namespace(config=None)
        with patch.dict(os.environ, {"XPYD_CONFIG": "env.yaml"}):
            assert _resolve_config_path(args) == "env.yaml"

    def test_default_file_fallback(self, tmp_path, monkeypatch):
        # Create xpyd.yaml in a temp dir and chdir there
        (tmp_path / "xpyd.yaml").write_text("model: test\n")
        monkeypatch.chdir(tmp_path)
        args = argparse.Namespace(config=None)
        env = {k: v for k, v in os.environ.items() if k != "XPYD_CONFIG"}
        with patch.dict(os.environ, env, clear=True):
            result = _resolve_config_path(args)
        assert result is not None
        assert result.endswith("xpyd.yaml")

    def test_no_config_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # no xpyd.yaml here
        args = argparse.Namespace(config=None)
        env = {k: v for k, v in os.environ.items() if k != "XPYD_CONFIG"}
        with patch.dict(os.environ, env, clear=True):
            assert _resolve_config_path(args) is None


# ------------------------------------------------------------------
# --validate-config
# ------------------------------------------------------------------


class TestValidateConfig:
    def test_valid_config(self, tmp_path):
        p = tmp_path / "valid.yaml"
        p.write_text(
            textwrap.dedent(
                """\
            model: /path/model
            decode:
              - "10.0.0.1:8000"
        """
            )
        )
        parser = _build_parser()
        args = parser.parse_args(["--validate-config", str(p)])
        args.config = args.validate_config
        config = ProxyConfig.from_args(args)
        assert config.model == "/path/model"

    def test_invalid_config(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("not_a_field: oops\n")
        parser = _build_parser()
        args = parser.parse_args(["--validate-config", str(p)])
        args.config = args.validate_config
        with pytest.raises(ValueError):
            ProxyConfig.from_args(args)


# ------------------------------------------------------------------
# Startup config in YAML
# ------------------------------------------------------------------


class TestStartupConfig:
    def test_startup_section(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text(
            textwrap.dedent(
                """\
            model: /m
            decode:
              - "10.0.0.1:8000"
            startup:
              wait_timeout_seconds: 120
              probe_interval_seconds: 5
        """
            )
        )
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
        assert cfg.wait_timeout_seconds == 120
        assert cfg.probe_interval_seconds == 5

    def test_startup_defaults(self):
        args = argparse.Namespace(
            config=None,
            model="/m",
            prefill=None,
            decode=["10.0.0.1:8000"],
            port=8000,
            generator_on_p_node=False,
            roundrobin=False,
            log_level="warning",
            wait_timeout_seconds=600,
            probe_interval_seconds=10,
        )
        cfg = ProxyConfig.from_args(args)
        assert cfg.wait_timeout_seconds == 600
        assert cfg.probe_interval_seconds == 10

    def test_startup_unknown_keys(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text(
            textwrap.dedent(
                """\
            model: /m
            decode:
              - "10.0.0.1:8000"
            startup:
              bad_key: 1
        """
            )
        )
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
        with pytest.raises(ValueError, match="Unknown keys in startup"):
            ProxyConfig.from_args(args)


# ------------------------------------------------------------------
# Node discovery
# ------------------------------------------------------------------


def _free_port():
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_health_server(port: int):
    """Start a tiny server that responds 200 on /health."""
    app = FastAPI()

    @app.get("/health")
    async def health():
        return PlainTextResponse("ok")

    def _run():
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        uvicorn.Server(config).run()

    threading.Thread(target=_run, daemon=True).start()
    time.sleep(1)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_discovery_finds_healthy_nodes():
    """Discovery should detect healthy nodes and become ready."""
    p_port = _free_port()
    d_port = _free_port()
    _start_health_server(p_port)
    _start_health_server(d_port)

    disc = NodeDiscovery(
        prefill_instances=[f"127.0.0.1:{p_port}"],
        decode_instances=[f"127.0.0.1:{d_port}"],
        probe_interval=0.5,
        wait_timeout=10,
    )
    await disc.start()
    ready = await disc.wait_until_ready()
    await disc.stop()

    assert ready is True
    assert disc.is_ready
    assert f"127.0.0.1:{p_port}" in disc.healthy_prefill
    assert f"127.0.0.1:{d_port}" in disc.healthy_decode


@pytest.mark.anyio
async def test_discovery_timeout_when_no_nodes():
    """Discovery should raise DiscoveryTimeout when nodes are unreachable."""
    disc = NodeDiscovery(
        prefill_instances=["127.0.0.1:1"],  # nothing listening
        decode_instances=["127.0.0.1:2"],
        probe_interval=0.2,
        wait_timeout=1.0,
    )
    await disc.start()
    # Remove the done callback so sys.exit() doesn't fire during tests
    disc._task.remove_done_callback(disc._on_probe_done)

    ready = await disc.wait_until_ready()
    assert ready is False
    assert not disc.is_ready

    # The probe loop should have raised DiscoveryTimeout
    with pytest.raises(DiscoveryTimeout):
        await disc._task


@pytest.mark.anyio
async def test_503_before_ready():
    """Proxy should return 503 before discovery reports ready."""
    disc = NodeDiscovery(
        prefill_instances=["127.0.0.1:1"],
        decode_instances=["127.0.0.1:2"],
        probe_interval=60,  # never actually probes
        wait_timeout=600,
    )

    app = FastAPI()

    @app.middleware("http")
    async def check_readiness(request, call_next):
        path = request.url.path
        if path in ("/health", "/ping", "/status", "/metrics"):
            return await call_next(request)
        if not disc.is_ready:
            return JSONResponse({"error": "waiting for backend nodes"}, status_code=503)
        return await call_next(request)

    @app.get("/health")
    async def health():
        return PlainTextResponse("ok")

    @app.post("/v1/completions")
    async def completions():
        return JSONResponse({"choices": []})

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # /health should work
        resp = await client.get("/health")
        assert resp.status_code == 200

        # /v1/completions should be 503
        resp = await client.post("/v1/completions", json={})
        assert resp.status_code == 503
        assert "waiting for backend nodes" in resp.json()["error"]
