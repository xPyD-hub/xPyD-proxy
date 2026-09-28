"""Regression checks for the standalone Prometheus example."""

import runpy
import socket
from pathlib import Path
from unittest.mock import patch

import pytest

DEMO = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "examples/monitoring/prometheus/run.py")
)


@pytest.mark.parametrize("host", [None, "127.0.0.1"])
def test_proxy_listener_uses_configured_host(host):
    from xpyd.config import ProxyConfig
    from xpyd.proxy import ProxyServer

    path = (
        Path(__file__).resolve().parents[2] / "examples/monitoring/prometheus/xpyd.yaml"
    )
    config = ProxyConfig.from_yaml(path)
    assert config.host == "127.0.0.1"
    if host is None:
        config = ProxyConfig(**config.model_dump(exclude={"host"}))
    with patch("xpyd.proxy.uvicorn.Server") as server:
        ProxyServer(config).run_server()
    listener = server.call_args.args[0]
    assert listener.host == (host or "0.0.0.0")
    assert listener.port == 18868


def timing_samples(prefill, ttft, transfer, count=1):
    values = {}
    labels = frozenset({("model", "demo")})
    for name, value in (
        ("proxy_prefill_duration_seconds", prefill),
        ("proxy_ttft_seconds", ttft),
        ("proxy_kv_transfer_duration_seconds", transfer),
    ):
        values[(name + "_sum", labels)] = value
        values[(name + "_count", labels)] = count
    return values


def test_pd_timing_uses_same_request_deltas():
    before = timing_samples(10, 30, 20, count=100)
    after = timing_samples(10.2, 30.5, 20.3, count=101)
    result = DEMO["assert_pd_timing"](before, after, "demo")
    assert result["proxy_kv_transfer_duration_seconds"] == pytest.approx(0.3)


@pytest.mark.parametrize(
    "after",
    [
        timing_samples(0.2, 0.2, 0.3),
        timing_samples(0.2, 0.5, 0.1),
        timing_samples(0.2, 0.5, 0.3, count=2),
        timing_samples(0.2, 0.5, -0.3),
    ],
)
def test_pd_timing_rejects_wrong_semantics(after):
    with pytest.raises(AssertionError):
        DEMO["assert_pd_timing"]({}, after, "demo")


def test_gauges_cannot_cancel_each_other():
    values = {
        ("proxy_active_requests", frozenset()): 0,
        ("proxy_decode_active_requests", frozenset({("model", "a")})): 1,
        ("proxy_decode_active_requests", frozenset({("model", "b")})): -1,
    }
    with pytest.raises(AssertionError):
        DEMO["assert_idle"](values)


def test_missing_active_gauge_is_not_idle():
    with pytest.raises(AssertionError):
        DEMO["assert_idle"]({})


@pytest.mark.parametrize("value", ["NaN", "+Inf", "-Inf"])
def test_nonfinite_samples_fail(value):
    with pytest.raises(AssertionError):
        DEMO["samples"](f"# TYPE demo gauge\ndemo {value}\n")


def test_listening_port_is_not_free():
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen()
        assert not DEMO["port_free"](server.getsockname()[1])


def test_closed_connection_does_not_block_restarting_server():
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.listen()
        with socket.create_connection(("127.0.0.1", port)) as client:
            connection, _ = server.accept()
            connection.close()
            assert client.recv(1) == b""
    assert DEMO["port_free"](port)
