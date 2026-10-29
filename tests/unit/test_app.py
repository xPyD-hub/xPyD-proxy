# SPDX-License-Identifier: Apache-2.0
"""Application assembly, dependency boundaries and proxy-first lifecycle."""

import json
import subprocess
import sys
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from xpyd.app import create_app
from xpyd.config import ProxyConfig
from xpyd.proxy import ProxyServer

from .routes._helpers import FakeResponse, fake_aiohttp


def test_support_modules_do_not_import_proxy_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import importlib, inspect, pkgutil, sys, typing",
                    "for name in ('xpyd.cli', 'xpyd.app', 'xpyd.routes',",
                    "             'xpyd.completion_handler', 'xpyd.interfaces'):",
                    "    module = importlib.import_module(name)",
                    "    assert 'xpyd.proxy' not in sys.modules, name",
                    "    modules = [module]",
                    "    if hasattr(module, '__path__'):",
                    "        modules += [importlib.import_module(item.name)",
                    "                    for item in pkgutil.iter_modules(",
                    "                        module.__path__, module.__name__ + '.')]",
                    "    for item in modules:",
                    "        for value in vars(item).values():",
                    "            if (inspect.isfunction(value)",
                    "                    and value.__module__ == item.__name__):",
                    "                typing.get_type_hints(value)",
                    "                parameters = inspect.signature(value).parameters",
                    "                assert not {'proxy', 'server'} & set(parameters)",
                    "    assert 'xpyd.proxy' not in sys.modules, name",
                ]
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("topology", ["disaggregated", "aggregated", "mixed"])
def test_proxy_first_discovery_loss_and_reconnection(topology):
    roles = []
    if topology != "aggregated":
        roles.extend(["prefill", "decode", "decode"])
    if topology != "disaggregated":
        roles.extend(["aggregated", "aggregated"])
    addresses = [f"127.0.0.1:{18100 + index}" for index in range(len(roles))]
    model_by_address = {
        address: "agg-model" if role == "aggregated" else "pd-model"
        for role, address in zip(roles, addresses)
    }
    config = ProxyConfig(
        instances=[
            {
                "role": role,
                "address": address,
                "model": model_by_address[address] if topology == "mixed" else "",
            }
            for role, address in zip(roles, addresses)
        ],
        scheduling="roundrobin",
        probe_interval_seconds=3600,
        heartbeat_interval_seconds=0.001,
        health_check={"enabled": True, "interval_seconds": 3600},
    )
    server = ProxyServer(config)
    proxy = server.proxy_instance
    proxy.ensure_tokenizer = Mock(return_value=True)
    forwarded = []

    async def forward(url, data, **kwargs):
        address = urlsplit(url).netloc
        assert address in online
        assert data["model"] == model_by_address[address]
        forwarded.append(address)
        yield json.dumps({"choices": [{"text": "ok"}]}).encode()

    proxy.forward_request = forward
    app = server.create_app()
    discovery = server.discovery
    monitor = server.health_monitor
    assert discovery._task is None
    assert monitor._task is None
    online = set()

    def backend(_method, url, _payload):
        target = urlsplit(url)
        if target.netloc not in online:
            raise ConnectionRefusedError(target.netloc)
        if target.path == "/health":
            return FakeResponse(200, {"status": "ok"})
        assert target.path == "/v1/models"
        return FakeResponse(200, {"data": [{"id": model_by_address[target.netloc]}]})

    with fake_aiohttp(backend) as session, TestClient(app) as client:
        assert not discovery._task.done()
        assert not monitor._task.done()

        async def probe():
            await discovery._probe_all(session)
            await monitor.check_once()

        def check_status():
            response = client.get("/status/instances")
            assert response.status_code == 200
            body = response.json()
            assert set(body) == {
                "prefill_instances",
                "decode_instances",
                "aggregated_instances",
            }
            for role in ("prefill", "decode", "aggregated"):
                assert len(body[f"{role}_instances"]) == roles.count(role)
            entries = [entry for instances in body.values() for entry in instances]
            assert {entry["address"] for entry in entries} == set(addresses)
            assert {
                entry["address"] for entry in entries if entry["status"] == "healthy"
            } == online
            assert all(entry["active_requests"] == 0 for entry in entries)

        def complete(expected_status):
            for model in set(model_by_address.values()):
                for path, prompt in (
                    ("/v1/completions", {"prompt": "hello"}),
                    (
                        "/v1/chat/completions",
                        {"messages": [{"role": "user", "content": "hello"}]},
                    ),
                ):
                    response = client.post(path, json={"model": model, **prompt})
                    assert response.status_code == expected_status, response.text

        client.portal.call(probe)
        complete(503)
        assert forwarded == []
        check_status()
        assert client.get("/status").status_code == 200
        assert client.get("/metrics").status_code == 200
        assert set(client.get("/health").json()) == set(addresses)
        assert set(client.get("/ping").json()) == set(addresses)
        assert client.post("/instances/add", json={}).status_code == 422
        assert client.post("/instances/remove", json={}).status_code == 422

        online.update(addresses)
        client.portal.call(probe)
        assert discovery.is_ready
        assert {call.args[0] for call in proxy.ensure_tokenizer.call_args_list} == set(
            model_by_address.values()
        )
        assert {
            entry["id"] for entry in client.get("/v1/models").json()["data"]
        } == set(model_by_address.values())
        complete(200)
        check_status()
        preflight = client.options(
            "/v1/completions",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "*"

        for role in ("decode", "aggregated"):
            candidates = [
                address
                for address, configured_role in zip(addresses, roles)
                if configured_role == role
            ]
            if candidates:
                online.remove(candidates[0])
        client.portal.call(probe)
        assert discovery.is_ready
        complete(200)
        check_status()

        with patch("xpyd.discovery.logger.info") as heartbeat:
            discovery._last_heartbeat = None
            client.portal.call(discovery._log_heartbeat_if_due)
        fields = []
        if topology != "aggregated":
            fields.extend(["P=1/1 online", "D=1/2 online"])
        if topology != "disaggregated":
            fields.append("aggregated=1/2 online")
        assert heartbeat.call_args.args == (
            "Node heartbeat | mode=%s | %s",
            topology,
            " | ".join(fields),
        )

        online.clear()
        client.portal.call(probe)
        assert not discovery.is_ready
        complete(503)
        check_status()

        online.update(addresses)
        client.portal.call(probe)
        assert discovery.is_ready
        complete(200)
        check_status()

    assert discovery._task.done()
    assert monitor._task.done()


def test_zmq_lifecycle_order():
    config = ProxyConfig(
        model="example",
        prefill=["127.0.0.1:18100"],
        decode=["127.0.0.1:18200"],
        scheduling="roundrobin",
        disaggregated_mode="zmq",
        zmq={
            "receivers": {
                "127.0.0.1:18200": {
                    "host": "127.0.0.1",
                    "init_ports": [7300],
                    "alloc_ports": [7400],
                }
            }
        },
        health_check={"enabled": True},
    )
    server = ProxyServer(config)
    events = []
    listener = Mock(
        start=AsyncMock(side_effect=lambda: events.append("listener:start")),
        stop=AsyncMock(side_effect=lambda: events.append("listener:stop")),
    )
    with (
        patch.object(
            server.discovery,
            "start",
            AsyncMock(side_effect=lambda: events.append("discovery:start")),
        ),
        patch.object(
            server.discovery,
            "stop",
            AsyncMock(side_effect=lambda: events.append("discovery:stop")),
        ),
        patch.object(
            server.health_monitor,
            "start",
            AsyncMock(side_effect=lambda: events.append("health:start")),
        ),
        patch.object(
            server.health_monitor,
            "stop",
            AsyncMock(side_effect=lambda: events.append("health:stop")),
        ),
    ):
        app = create_app(
            server.proxy_instance.router,
            discovery=server.discovery,
            health_monitor=server.health_monitor,
            zmq_notifications=listener,
        )
        assert events == []
        with TestClient(app):
            assert events == ["listener:start", "discovery:start", "health:start"]
    assert events == [
        "listener:start",
        "discovery:start",
        "health:start",
        "discovery:stop",
        "listener:stop",
        "health:stop",
    ]


def test_zmq_route_uses_composed_listener():
    config = ProxyConfig(
        model="example",
        prefill=["127.0.0.1:18100"],
        decode=["127.0.0.1:18200"],
        scheduling="roundrobin",
        disaggregated_mode="zmq",
        zmq={
            "receivers": {
                "127.0.0.1:18200": {
                    "host": "127.0.0.1",
                    "init_ports": [7300],
                    "alloc_ports": [7400],
                }
            }
        },
    )
    server = ProxyServer(config)
    server.discovery.healthy_prefill.update(config.prefill)
    server.discovery.healthy_decode.update(config.decode)
    listener = server.zmq_notifications
    forwarded = []

    async def forward(url, data, **kwargs):
        forwarded.append((url, data))
        yield b'{"choices":[{"text":"ok"}]}'

    server.proxy_instance.forward_request = forward
    server.proxy_instance.tokenize_on_backend = AsyncMock(return_value=[1, 2])
    with (
        patch.object(server.discovery, "start", AsyncMock()),
        patch.object(server.discovery, "stop", AsyncMock()),
        patch.object(listener, "start", AsyncMock()) as start,
        patch.object(listener, "stop", AsyncMock()) as stop,
        patch.object(listener, "register", AsyncMock()) as register,
        patch.object(listener, "wait", AsyncMock()) as wait,
    ):
        with TestClient(server.create_app()) as client:
            response = client.post(
                "/v1/completions",
                json={"model": "example", "prompt": "hello", "max_tokens": 2},
            )
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["text"] == "ok"
        start.assert_awaited_once()
        stop.assert_awaited_once()
        request_id = forwarded[0][1]["kv_transfer_params"]["disagg_spec"]["req_id"]
        register.assert_awaited_once_with(request_id)
        wait.assert_awaited_once_with(request_id, 1)
    assert [url for url, _ in forwarded] == [
        "http://127.0.0.1:18100/v1/completions",
        "http://127.0.0.1:18200/v1/completions",
    ]
    assert all(
        server.registry.get_active_requests(address) == 0
        for address in config.prefill + config.decode
    )
