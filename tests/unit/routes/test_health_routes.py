# SPDX-License-Identifier: Apache-2.0
"""Tests for the health, info and metrics routes."""

# Standard
import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

# Third Party
import aiohttp
import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

# First Party
from xpyd.proxy import Proxy
from xpyd.routes import health, metrics, models, version

from ._helpers import FakeResponse, build_app, fake_aiohttp, make_proxy


def register(router, server):
    for module in (health, models, version, metrics):
        module.register(router, server)


class _HealthServer:
    """Records how the routes call into the proxy fetch helper."""

    def __init__(self, registry: Any = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self.registry = registry

    async def get_from_instance(self, path: str, is_full_instancelist: int = 0):
        self.calls.append((path, is_full_instancelist))
        return JSONResponse({"path": path})


def _client(server: Any) -> TestClient:
    return TestClient(build_app(register, server))


# --------------------------------------------------------------------------
# Route wiring
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["get", "post"])
def test_ping_queries_every_instance(method: str) -> None:
    """Both ``/ping`` verbs share a handler that fans out to all instances."""
    server = _HealthServer()

    response = getattr(_client(server), method)("/ping")

    assert response.status_code == 200
    assert server.calls == [("/ping", 1)]


def test_health_queries_every_instance() -> None:
    """``/health`` aggregates the whole cluster, not just one node."""
    server = _HealthServer()

    assert _client(server).get("/health").status_code == 200
    assert server.calls == [("/health", 1)]


def test_version_queries_a_single_instance() -> None:
    """``/version`` is identical across nodes, so one probe is enough."""
    server = _HealthServer()

    assert _client(server).get("/version").status_code == 200
    assert server.calls == [("/version", 0)]


def test_models_are_served_from_the_registry() -> None:
    """Multi-model deployments list every registered model."""
    registry = MagicMock()
    registry.get_registered_models.return_value = ["opt-125m", "llama-3"]
    server = _HealthServer(registry=registry)

    payload = _client(server).get("/v1/models").json()

    assert payload["object"] == "list"
    assert [entry["id"] for entry in payload["data"]] == ["opt-125m", "llama-3"]
    assert {entry["created"] for entry in payload["data"]} == {0}
    assert {entry["owned_by"] for entry in payload["data"]} == {"system"}
    assert server.calls == []


def test_models_fall_back_to_a_backend_without_registry() -> None:
    """Without a registry the request is forwarded to a backend instance."""
    server = _HealthServer(registry=None)

    assert _client(server).get("/v1/models").status_code == 200
    assert server.calls == [("/v1/models", 0)]


def test_metrics_uses_the_prometheus_content_type() -> None:
    """Prometheus only scrapes the versioned text exposition format."""
    response = _client(_HealthServer()).get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"] == (
        "text/plain; version=0.0.4; charset=utf-8"
    )


@pytest.mark.parametrize("route", ["/health", "/ping", "/v1/models", "/version"])
def test_options_is_allowed(route: str) -> None:
    """CORS preflight is answered for every informational route."""
    assert _client(_HealthServer()).options(route).status_code == 200


# --------------------------------------------------------------------------
# Proxy.get_from_instance behaviour
# --------------------------------------------------------------------------


def _fetch(proxy: Any, path: str, full: int, handler: Any) -> tuple[Any, Any]:
    with fake_aiohttp(handler) as session:
        response = asyncio.run(proxy.get_from_instance(path, is_full_instancelist=full))
    return response, session


def test_aggregated_instances_are_probed() -> None:
    """Health aggregation queries every configured aggregated-role instance."""
    proxy = make_proxy(aggregated={"llama": ["127.0.0.1:8000", "127.0.0.1:8001"]})

    response, session = _fetch(
        proxy, "/health", 1, lambda *_a: FakeResponse(200, {"status": "ok"})
    )

    assert response.status_code == 200
    assert session.urls == [
        "http://127.0.0.1:8000/health",
        "http://127.0.0.1:8001/health",
    ]


def test_single_instance_mode_probes_only_the_first_node() -> None:
    """``is_full_instancelist=0`` avoids fanning out for per-node data."""
    proxy = make_proxy(prefill=["127.0.0.1:8100"], decode=["127.0.0.1:8200"])

    _, session = _fetch(
        proxy, "/version", 0, lambda *_a: FakeResponse(200, {"version": "1"})
    )

    assert session.urls == ["http://127.0.0.1:8100/version"]


def test_single_instance_mode_skips_a_known_dead_node() -> None:
    """A downed first node must not make ``/version`` answer 503."""
    registry = SimpleNamespace(
        get_available_instances=lambda role, model="": (
            ["127.0.0.1:8200"] if role == "decode" else []
        )
    )
    proxy = make_proxy(
        prefill=["127.0.0.1:8100"], decode=["127.0.0.1:8200"], registry=registry
    )

    response, session = _fetch(
        proxy, "/version", 0, lambda *_a: FakeResponse(200, {"version": "1"})
    )

    assert session.urls == ["http://127.0.0.1:8200/version"]
    assert response.status_code == 200


def test_single_instance_mode_falls_back_when_nothing_is_healthy() -> None:
    """With every node down the probe still reports the failure honestly."""
    registry = SimpleNamespace(get_available_instances=lambda role, model="": [])
    proxy = make_proxy(prefill=["127.0.0.1:8100"], registry=registry)

    response, session = _fetch(
        proxy, "/version", 0, lambda *_a: FakeResponse(500, {"error": "down"})
    )

    assert session.urls == ["http://127.0.0.1:8100/version"]
    assert response.status_code == 503


def test_empty_cluster_reports_an_error() -> None:
    """With nothing registered there is no health to report."""
    proxy = make_proxy()

    response, session = _fetch(proxy, "/health", 1, lambda *_a: None)

    assert response.status_code == 500
    assert session.calls == []


def test_all_backends_unreachable_reports_service_unavailable() -> None:
    """A fully down cluster must not answer 200 to a load balancer probe."""
    proxy = make_proxy(prefill=["127.0.0.1:8100"], decode=["127.0.0.1:8200"])

    response, _ = _fetch(
        proxy,
        "/health",
        1,
        lambda *_a: aiohttp.ClientConnectorError(MagicMock(), OSError("refused")),
    )

    assert response.status_code == 503
    body = json.loads(response.body)
    assert set(body) == {"127.0.0.1:8100", "127.0.0.1:8200"}
    assert all(entry["status"] == 500 for entry in body.values())


def test_partially_degraded_cluster_stays_available() -> None:
    """One healthy node is enough to keep serving traffic."""
    proxy = make_proxy(prefill=["127.0.0.1:8100"], decode=["127.0.0.1:8200"])

    def handler(_method: str, url: str, _payload: Any) -> Any:
        if url.startswith("http://127.0.0.1:8100"):
            return FakeResponse(200, {"status": "ok"})
        return aiohttp.ClientConnectorError(MagicMock(), OSError("refused"))

    response, _ = _fetch(proxy, "/health", 1, handler)

    assert response.status_code == 200
    body = json.loads(response.body)
    assert body["127.0.0.1:8100"]["status"] == 200
    assert body["127.0.0.1:8200"]["error"] == "Failed to connect to instance"


def test_backend_server_errors_are_treated_as_unavailable() -> None:
    """A cluster answering only 5xx is not healthy either."""
    proxy = make_proxy(prefill=["127.0.0.1:8100"])

    response, _ = _fetch(
        proxy, "/health", 1, lambda *_a: FakeResponse(500, {"status": "bad"})
    )

    assert response.status_code == 503


def test_non_json_backend_payload_is_returned_as_text() -> None:
    """vLLM answers ``/health`` with an empty non-JSON body."""
    proxy = make_proxy(prefill=["127.0.0.1:8100"])

    response, _ = _fetch(proxy, "/health", 1, lambda *_a: FakeResponse(200, None, "OK"))

    assert response.status_code == 200
    body = json.loads(response.body)
    assert body["127.0.0.1:8100"] == {"status": 200, "type": "text", "data": "OK"}


def test_get_from_instance_is_bound_to_the_proxy_class() -> None:
    """Guard against the helper drifting away from the real implementation."""
    assert make_proxy().get_from_instance.__func__ is Proxy.get_from_instance
