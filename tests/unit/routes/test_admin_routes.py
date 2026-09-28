# SPDX-License-Identifier: Apache-2.0
"""Tests for the admin routes (``/status`` and ``/instances/add``)."""

# Standard
from typing import Any
from unittest.mock import AsyncMock

# Third Party
import pytest
from fastapi.testclient import TestClient

# First Party
from xpyd.routes.admin import register

from ._helpers import build_app

API_KEY = "s3cr3t"


class _AdminServer:
    """Minimal proxy stand-in exposing the state the admin routes mutate."""

    def __init__(
        self,
        prefill: list[str] | None = None,
        decode: list[str] | None = None,
        valid: bool = True,
    ) -> None:
        self.prefill_instances = list(prefill or [])
        self.decode_instances = list(decode or [])
        self.validate_instance = AsyncMock(return_value=valid)
        self.drain_and_remove_instance = AsyncMock()

        async def _add_instance(role: str, address: str) -> bool:
            if not valid:
                return False
            instances = getattr(self, f"{role}_instances")
            if address in instances:
                raise ValueError("Instance already exists")
            instances.append(address)
            return True

        self.add_instance = AsyncMock(side_effect=_add_instance)


@pytest.fixture(name="admin_key")
def _admin_key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("ADMIN_API_KEY", API_KEY)
    return API_KEY


def _client(server: Any) -> TestClient:
    return TestClient(build_app(register, server))


def _add(client: TestClient, payload: dict, key: str | None = API_KEY):
    headers = {} if key is None else {"x-api-key": key}
    return client.post("/instances/add", json=payload, headers=headers)


def _remove(client: TestClient, payload: dict, key: str | None = API_KEY):
    headers = {} if key is None else {"x-api-key": key}
    return client.post("/instances/remove", json=payload, headers=headers)


def test_missing_admin_key_env_is_a_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unconfigured ADMIN_API_KEY must not silently accept requests."""
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    server = _AdminServer()

    response = _add(_client(server), {"type": "prefill", "instance": "127.0.0.1:8100"})

    assert response.status_code == 500
    assert server.add_instance.await_count == 0


def test_wrong_api_key_is_forbidden(admin_key: str) -> None:
    """A mismatching key is rejected before any state is touched."""
    server = _AdminServer()

    response = _add(
        _client(server),
        {"type": "prefill", "instance": "127.0.0.1:8100"},
        key="wrong",
    )

    assert response.status_code == 403
    assert server.prefill_instances == []


def test_absent_api_key_header_is_rejected(admin_key: str) -> None:
    """FastAPI enforces the required header before the handler runs."""
    server = _AdminServer()

    response = _add(
        _client(server), {"type": "prefill", "instance": "127.0.0.1:8100"}, key=None
    )

    assert response.status_code == 422


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_instance_is_added_to_role_membership(admin_key: str, role: str) -> None:
    """The admin route delegates membership changes to the proxy."""
    server = _AdminServer(prefill=["127.0.0.1:8100"], decode=["127.0.0.1:8200"])
    new_instance = "127.0.0.2:8300"

    response = _add(_client(server), {"type": role, "instance": new_instance})

    assert response.status_code == 200
    instances = getattr(server, f"{role}_instances")
    assert new_instance in instances
    server.add_instance.assert_awaited_once_with(role, new_instance)


@pytest.mark.parametrize("role", ["both", "", None])
def test_invalid_instance_type_is_rejected(admin_key: str, role: Any) -> None:
    """Only the prefill and decode roles may be registered at runtime."""
    server = _AdminServer()

    response = _add(_client(server), {"type": role, "instance": "127.0.0.1:8100"})

    assert response.status_code == 400
    assert "Invalid instance type" in response.json()["error"]["message"]


@pytest.mark.parametrize("instance", ["127.0.0.1", "", None])
def test_instance_without_port_is_rejected(admin_key: str, instance: Any) -> None:
    """An address must carry an explicit port."""
    server = _AdminServer()

    response = _add(_client(server), {"type": "prefill", "instance": instance})

    assert response.status_code == 400
    assert "Invalid instance format" in response.json()["error"]["message"]


def test_ipv6_address_is_rejected_with_400(admin_key: str) -> None:
    """IPv6 is unsupported, but must fail like ProxyConfig does, not with 500."""
    server = _AdminServer()

    response = _add(_client(server), {"type": "prefill", "instance": "::1:8100"})

    assert response.status_code == 400
    assert "Invalid instance format" in response.json()["error"]["message"]


@pytest.mark.parametrize("port", ["0", "65536", "abc", "-1"])
def test_out_of_range_port_is_rejected(admin_key: str, port: str) -> None:
    """Ports outside 1-65535 and non-numeric ports are refused."""
    server = _AdminServer()

    response = _add(
        _client(server), {"type": "prefill", "instance": f"127.0.0.1:{port}"}
    )

    assert response.status_code == 400
    assert server.prefill_instances == []


def test_non_ip_host_is_rejected(admin_key: str) -> None:
    """Hostnames other than localhost are not resolvable addresses here."""
    server = _AdminServer()

    response = _add(_client(server), {"type": "prefill", "instance": "not-an-ip:8100"})

    assert response.status_code == 400
    assert "Invalid instance address" in response.json()["error"]["message"]


def test_localhost_is_accepted(admin_key: str) -> None:
    """``localhost`` is explicitly allowed alongside literal IP addresses."""
    server = _AdminServer()

    response = _add(_client(server), {"type": "prefill", "instance": "localhost:8100"})

    assert response.status_code == 200
    assert server.prefill_instances == ["localhost:8100"]


def test_unvalidatable_instance_is_rejected(admin_key: str) -> None:
    """An instance that fails the model handshake is not registered."""
    server = _AdminServer(valid=False)

    response = _add(_client(server), {"type": "prefill", "instance": "127.0.0.1:8100"})

    assert response.status_code == 400
    assert "Instance validation failed" in response.json()["error"]["message"]
    assert server.prefill_instances == []


def test_duplicate_instance_is_rejected(admin_key: str) -> None:
    """Re-adding a known instance is an error, not a silent no-op."""
    server = _AdminServer(prefill=["127.0.0.1:8100"])

    response = _add(_client(server), {"type": "prefill", "instance": "127.0.0.1:8100"})

    assert response.status_code == 400
    assert "Instance already exists" in response.json()["error"]["message"]
    assert server.prefill_instances == ["127.0.0.1:8100"]


def test_unexpected_failure_is_reported_as_server_error(admin_key: str) -> None:
    """An exception inside the handler yields a structured 500."""
    server = _AdminServer()
    server.add_instance = AsyncMock(side_effect=RuntimeError("boom"))

    response = _add(_client(server), {"type": "prefill", "instance": "127.0.0.1:8100"})

    assert response.status_code == 500
    assert "Internal error" in response.json()["error"]["message"]


def test_status_reports_counts_and_members() -> None:
    """``/status`` exposes both the counts and the addresses per role."""
    server = _AdminServer(
        prefill=["127.0.0.1:8100", "127.0.0.2:8100"], decode=["127.0.0.1:8200"]
    )

    payload = _client(server).get("/status").json()

    assert payload == {
        "prefill_node_count": 2,
        "decode_node_count": 1,
        "prefill_nodes": ["127.0.0.1:8100", "127.0.0.2:8100"],
        "decode_nodes": ["127.0.0.1:8200"],
    }


def test_status_options_is_allowed() -> None:
    """CORS preflight is answered for ``/status``."""
    assert _client(_AdminServer()).options("/status").status_code == 200


@pytest.mark.parametrize("route", ["/instances/add", "/instances/remove"])
def test_instance_admin_options_are_allowed(route: str) -> None:
    assert _client(_AdminServer()).options(route).status_code == 200


def test_remove_drains_instance_before_success(admin_key: str) -> None:
    server = _AdminServer(decode=["127.0.0.1:8200"])

    response = _remove(
        _client(server),
        {
            "type": "decode",
            "instance": "127.0.0.1:8200",
            "timeout_seconds": 12.5,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "message": "Removed 127.0.0.1:8200 from decode_instances."
    }
    server.drain_and_remove_instance.assert_awaited_once_with(
        "decode", "127.0.0.1:8200", 12.5
    )


def test_remove_missing_instance_is_not_found(admin_key: str) -> None:
    server = _AdminServer()
    server.drain_and_remove_instance.side_effect = KeyError("missing")

    response = _remove(
        _client(server),
        {"type": "decode", "instance": "127.0.0.1:8200"},
    )

    assert response.status_code == 404


def test_remove_wrong_role_is_invalid(admin_key: str) -> None:
    server = _AdminServer()
    server.drain_and_remove_instance.side_effect = ValueError("wrong role")

    response = _remove(
        _client(server),
        {"type": "prefill", "instance": "127.0.0.1:8200"},
    )

    assert response.status_code == 400
    assert "wrong role" in response.json()["error"]["message"]


def test_remove_timeout_is_reported(admin_key: str) -> None:
    server = _AdminServer()
    server.drain_and_remove_instance.side_effect = TimeoutError(
        "Timed out draining node"
    )

    response = _remove(
        _client(server),
        {"type": "decode", "instance": "127.0.0.1:8200"},
    )

    assert response.status_code == 504
    assert "Timed out draining node" in response.json()["error"]["message"]


@pytest.mark.parametrize("timeout", [-1, 3601, "60", True])
def test_remove_rejects_invalid_timeout(admin_key: str, timeout: Any) -> None:
    server = _AdminServer()

    response = _remove(
        _client(server),
        {
            "type": "decode",
            "instance": "127.0.0.1:8200",
            "timeout_seconds": timeout,
        },
    )

    assert response.status_code == 400
    server.drain_and_remove_instance.assert_not_awaited()


def test_request_body_is_not_logged_at_warning(
    admin_key: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Admin payloads must not be echoed into operational warning logs."""
    server = _AdminServer()

    with caplog.at_level("WARNING", logger="xpyd.proxy"):
        _add(_client(server), {"type": "prefill", "instance": "127.0.0.1:8100"})

    assert "127.0.0.1:8100" not in caplog.text
