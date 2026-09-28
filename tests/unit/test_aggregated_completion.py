# SPDX-License-Identifier: Apache-2.0
"""Aggregated routing through real policies and request reservations."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from xpyd.proxy import Proxy
from xpyd.registry import InstanceRegistry
from xpyd.routes.completions import handle_completion
from xpyd.scheduler import SchedulingContext, default_registry

MODEL = "qwen-2"
NODES = ["10.0.0.1:8000", "10.0.0.2:8000"]


def make_proxy(strategy="roundrobin", override=None, registry=True):
    reg = InstanceRegistry() if registry else None
    if reg is not None:
        for address in NODES:
            reg.add("aggregated", address, model=MODEL)
            reg.mark_healthy(address)
    return Proxy(
        [],
        [],
        MODEL,
        default_registry.build(strategy, registry=reg),
        registry=reg,
        aggregated_instances={MODEL: list(NODES)},
        model_schedulers={MODEL: override} if override else {},
    )


def test_model_topology_detection():
    proxy = make_proxy()
    assert proxy._is_aggregated_model(MODEL)
    assert not proxy._is_aggregated_model("unknown")
    proxy.aggregated_instances.clear()
    assert proxy._is_aggregated_model(MODEL)
    proxy.registry = None
    assert not proxy._is_aggregated_model(MODEL)


def test_auto_discovered_aggregated_model():
    proxy = make_proxy()
    proxy.aggregated_instances = {"": list(NODES)}
    lease = proxy.reserve(SchedulingContext(role="aggregated", model=MODEL))
    assert lease.address in NODES
    lease.release()
    assert proxy.reserve(SchedulingContext(role="aggregated", model="unknown")) is None


@pytest.mark.parametrize(
    ("global_strategy", "override"),
    [
        ("roundrobin", None),
        ("loadbalanced", "roundrobin"),
        ("loadbalanced", "round_robin"),
    ],
)
def test_round_robin_dispatch(global_strategy, override):
    proxy = make_proxy(global_strategy, override)
    leases = [
        proxy.reserve(SchedulingContext(role="aggregated", model=MODEL))
        for _ in range(4)
    ]
    assert [lease.address for lease in leases] == NODES * 2
    for lease in leases:
        lease.release()
    assert all(proxy.registry.get_active_requests(address) == 0 for address in NODES)


@pytest.mark.parametrize(
    ("global_strategy", "override"),
    [
        ("loadbalanced", None),
        ("roundrobin", "loadbalanced"),
        ("roundrobin", "load_balanced"),
        ("power_of_two", None),
        ("roundrobin", "power_of_two"),
    ],
)
def test_load_aware_dispatch(global_strategy, override):
    proxy = make_proxy(global_strategy, override)
    proxy.registry.increment_active_requests(NODES[0])
    lease = proxy.reserve(SchedulingContext(role="aggregated", model=MODEL))
    assert lease.address == NODES[1]
    lease.release()


@pytest.mark.parametrize("strategy", ["consistent_hash", "cache_aware"])
def test_affinity_dispatch(strategy):
    proxy = make_proxy(override=strategy)
    leases = [
        proxy.reserve(
            SchedulingContext(
                role="aggregated",
                model=MODEL,
                header="same-session",
                prompt="same prefix",
            )
        )
        for _ in range(3)
    ]
    assert len({lease.address for lease in leases}) == 1
    assert leases[0].address in NODES
    for lease in leases:
        lease.release()


@pytest.mark.parametrize("registry", [True, False])
def test_reservation_releases_once(registry):
    proxy = make_proxy(registry=registry)
    lease = proxy.reserve(SchedulingContext(role="aggregated", model=MODEL))
    assert lease is not None
    if registry:
        assert proxy.registry.get_active_requests(lease.address) == 1
    lease.release()
    lease.release()
    if registry:
        assert proxy.registry.get_active_requests(lease.address) == 0


@pytest.mark.parametrize("strategy", ["loadbalanced", "power_of_two"])
def test_load_accounting_without_registry(strategy):
    proxy = make_proxy(strategy, registry=False)
    first = proxy.reserve(SchedulingContext(role="aggregated", model=MODEL))
    second = proxy.reserve(SchedulingContext(role="aggregated", model=MODEL))
    assert first.address != second.address
    first.release()
    second.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, RuntimeError, asyncio.CancelledError])
async def test_real_stream_handler_releases_and_records_outcome(failure):
    proxy = make_proxy()
    request = Mock(headers={}, client=None)
    request.json = AsyncMock(
        return_value={"model": MODEL, "prompt": "hello", "stream": True}
    )

    async def forward(*args, **kwargs):
        yield b'data: {"choices":[{"text":"hello"}]}\n\n'
        if failure:
            raise failure()
        yield b"data: [DONE]\n\n"

    proxy.forward_request = forward
    with (
        patch.object(proxy.registry, "record_success") as success,
        patch.object(proxy.registry, "record_failure") as failed,
    ):
        response = await handle_completion("/v1/completions", request, proxy, False)
        if failure is RuntimeError:
            with pytest.raises(RuntimeError):
                _ = [chunk async for chunk in response.body_iterator]
        else:
            _ = [chunk async for chunk in response.body_iterator]
        assert success.call_count == (1 if failure is None else 0)
        assert failed.call_count == (1 if failure is RuntimeError else 0)
    assert all(proxy.registry.get_active_requests(address) == 0 for address in NODES)


@pytest.mark.asyncio
async def test_nonstream_cancellation_releases_reservation():
    proxy = make_proxy()
    request = Mock(headers={}, client=None)
    request.json = AsyncMock(return_value={"model": MODEL, "prompt": "hello"})

    async def forward(*args, **kwargs):
        raise asyncio.CancelledError()
        yield b""  # pragma: no cover

    proxy.forward_request = forward
    with patch("xpyd.routes.completions.track_request_end") as ended:
        with pytest.raises(asyncio.CancelledError):
            await handle_completion("/v1/completions", request, proxy, False)
        ended.assert_called_once()
    assert all(proxy.registry.get_active_requests(address) == 0 for address in NODES)


@pytest.mark.asyncio
async def test_closing_stream_is_not_a_success():
    proxy = make_proxy()
    request = Mock(headers={}, client=None)
    request.json = AsyncMock(
        return_value={"model": MODEL, "prompt": "hello", "stream": True}
    )

    async def forward(*args, **kwargs):
        yield b"first"
        yield b"second"

    proxy.forward_request = forward
    with patch.object(proxy.registry, "record_success") as success:
        response = await handle_completion("/v1/completions", request, proxy, False)
        await response.body_iterator.__anext__()
        await response.body_iterator.aclose()
        success.assert_not_called()
    assert all(proxy.registry.get_active_requests(address) == 0 for address in NODES)
