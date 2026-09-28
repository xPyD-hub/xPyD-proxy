# SPDX-License-Identifier: Apache-2.0
"""Tests for draining and removing runtime instances."""

# Standard
import asyncio
import itertools
from unittest.mock import AsyncMock, MagicMock, patch

# Third Party
import pytest

# First Party
from xpyd.proxy import Proxy
from xpyd.registry import InstanceRegistry, InstanceStatus
from xpyd.scheduler import LoadBalancedScheduler, RoundRobinSchedulingPolicy


def _proxy(registry: InstanceRegistry) -> Proxy:
    proxy = Proxy(
        prefill_instances=["127.0.0.1:8100"],
        decode_instances=["127.0.0.1:8200", "127.0.0.1:8201"],
        model="test/model",
        registry=registry,
        scheduling_policy=RoundRobinSchedulingPolicy(registry=registry),
    )
    proxy.health_monitor = MagicMock()
    proxy.discovery = MagicMock()
    return proxy


@pytest.mark.asyncio
async def test_remove_waits_for_active_requests_to_finish() -> None:
    registry = InstanceRegistry()
    registry.add("prefill", "127.0.0.1:8100", model="test/model")
    registry.add("decode", "127.0.0.1:8200", model="test/model")
    registry.add("decode", "127.0.0.1:8201", model="test/model")
    for address in ("127.0.0.1:8100", "127.0.0.1:8200", "127.0.0.1:8201"):
        registry.mark_healthy(address)
    registry.increment_active_requests("127.0.0.1:8200")
    proxy = _proxy(registry)

    removal = asyncio.create_task(
        Proxy.drain_and_remove_instance(proxy, "decode", "127.0.0.1:8200", 1)
    )
    await asyncio.sleep(0.01)

    assert removal.done() is False
    assert (
        registry.get_instance_info("127.0.0.1:8200").status == InstanceStatus.DRAINING
    )
    assert registry.get_available_instances("decode") == ["127.0.0.1:8201"]

    registry.decrement_active_requests("127.0.0.1:8200")
    await removal

    assert proxy.decode_instances == ["127.0.0.1:8201"]
    with pytest.raises(KeyError):
        registry.get_instance_info("127.0.0.1:8200")
    proxy.health_monitor.remove_node.assert_called_once_with("127.0.0.1:8200")
    proxy.discovery.remove_instance.assert_called_once_with("decode", "127.0.0.1:8200")


@pytest.mark.asyncio
async def test_remove_timeout_leaves_instance_draining_for_retry() -> None:
    registry = InstanceRegistry()
    registry.add("decode", "127.0.0.1:8200")
    registry.mark_healthy("127.0.0.1:8200")
    registry.increment_active_requests("127.0.0.1:8200")
    proxy = _proxy(registry)

    with pytest.raises(TimeoutError, match="1 requests remain"):
        await Proxy.drain_and_remove_instance(proxy, "decode", "127.0.0.1:8200", 0)

    info = registry.get_instance_info("127.0.0.1:8200")
    assert info.status == InstanceStatus.DRAINING
    assert proxy.decode_instances == ["127.0.0.1:8200", "127.0.0.1:8201"]


@pytest.mark.asyncio
async def test_remove_aggregated_instance_updates_model_pool() -> None:
    registry = InstanceRegistry()
    registry.add("aggregated", "127.0.0.1:8000", model="test/model")
    registry.mark_healthy("127.0.0.1:8000")
    proxy = _proxy(registry)
    proxy.aggregated_instances = {"test/model": ["127.0.0.1:8000"]}

    await Proxy.drain_and_remove_instance(proxy, "aggregated", "127.0.0.1:8000", 1)

    assert proxy.aggregated_instances == {}
    with pytest.raises(KeyError):
        registry.get_instance_info("127.0.0.1:8000")
    proxy.discovery.remove_instance.assert_called_once_with(
        "aggregated", "127.0.0.1:8000"
    )


def test_pd_schedule_tracks_registry_active_requests() -> None:
    registry = InstanceRegistry()
    registry.add("prefill", "127.0.0.1:8100", model="test/model")
    registry.mark_healthy("127.0.0.1:8100")
    proxy = _proxy(registry)
    selected = Proxy.schedule(
        proxy,
        proxy.prefill_cycler,
        is_prompt=True,
        request_len=1,
        max_tokens=1,
        model="test/model",
    )

    assert selected == "127.0.0.1:8100"
    assert registry.get_active_requests(selected) == 1

    Proxy.schedule_completion(proxy, prefill_instance=selected, req_len=1)

    assert registry.get_active_requests(selected) == 0


@pytest.mark.asyncio
async def test_add_instance_updates_every_runtime_component() -> None:
    registry = InstanceRegistry()
    registry.add("decode", "127.0.0.1:8200", model="test/model")
    registry.mark_healthy("127.0.0.1:8200")
    proxy = _proxy(registry)

    with patch.object(
        Proxy,
        "_validated_instance_details",
        AsyncMock(return_value=("test/model", 4096)),
    ):
        added = await Proxy.add_instance(proxy, "decode", "127.0.0.1:8202")

    assert added is True
    info = registry.get_instance_info("127.0.0.1:8202")
    assert info.model == "test/model"
    assert info.status == InstanceStatus.HEALTHY
    assert proxy.decode_instances[-1] == "127.0.0.1:8202"
    proxy.health_monitor.add_node.assert_called_once_with("127.0.0.1:8202")
    proxy.discovery.add_instance.assert_called_once_with("decode", "127.0.0.1:8202")


@pytest.mark.asyncio
async def test_add_aggregated_instance_updates_model_pool() -> None:
    registry = InstanceRegistry()
    registry.add("aggregated", "127.0.0.1:8000", model="test/model")
    registry.mark_healthy("127.0.0.1:8000")
    proxy = _proxy(registry)
    proxy.aggregated_instances = {"test/model": ["127.0.0.1:8000"]}

    with patch.object(
        Proxy,
        "_validated_instance_details",
        AsyncMock(return_value=("test/model", 4096)),
    ):
        added = await Proxy.add_instance(proxy, "aggregated", "localhost:8000")

    assert added is True
    assert proxy.aggregated_instances["test/model"] == [
        "127.0.0.1:8000",
        "localhost:8000",
    ]
    assert registry.get_instance_info("localhost:8000").status == InstanceStatus.HEALTHY
    proxy.discovery.add_instance.assert_called_once_with("aggregated", "localhost:8000")


@pytest.mark.asyncio
async def test_load_balanced_scheduler_can_select_runtime_instance() -> None:
    registry = InstanceRegistry()
    registry.add("prefill", "127.0.0.1:8100", model="test/model")
    registry.add("decode", "127.0.0.1:8200", model="test/model")
    registry.mark_healthy("127.0.0.1:8100")
    registry.mark_healthy("127.0.0.1:8200")
    with patch(
        "xpyd.scheduler.load_balanced.query_instance_model_len",
        side_effect=lambda instances: [4096] * len(instances),
    ):
        policy = LoadBalancedScheduler(
            ["127.0.0.1:8100"],
            ["127.0.0.1:8200"],
            registry=registry,
        )
    proxy = _proxy(registry)
    proxy.scheduling_policy = policy
    proxy.prefill_instances = policy.prefill_instances
    proxy.decode_instances = policy.decode_instances
    proxy.prefill_cycler = itertools.cycle(proxy.prefill_instances)
    proxy.decode_cycler = itertools.cycle(proxy.decode_instances)
    policy.decode_bs_counter[0] = 1

    with patch.object(
        Proxy,
        "_validated_instance_details",
        AsyncMock(return_value=("test/model", 8192)),
    ):
        await Proxy.add_instance(proxy, "decode", "127.0.0.1:8202")

    selected = policy.schedule(
        proxy.decode_cycler,
        is_prompt=False,
        request_len=1,
        max_tokens=1,
        model="test/model",
    )
    assert selected == "127.0.0.1:8202"
    assert policy.decode_model_len == [4096, 8192]
