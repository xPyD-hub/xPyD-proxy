# SPDX-License-Identifier: Apache-2.0
"""Contracts shared by every configured policy and topology."""

import concurrent.futures
from unittest.mock import AsyncMock, patch

import pytest

from xpyd.config import ProxyConfig
from xpyd.proxy import Proxy, ProxyServer
from xpyd.registry import InstanceRegistry
from xpyd.scheduler import (
    Scheduler,
    SchedulingContext,
    SchedulingPolicy,
    default_registry,
)

STRATEGIES = [
    "roundrobin",
    "loadbalanced",
    "consistent_hash",
    "power_of_two",
    "cache_aware",
]
ROLES = ["aggregated", "prefill", "decode"]


@pytest.fixture
def model_lengths():
    with patch(
        "xpyd.scheduler.load_balanced.query_instance_model_len",
        side_effect=lambda nodes: [4096] * len(nodes),
    ):
        yield


def make_proxy(strategy, role):
    nodes = ["127.0.0.1:18101", "127.0.0.1:18102"]
    registry = InstanceRegistry()
    for address in nodes:
        registry.add(role, address, model="demo")
        registry.mark_healthy(address)
    prefill = list(nodes) if role == "prefill" else []
    decode = list(nodes) if role == "decode" else []
    policy = default_registry.build(
        strategy,
        prefill_instances=prefill,
        decode_instances=decode,
        workers=nodes,
        registry=registry,
    )
    return (
        Proxy(
            prefill,
            decode,
            "demo",
            policy,
            registry=registry,
            aggregated_instances={"demo": list(nodes)} if role == "aggregated" else {},
        ),
        nodes,
    )


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("role", ROLES)
def test_filtering_and_idempotent_release(model_lengths, strategy, role):
    proxy, nodes = make_proxy(strategy, role)
    context = SchedulingContext(
        role=role, model="demo", request_len=5, max_tokens=1, prompt="prefix"
    )
    proxy.registry.mark_unhealthy(nodes[0])
    lease = proxy.reserve(context)
    assert lease.address == nodes[1]
    assert proxy.registry.get_active_requests(nodes[1]) == 1
    lease.release()
    lease.release()
    assert proxy.registry.get_active_requests(nodes[1]) == 0
    proxy.registry.begin_draining(role, nodes[1])
    assert proxy.reserve(context) is None


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.asyncio
async def test_membership_changes_through_proxy(model_lengths, strategy, role):
    proxy, nodes = make_proxy(strategy, role)
    context = SchedulingContext(role=role, model="demo", request_len=5, prompt="prefix")
    # Create any lazily configured per-model policy before changing membership.
    proxy.reserve(context).release()
    for address in nodes:
        await proxy.drain_and_remove_instance(role, address, 0)
    assert proxy.reserve(context) is None
    with patch.object(
        Proxy, "_validated_instance_details", AsyncMock(return_value=("demo", 8192))
    ):
        assert await proxy.add_instance(role, nodes[0])
    lease = proxy.reserve(context)
    assert lease.address == nodes[0]
    lease.release()
    assert proxy.registry.get_active_requests(nodes[0]) == 0


@pytest.mark.parametrize("strategy", ["loadbalanced", "power_of_two"])
@pytest.mark.parametrize("role", ROLES)
def test_concurrent_reservations_preserve_load(model_lengths, strategy, role):
    proxy, nodes = make_proxy(strategy, role)
    # Initialize per-model policy before calling from multiple worker threads.
    proxy.reserve(SchedulingContext(role=role, model="demo", request_len=5)).release()
    context = SchedulingContext(role=role, model="demo", request_len=5, max_tokens=1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        leases = list(pool.map(lambda _: proxy.reserve(context), range(20)))
        assert [proxy.registry.get_active_requests(node) for node in nodes] == [10, 10]
        list(pool.map(lambda lease: lease.release(), leases * 2))
    assert [proxy.registry.get_active_requests(node) for node in nodes] == [0, 0]


def test_release_uses_original_policy_after_fallback_changes(model_lengths):
    proxy, nodes = make_proxy("loadbalanced", "prefill")
    lease = proxy.reserve(
        SchedulingContext(role="prefill", model="demo", request_len=30)
    )
    proxy._round_robin_models.add("demo")
    lease.release()
    assert proxy.scheduling_policy.prefill_bs_counter == [0, 0]
    assert proxy.scheduling_policy.prefill_utils_counter == [0, 0]


def test_invalid_selection_is_rejected_without_accounting():
    class InvalidPolicy(SchedulingPolicy):
        def select_node(self, context, candidates):
            return "not-a-candidate"

    runtime = Scheduler()
    with pytest.raises(ValueError, match="ineligible"):
        runtime.reserve(InvalidPolicy(), SchedulingContext(role="decode"), ["d"])
    assert not runtime._active


@pytest.mark.parametrize("role", ROLES)
def test_custom_policy_from_yaml_needs_no_proxy_branch(tmp_path, monkeypatch, role):
    class LastPolicy(SchedulingPolicy):
        def __init__(self, workers=(), registry=None, offset=0):
            super().__init__(registry)
            self.offset = offset
            self.finished = []

        def select_node(self, context, candidates):
            return candidates[-1 - self.offset].address

        def on_request_finished(self, context, address):
            self.finished.append((context.role, address))

    monkeypatch.setitem(default_registry._policies, "test_last", LastPolicy)
    path = tmp_path / "proxy.yaml"
    opposite = "decode" if role == "prefill" else "prefill"
    path.write_text(
        "scheduling: test_last\n"
        "test_last:\n  offset: 1\n"
        "instances:\n"
        f"  - {{address: '127.0.0.1:18101', role: {role}, model: demo}}\n"
        f"  - {{address: '127.0.0.1:18102', role: {role}, model: demo}}\n"
        + (
            f"  - {{address: '127.0.0.1:18103', role: {opposite}, model: demo}}\n"
            if role != "aggregated"
            else ""
        )
    )
    server = ProxyServer(ProxyConfig.from_yaml(path))
    lease = server.proxy_instance.reserve(SchedulingContext(role=role, model="demo"))
    assert lease.address == "127.0.0.1:18101"
    lease.release()
    assert lease._policy.finished == [(role, lease.address)]


def test_configured_hash_options_reach_aggregated_policy(tmp_path):
    path = tmp_path / "proxy.yaml"
    path.write_text(
        "scheduling: consistent_hash\nconsistent_hash:\n  virtual_nodes: 7\n"
        "instances:\n"
        "  - {address: '127.0.0.1:18101', role: aggregated, model: demo}\n"
    )
    proxy = ProxyServer(ProxyConfig.from_yaml(path)).proxy_instance
    lease = proxy.reserve(SchedulingContext(role="aggregated", model="demo"))
    assert lease._policy._virtual_nodes == 7
    lease.release()


def test_configuration_selects_policy_through_common_factory():
    from xpyd.proxy import _create_scheduling_policy
    from xpyd.scheduler import RoundRobinSchedulingPolicy

    config = ProxyConfig(
        model="demo",
        prefill=["127.0.0.1:18101"],
        decode=["127.0.0.1:18102"],
        scheduling="roundrobin",
    )
    policy = _create_scheduling_policy(config)
    assert isinstance(policy, RoundRobinSchedulingPolicy)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_only_unified_policy_entry_points_are_exposed(model_lengths, strategy):
    policy = default_registry.build(strategy)
    for name in (
        "schedule",
        "schedule_completion",
        "select",
        "select_from",
        "add_worker",
        "remove_worker",
        "add_instance_state",
        "remove_instance_state",
    ):
        assert not hasattr(policy, name)


def test_no_proxy_scheduler_compatibility_exports_or_adapters():
    import xpyd.proxy as proxy_module

    for name in ("RoundRobinSchedulingPolicy", "LoadBalancedScheduler"):
        assert not hasattr(proxy_module, name)
    for name in (
        "schedule",
        "schedule_completion",
        "schedule_aggregated",
        "schedule_aggregated_completion",
        "on_done",
        "exception_handler",
    ):
        assert not hasattr(Proxy, name)
    assert not hasattr(Scheduler, "remember")
    assert not hasattr(Scheduler, "finish")
    assert not hasattr(default_registry, "create")


def test_legacy_roundrobin_flag_is_not_a_strategy_options_mapping(tmp_path):
    path = tmp_path / "proxy.yaml"
    path.write_text(
        "model: demo\nroundrobin: true\n"
        "prefill: ['127.0.0.1:18101']\ndecode: ['127.0.0.1:18102']\n"
    )
    config = ProxyConfig.from_yaml(path)
    assert config.roundrobin


@pytest.mark.asyncio
async def test_auto_discovered_membership_removal():
    proxy, nodes = make_proxy("consistent_hash", "aggregated")
    proxy.aggregated_instances = {"": list(nodes)}
    proxy.reserve(SchedulingContext(role="aggregated", model="demo")).release()
    await proxy.drain_and_remove_instance("aggregated", nodes[0], 0)
    lease = proxy.reserve(SchedulingContext(role="aggregated", model="demo"))
    assert lease.address == nodes[1]
    lease.release()


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_unknown_policy_option_is_not_silently_ignored(strategy):
    with pytest.raises(TypeError):
        default_registry.build(strategy, unknown_option=True)


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.asyncio
async def test_failed_membership_update_rolls_back(model_lengths, strategy):
    from unittest.mock import Mock

    proxy, nodes = make_proxy(strategy, "decode")
    proxy.discovery = Mock()
    proxy.discovery.add_instance.side_effect = RuntimeError("discovery failed")
    address = "127.0.0.1:18103"
    with patch.object(
        Proxy, "_validated_instance_details", AsyncMock(return_value=("demo", 8192))
    ):
        with pytest.raises(RuntimeError, match="discovery failed"):
            await proxy.add_instance("decode", address)
    assert proxy.decode_instances == nodes
    with pytest.raises(KeyError):
        proxy.registry.get_instance_info(address)
    lease = proxy.reserve(
        SchedulingContext(role="decode", model="demo", request_len=5, prompt="prefix")
    )
    assert lease.address in nodes
    lease.release()
