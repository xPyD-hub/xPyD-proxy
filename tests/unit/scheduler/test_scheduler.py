# SPDX-License-Identifier: Apache-2.0
"""Policy selection and runtime eligibility are separate contracts."""

from unittest.mock import patch

import pytest

from xpyd.registry import InstanceRegistry
from xpyd.scheduler import (
    Candidate,
    RoundRobinSchedulingPolicy,
    Scheduler,
    SchedulingContext,
    SchedulingPolicy,
    default_registry,
)


def test_base_class_is_abstract():
    with pytest.raises(TypeError):
        SchedulingPolicy()


def test_round_robin_cycles_through_instances():
    policy = RoundRobinSchedulingPolicy()
    candidates = [Candidate(address) for address in ["a:1", "b:2", "c:3"]]
    context = SchedulingContext(role="prefill")
    assert [policy.select_node(context, candidates) for _ in range(6)] == [
        "a:1",
        "b:2",
        "c:3",
        "a:1",
        "b:2",
        "c:3",
    ]


def test_round_robin_ignores_lengths_and_tracks_each_pool():
    policy = RoundRobinSchedulingPolicy()
    candidates = [Candidate("a"), Candidate("b")]
    for role in ("prefill", "decode", "aggregated"):
        for model in ("first", "second"):
            assert (
                policy.select_node(
                    SchedulingContext(
                        role=role, model=model, request_len=9999, max_tokens=9999
                    ),
                    candidates,
                )
                == "a"
            )
            assert (
                policy.select_node(
                    SchedulingContext(
                        role=role, model=model, request_len=1, max_tokens=1
                    ),
                    candidates,
                )
                == "b"
            )


def test_round_robin_finish_is_noop():
    RoundRobinSchedulingPolicy().on_request_finished(
        SchedulingContext(role="prefill", request_len=100), "a"
    )


@pytest.mark.parametrize("strategy", ["roundrobin", "loadbalanced"])
@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize("healthy", [[], ["node-4"], ["node-1", "node-3"]])
def test_runtime_filters_unhealthy(strategy, role, healthy):
    nodes = [f"node-{i}" for i in range(5)]
    registry = InstanceRegistry()
    for node in nodes:
        registry.add(role, node)
    for node in healthy:
        registry.mark_healthy(node)
    with patch(
        "xpyd.scheduler.load_balanced.query_instance_model_len",
        side_effect=lambda instances: [4096] * len(instances),
    ):
        policy = default_registry.build(
            strategy,
            prefill_instances=nodes if role == "prefill" else [],
            decode_instances=nodes if role == "decode" else [],
        )
    runtime = Scheduler(registry)
    context = SchedulingContext(role=role, request_len=100, max_tokens=10)
    leases = [runtime.reserve(policy, context, nodes) for _ in range(6)]
    if healthy:
        assert {lease.address for lease in leases} == set(healthy)
        for lease in leases:
            lease.release()
        assert all(registry.get_active_requests(node) == 0 for node in nodes)
    else:
        assert leases == [None] * 6


@pytest.mark.parametrize("strategy", ["roundrobin", "loadbalanced"])
@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_runtime_without_registry_uses_explicit_candidates(strategy, role):
    nodes = ["a", "b", "c"]
    with patch(
        "xpyd.scheduler.load_balanced.query_instance_model_len",
        side_effect=lambda instances: [4096] * len(instances),
    ):
        policy = default_registry.build(
            strategy,
            prefill_instances=nodes if role == "prefill" else [],
            decode_instances=nodes if role == "decode" else [],
        )
    runtime = Scheduler()
    context = SchedulingContext(role=role, request_len=100, max_tokens=1)
    leases = [runtime.reserve(policy, context, nodes) for _ in nodes]
    assert {lease.address for lease in leases} == set(nodes)
    for lease in leases:
        lease.release()
    assert not any(runtime._active.values())
