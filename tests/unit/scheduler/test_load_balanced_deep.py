# SPDX-License-Identifier: Apache-2.0
"""Capacity, load accounting, tie-breaking and concurrency through reservations."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from xpyd.scheduler import LoadBalancedScheduler, Scheduler, SchedulingContext


@pytest.fixture
def policy():
    with patch(
        "xpyd.scheduler.load_balanced.query_instance_model_len",
        side_effect=lambda nodes: [4096] * len(nodes),
    ):
        return LoadBalancedScheduler(["p1", "p2"], ["d1", "d2"])


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_release_reduces_load_and_is_idempotent(policy, role):
    runtime = Scheduler()
    nodes = policy.prefill_instances if role == "prefill" else policy.decode_instances
    context = SchedulingContext(role=role, request_len=100, max_tokens=50)
    leases = [runtime.reserve(policy, context, nodes) for _ in range(4)]
    counts = (
        policy.prefill_bs_counter if role == "prefill" else policy.decode_bs_counter
    )
    tokens = (
        policy.prefill_utils_counter
        if role == "prefill"
        else policy.decode_kv_utils_counter
    )
    assert counts == [2, 2]
    assert tokens == [200, 200]
    leases[0].release()
    leases[0].release()
    assert sum(counts) == 3
    assert sum(tokens) == 300
    for lease in leases:
        lease.release()
    assert counts == [0, 0]
    assert policy.prefill_utils_counter == policy.decode_kv_utils_counter == [0, 0]
    assert not any(runtime._active.values())


@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize(
    "length,max_tokens,accepted",
    [(4000, 95, True), (4000, 96, True), (4000, 97, False)],
)
def test_model_capacity_boundary(policy, role, length, max_tokens, accepted):
    nodes = policy.prefill_instances if role == "prefill" else policy.decode_instances
    lease = Scheduler().reserve(
        policy,
        SchedulingContext(role=role, request_len=length, max_tokens=max_tokens),
        nodes,
    )
    assert (lease is not None) is accepted
    if lease:
        lease.release()
    assert policy.prefill_bs_counter == policy.decode_bs_counter == [0, 0]


def test_concurrent_reservations_preserve_all_counters(policy):
    runtime = Scheduler()

    def run(_):
        for _ in range(50):
            p = runtime.reserve(
                policy,
                SchedulingContext(role="prefill", request_len=100, max_tokens=50),
                policy.prefill_instances,
            )
            d = runtime.reserve(
                policy,
                SchedulingContext(role="decode", request_len=100, max_tokens=50),
                policy.decode_instances,
            )
            assert p is not None and d is not None
            p.release()
            d.release()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, range(8)))
    assert policy.prefill_bs_counter == policy.decode_bs_counter == [0, 0]
    assert policy.prefill_utils_counter == policy.decode_kv_utils_counter == [0, 0]
    assert not any(runtime._active.values())


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_lower_token_load_wins_with_equal_busy_counts(policy, role):
    runtime = Scheduler()
    nodes = policy.prefill_instances if role == "prefill" else policy.decode_instances
    leases = [
        runtime.reserve(
            policy,
            SchedulingContext(role=role, request_len=length, max_tokens=1),
            nodes,
        )
        for length in (2000, 1000, 100)
    ]
    assert [lease.address for lease in leases] == [nodes[0], nodes[1], nodes[1]]
    for lease in leases:
        lease.release()


def test_decode_request_count_has_priority_over_token_load(policy):
    runtime = Scheduler()
    nodes = policy.decode_instances
    leases = [
        runtime.reserve(
            policy,
            SchedulingContext(role="decode", request_len=length, max_tokens=1),
            nodes,
        )
        for length in (2000, 100, 100, 100)
    ]
    assert [lease.address for lease in leases] == ["d1", "d2", "d2", "d1"]
    for lease in leases:
        lease.release()
