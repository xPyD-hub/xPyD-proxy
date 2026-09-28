# SPDX-License-Identifier: Apache-2.0
"""Power-of-two selects candidates; the runtime owns request counts."""

from collections import Counter
from unittest.mock import patch

import pytest

from xpyd.registry import InstanceRegistry
from xpyd.scheduler import Candidate, PowerOfTwoPolicy, Scheduler, SchedulingContext

CONTEXT = SchedulingContext(role="decode")


@pytest.mark.parametrize(
    "pair,expected", [(["w1", "w2"], "w2"), (["w1", "w3"], "w3"), (["w2", "w3"], "w2")]
)
def test_picks_less_loaded_member_of_sampled_pair(pair, expected):
    policy = PowerOfTwoPolicy()
    candidates = [Candidate("w1", 10), Candidate("w2", 2), Candidate("w3", 5)]
    with patch(
        "xpyd.scheduler.power_of_two.random.sample", return_value=pair
    ) as sample:
        assert policy.select_node(CONTEXT, candidates) == expected
    sample.assert_called_once_with(["w1", "w2", "w3"], 2)
    assert policy.last_pair == tuple(pair)


def test_random_pairs_and_equal_load_distribution():
    policy = PowerOfTwoPolicy()
    candidates = [Candidate(f"w{i}") for i in range(4)]
    pairs = set()
    counts = Counter()
    for _ in range(1000):
        counts[policy.select_node(CONTEXT, candidates)] += 1
        pairs.add(tuple(sorted(policy.last_pair)))
    assert len(pairs) == 6
    assert set(counts) == {candidate.address for candidate in candidates}


def test_candidates_are_the_only_membership_source():
    policy = PowerOfTwoPolicy()
    assert policy.select_node(CONTEXT, [Candidate("w1")]) == "w1"
    assert policy.last_pair == ("w1",)
    policy.on_instance_added("decode", "w2", 4096)
    assert policy.select_node(CONTEXT, [Candidate("w2")]) == "w2"
    policy.on_instance_removed("decode", "w2", 0)
    assert policy.select_node(CONTEXT, []) is None
    assert policy.last_pair == ()


def test_selection_does_not_maintain_a_second_load_counter():
    policy = PowerOfTwoPolicy()
    candidates = [Candidate("w1", 100), Candidate("w2", 0)]
    for _ in range(200):
        assert policy.select_node(CONTEXT, candidates) == "w2"
    assert candidates == [Candidate("w1", 100), Candidate("w2", 0)]


@pytest.mark.parametrize("with_registry", [True, False])
def test_runtime_increments_and_releases_load(with_registry):
    registry = InstanceRegistry() if with_registry else None
    nodes = ["w1", "w2"]
    if registry:
        for node in nodes:
            registry.add("decode", node)
            registry.mark_healthy(node)
    runtime = Scheduler(registry)
    policy = PowerOfTwoPolicy()
    leases = [runtime.reserve(policy, CONTEXT, nodes) for _ in range(10)]
    assert Counter(lease.address for lease in leases) == {"w1": 5, "w2": 5}
    for lease in leases:
        lease.release()
        lease.release()
    if registry:
        assert [registry.get_active_requests(node) for node in nodes] == [0, 0]
    else:
        assert not any(runtime._active.values())


def test_registry_load_is_role_aware():
    registry = InstanceRegistry()
    nodes = ["p1", "p2", "d1", "d2"]
    for role, addresses in (("prefill", nodes[:2]), ("decode", nodes[2:])):
        for address in addresses:
            registry.add(role, address, model="model")
            registry.mark_healthy(address)
    registry.increment_active_requests("p1")
    registry.increment_active_requests("d1")
    runtime = Scheduler(registry)
    policy = PowerOfTwoPolicy()
    p = runtime.reserve(policy, SchedulingContext(role="prefill", model="model"), nodes)
    d = runtime.reserve(policy, SchedulingContext(role="decode", model="model"), nodes)
    assert p.address == "p2"
    assert d.address == "d2"
    p.release()
    d.release()
