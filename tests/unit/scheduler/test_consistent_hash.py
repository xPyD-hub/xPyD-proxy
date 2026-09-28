# SPDX-License-Identifier: Apache-2.0
"""Tests for ConsistentHashPolicy."""

from unittest.mock import patch

from xpyd.registry import InstanceRegistry
from xpyd.scheduler import Candidate, Scheduler, SchedulingContext
from xpyd.scheduler.consistent_hash import ConsistentHashPolicy


class TestConsistentHashPolicy:
    """Unit tests for the consistent-hash scheduling policy."""

    def test_same_session_same_worker(self):
        """Identical session_id always maps to the same worker."""
        policy = ConsistentHashPolicy(workers=["w1", "w2", "w3", "w4"])
        w1 = policy.select_node(
            SchedulingContext(role="decode", session_id="user-abc"),
            [Candidate(address) for address in sorted(policy._workers)],
        )
        w2 = policy.select_node(
            SchedulingContext(role="decode", session_id="user-abc"),
            [Candidate(address) for address in sorted(policy._workers)],
        )
        w3 = policy.select_node(
            SchedulingContext(role="decode", session_id="user-abc"),
            [Candidate(address) for address in sorted(policy._workers)],
        )
        assert w1 == w2 == w3

    def test_different_sessions_distribute(self):
        """Different session keys spread across multiple workers."""
        policy = ConsistentHashPolicy(workers=["w1", "w2", "w3", "w4"])
        selected = {
            policy.select_node(
                SchedulingContext(role="decode", session_id=f"user-{i}"),
                [Candidate(address) for address in sorted(policy._workers)],
            )
            for i in range(100)
        }
        assert len(selected) > 1  # not all on same worker

    def test_minimal_redistribution_on_node_removal(self):
        """Removing one of four workers moves roughly 25% of keys."""
        policy = ConsistentHashPolicy(workers=["w1", "w2", "w3", "w4"])
        before = {
            f"s{i}": policy.select_node(
                SchedulingContext(role="decode", session_id=f"s{i}"),
                [Candidate(address) for address in sorted(policy._workers)],
            )
            for i in range(100)
        }
        policy.on_instance_removed("decode", "w3", 0)
        after = {
            f"s{i}": policy.select_node(
                SchedulingContext(role="decode", session_id=f"s{i}"),
                [Candidate(address) for address in sorted(policy._workers)],
            )
            for i in range(100)
        }
        moved = sum(1 for s in before if before[s] != after[s])
        assert moved < 35  # ~25% expected

    def test_hash_key_priority(self):
        """header > user > client_ip."""
        policy = ConsistentHashPolicy(workers=["w1", "w2"])
        r1 = policy.select_node(
            SchedulingContext(
                role="decode", header="sess-1", user=None, client_ip="1.2.3.4"
            ),
            [Candidate(address) for address in sorted(policy._workers)],
        )
        r2 = policy.select_node(
            SchedulingContext(
                role="decode", header="sess-1", user="different", client_ip="5.6.7.8"
            ),
            [Candidate(address) for address in sorted(policy._workers)],
        )
        assert r1 == r2  # header takes priority

    def test_single_worker_no_error(self):
        """A ring with one worker works without errors."""
        policy = ConsistentHashPolicy(workers=["only-one"])
        assert (
            policy.select_node(
                SchedulingContext(role="decode", session_id="any"),
                [Candidate(address) for address in sorted(policy._workers)],
            )
            == "only-one"
        )

    def test_zero_workers_returns_none(self):
        """An empty ring returns None."""
        policy = ConsistentHashPolicy(workers=[])
        assert (
            policy.select_node(
                SchedulingContext(role="decode", session_id="any"),
                [Candidate(address) for address in sorted(policy._workers)],
            )
            is None
        )

    def test_add_worker(self):
        """Dynamically adding a worker distributes some keys to it."""
        policy = ConsistentHashPolicy(workers=["w1", "w2"])
        before = {
            f"k{i}": policy.select_node(
                SchedulingContext(role="decode", session_id=f"k{i}"),
                [Candidate(address) for address in sorted(policy._workers)],
            )
            for i in range(200)
        }
        policy.on_instance_added("decode", "w3", 4096)
        after = {
            f"k{i}": policy.select_node(
                SchedulingContext(role="decode", session_id=f"k{i}"),
                [Candidate(address) for address in sorted(policy._workers)],
            )
            for i in range(200)
        }
        # Some keys should now go to w3
        assert "w3" in after.values()
        # Most keys should stay put
        moved = sum(1 for k in before if before[k] != after[k])
        assert moved < 100  # less than half

    def test_user_key_over_client_ip(self):
        """user field takes priority over client_ip."""
        policy = ConsistentHashPolicy(workers=["w1", "w2", "w3"])
        r1 = policy.select_node(
            SchedulingContext(role="decode", user="alice", client_ip="1.2.3.4"),
            [Candidate(address) for address in sorted(policy._workers)],
        )
        r2 = policy.select_node(
            SchedulingContext(role="decode", user="alice", client_ip="9.9.9.9"),
            [Candidate(address) for address in sorted(policy._workers)],
        )
        assert r1 == r2

    def test_runtime_preserves_request_context(self):
        policy = ConsistentHashPolicy(workers=["w1", "w2", "w3"])
        runtime = Scheduler()
        nodes = ["w1", "w2", "w3"]
        for field in ("header", "user", "client_ip", "session_id"):
            context = SchedulingContext(role="decode", **{field: "session"})
            leases = [runtime.reserve(policy, context, nodes) for _ in range(3)]
            expected = policy.select_node(context, [Candidate(node) for node in nodes])
            assert {lease.address for lease in leases} == {expected}
            for lease in leases:
                lease.release()

    def test_schedule_does_not_fall_back_to_a_draining_worker(self):
        registry = InstanceRegistry()
        registry.add("decode", "w1")
        registry.mark_healthy("w1")
        registry.begin_draining("decode", "w1")
        policy = ConsistentHashPolicy(workers=["w1"], registry=registry)

        selected = Scheduler(registry).reserve(
            policy,
            SchedulingContext(role="decode", session_id="session"),
            ["w1"],
        )

        assert selected is None

    def test_hash_collision_handling(self):
        """Collision in the hash ring is skipped gracefully."""
        policy = ConsistentHashPolicy(workers=[], virtual_nodes=3)

        # Add first worker normally
        policy.on_instance_added("decode", "w1", 4096)
        assert len(policy._ring_keys) == 3  # noqa: SLF001

        # Mock _hash so w2's vnodes collide with w1's
        original_hash = ConsistentHashPolicy._hash
        w1_hashes = [original_hash("w1", i) for i in range(3)]

        def colliding_hash(key: str, index: int) -> int:
            if key == "w2":
                return w1_hashes[index]  # force collision
            return original_hash(key, index)

        with patch.object(ConsistentHashPolicy, "_hash", staticmethod(colliding_hash)):
            policy.on_instance_added("decode", "w2", 4096)

        # w2 in workers set but all vnodes collided; ring keeps w1 only
        assert "w2" in policy._workers  # noqa: SLF001
        assert len(policy._ring_keys) == 3  # noqa: SLF001
        # All ring entries still point to w1
        for h in policy._ring_keys:  # noqa: SLF001
            assert policy._ring_map[h] == "w1"  # noqa: SLF001
        # select still works (no crash)
        assert (
            policy.select_node(
                SchedulingContext(role="decode", session_id="test"),
                [Candidate(address) for address in sorted(policy._workers)],
            )
            == "w1"
        )

        # Removing w2 must NOT corrupt w1's vnodes
        with patch.object(ConsistentHashPolicy, "_hash", staticmethod(colliding_hash)):
            policy.on_instance_removed("decode", "w2", 0)

        assert "w2" not in policy._workers  # noqa: SLF001
        assert len(policy._ring_keys) == 3  # noqa: SLF001
        for h in policy._ring_keys:  # noqa: SLF001
            assert policy._ring_map[h] == "w1"  # noqa: SLF001
        assert (
            policy.select_node(
                SchedulingContext(role="decode", session_id="test"),
                [Candidate(address) for address in sorted(policy._workers)],
            )
            == "w1"
        )
