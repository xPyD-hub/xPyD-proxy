# SPDX-License-Identifier: Apache-2.0
"""Tests for ConsistentHashPolicy."""

import itertools
from unittest.mock import patch

from scheduler.consistent_hash import ConsistentHashPolicy


class TestConsistentHashPolicy:
    """Unit tests for the consistent-hash scheduling policy."""

    def test_same_session_same_worker(self):
        """Identical session_id always maps to the same worker."""
        policy = ConsistentHashPolicy(workers=["w1", "w2", "w3", "w4"])
        w1 = policy.select(session_id="user-abc")
        w2 = policy.select(session_id="user-abc")
        w3 = policy.select(session_id="user-abc")
        assert w1 == w2 == w3

    def test_different_sessions_distribute(self):
        """Different session keys spread across multiple workers."""
        policy = ConsistentHashPolicy(workers=["w1", "w2", "w3", "w4"])
        selected = {policy.select(session_id=f"user-{i}") for i in range(100)}
        assert len(selected) > 1  # not all on same worker

    def test_minimal_redistribution_on_node_removal(self):
        """Removing one of four workers moves roughly 25% of keys."""
        policy = ConsistentHashPolicy(workers=["w1", "w2", "w3", "w4"])
        before = {f"s{i}": policy.select(session_id=f"s{i}") for i in range(100)}
        policy.remove_worker("w3")
        after = {f"s{i}": policy.select(session_id=f"s{i}") for i in range(100)}
        moved = sum(1 for s in before if before[s] != after[s])
        assert moved < 35  # ~25% expected

    def test_hash_key_priority(self):
        """header > user > client_ip."""
        policy = ConsistentHashPolicy(workers=["w1", "w2"])
        r1 = policy.select(header="sess-1", user=None, client_ip="1.2.3.4")
        r2 = policy.select(header="sess-1", user="different", client_ip="5.6.7.8")
        assert r1 == r2  # header takes priority

    def test_single_worker_no_error(self):
        """A ring with one worker works without errors."""
        policy = ConsistentHashPolicy(workers=["only-one"])
        assert policy.select(session_id="any") == "only-one"

    def test_zero_workers_returns_none(self):
        """An empty ring returns None."""
        policy = ConsistentHashPolicy(workers=[])
        assert policy.select(session_id="any") is None

    def test_add_worker(self):
        """Dynamically adding a worker distributes some keys to it."""
        policy = ConsistentHashPolicy(workers=["w1", "w2"])
        before = {f"k{i}": policy.select(session_id=f"k{i}") for i in range(200)}
        policy.add_worker("w3")
        after = {f"k{i}": policy.select(session_id=f"k{i}") for i in range(200)}
        # Some keys should now go to w3
        assert "w3" in after.values()
        # Most keys should stay put
        moved = sum(1 for k in before if before[k] != after[k])
        assert moved < 100  # less than half

    def test_user_key_over_client_ip(self):
        """user field takes priority over client_ip."""
        policy = ConsistentHashPolicy(workers=["w1", "w2", "w3"])
        r1 = policy.select(user="alice", client_ip="1.2.3.4")
        r2 = policy.select(user="alice", client_ip="9.9.9.9")
        assert r1 == r2

    def test_schedule_with_request_context(self):
        """schedule() passes request context kwargs to select()."""
        policy = ConsistentHashPolicy(workers=["w1", "w2", "w3"])
        cycler = itertools.cycle(["w1", "w2", "w3"])
        # schedule with header should be consistent
        r1 = policy.schedule(cycler, header="sess-X")
        r2 = policy.schedule(cycler, header="sess-X")
        assert r1 == r2
        # schedule with user
        r3 = policy.schedule(cycler, user="bob")
        r4 = policy.schedule(cycler, user="bob")
        assert r3 == r4
        # schedule with client_ip
        r5 = policy.schedule(cycler, client_ip="10.0.0.1")
        r6 = policy.schedule(cycler, client_ip="10.0.0.1")
        assert r5 == r6
        # header takes priority over user/client_ip in schedule too
        r7 = policy.schedule(cycler, header="sess-X", user="other", client_ip="9.9.9.9")
        assert r7 == r1

    def test_hash_collision_handling(self):
        """Collision in the hash ring is skipped gracefully."""
        policy = ConsistentHashPolicy(workers=[], virtual_nodes=3)

        # Add first worker normally
        policy.add_worker("w1")
        assert len(policy._ring_keys) == 3  # noqa: SLF001

        # Mock _hash so w2's vnodes collide with w1's
        original_hash = ConsistentHashPolicy._hash
        w1_hashes = [original_hash("w1", i) for i in range(3)]

        def colliding_hash(key: str, index: int) -> int:
            if key == "w2":
                return w1_hashes[index]  # force collision
            return original_hash(key, index)

        with patch.object(ConsistentHashPolicy, "_hash", staticmethod(colliding_hash)):
            policy.add_worker("w2")

        # w2 in workers set but all vnodes collided; ring keeps w1 only
        assert "w2" in policy._workers  # noqa: SLF001
        assert len(policy._ring_keys) == 3  # noqa: SLF001
        # All ring entries still point to w1
        for h in policy._ring_keys:  # noqa: SLF001
            assert policy._ring_map[h] == "w1"  # noqa: SLF001
        # select still works (no crash)
        assert policy.select(session_id="test") == "w1"

        # Removing w2 must NOT corrupt w1's vnodes
        with patch.object(ConsistentHashPolicy, "_hash", staticmethod(colliding_hash)):
            policy.remove_worker("w2")

        assert "w2" not in policy._workers  # noqa: SLF001
        assert len(policy._ring_keys) == 3  # noqa: SLF001
        for h in policy._ring_keys:  # noqa: SLF001
            assert policy._ring_map[h] == "w1"  # noqa: SLF001
        assert policy.select(session_id="test") == "w1"
