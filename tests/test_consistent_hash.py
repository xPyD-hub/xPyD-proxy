# SPDX-License-Identifier: Apache-2.0
"""Tests for ConsistentHashPolicy."""

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
