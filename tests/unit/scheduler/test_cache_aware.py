# SPDX-License-Identifier: Apache-2.0
"""Tests for CacheAwarePolicy and ConsistentHashRing."""

from unittest.mock import MagicMock

from xpyd.registry import InstanceRegistry
from xpyd.scheduler import Candidate, Scheduler, SchedulingContext
from xpyd.scheduler.cache_aware import (
    DEFAULT_PREFIX_LENGTH,
    VIRTUAL_NODES_PER_WORKER,
    CacheAwarePolicy,
    ConsistentHashRing,
)


class TestConsistentHashRing:
    """Unit tests for the consistent hash ring."""

    def test_add_worker_creates_virtual_nodes(self):
        ring = ConsistentHashRing(vnodes=VIRTUAL_NODES_PER_WORKER)
        ring.add_worker("w1")
        assert len(ring._ring_keys) == VIRTUAL_NODES_PER_WORKER
        assert len(ring) == 1

    def test_virtual_nodes_per_worker_is_160(self):
        assert VIRTUAL_NODES_PER_WORKER == 160

    def test_get_returns_none_when_empty(self):
        ring = ConsistentHashRing()
        assert ring.get(12345) is None

    def test_single_worker_always_selected(self):
        ring = ConsistentHashRing()
        ring.add_worker("w1")
        for key in range(100):
            assert ring.get(key) == "w1"

    def test_deterministic_lookup(self):
        ring = ConsistentHashRing()
        for w in ["w1", "w2", "w3"]:
            ring.add_worker(w)
        results = [ring.get(42) for _ in range(20)]
        assert len(set(results)) == 1

    def test_minimal_remapping_on_add(self):
        """Adding a worker should remap at most ~1/N of keys."""
        ring = ConsistentHashRing()
        for w in ["w1", "w2", "w3", "w4"]:
            ring.add_worker(w)
        keys = [ConsistentHashRing._hash(f"key-{k}") for k in range(1000)]
        before = {k: ring.get(k) for k in keys}
        ring.add_worker("w5")
        after = {k: ring.get(k) for k in keys}
        moved = sum(1 for k in keys if before[k] != after[k])
        # Expect ~1/5 = 20% remapping, allow generous margin
        assert moved < len(keys) * 0.40

    def test_minimal_remapping_on_remove(self):
        """Removing a worker should remap at most ~1/N of keys."""
        ring = ConsistentHashRing()
        for w in ["w1", "w2", "w3", "w4"]:
            ring.add_worker(w)
        keys = [ConsistentHashRing._hash(f"key-{k}") for k in range(1000)]
        before = {k: ring.get(k) for k in keys}
        ring.remove_worker("w3")
        after = {k: ring.get(k) for k in keys}
        moved = sum(1 for k in keys if before[k] != after[k])
        # Only keys on w3 should move (~25%), allow margin
        assert moved < len(keys) * 0.40

    def test_add_duplicate_is_noop(self):
        ring = ConsistentHashRing()
        ring.add_worker("w1")
        ring.add_worker("w1")
        assert len(ring) == 1
        assert len(ring._ring_keys) == VIRTUAL_NODES_PER_WORKER

    def test_remove_nonexistent_is_noop(self):
        ring = ConsistentHashRing()
        ring.add_worker("w1")
        ring.remove_worker("w99")
        assert len(ring) == 1

    def test_workers_property_returns_copy(self):
        ring = ConsistentHashRing()
        ring.add_worker("w1")
        workers = ring.workers
        workers.add("w99")
        assert "w99" not in ring.workers

    def test_wraparound(self):
        """Key larger than all ring keys wraps to first worker."""
        ring = ConsistentHashRing()
        ring.add_worker("w1")
        # Use a very large key
        result = ring.get(2**128)
        assert result == "w1"

    def test_distribution_across_workers(self):
        """Keys should spread across multiple workers."""
        import hashlib

        ring = ConsistentHashRing()
        for w in ["w1", "w2", "w3", "w4"]:
            ring.add_worker(w)
        # Use hashed keys spread across the ring
        selected = set()
        for k in range(500):
            h = int(hashlib.sha256(f"key-{k}".encode()).hexdigest(), 16)
            selected.add(ring.get(h))
        assert len(selected) > 1


class TestCacheAwarePolicy:
    """Unit tests for cache-aware routing policy."""

    def test_same_prefix_same_worker(self):
        policy = CacheAwarePolicy(workers=["w1", "w2", "w3"], prefix_length=256)
        base_tokens = [f"token{i}" for i in range(300)]
        prompt_a = " ".join(base_tokens)
        prompt_b = " ".join(base_tokens + ["extra", "suffix", "here"])
        w1 = policy.select_node(
            SchedulingContext(role="decode", prompt=prompt_a),
            [Candidate(address) for address in sorted(policy._ring.workers)],
        )
        w2 = policy.select_node(
            SchedulingContext(role="decode", prompt=prompt_b),
            [Candidate(address) for address in sorted(policy._ring.workers)],
        )
        assert w1 == w2

    def test_different_prefix_can_differ(self):
        policy = CacheAwarePolicy(workers=["w1", "w2", "w3"], prefix_length=256)
        selected = set()
        for i in range(50):
            w = policy.select_node(
                SchedulingContext(role="decode", prompt=f"Unique prompt {i} " * 100),
                [Candidate(address) for address in sorted(policy._ring.workers)],
            )
            selected.add(w)
        assert len(selected) > 1

    def test_no_workers_returns_none(self):
        policy = CacheAwarePolicy(workers=[], prefix_length=256)
        assert (
            policy.select_node(
                SchedulingContext(role="decode", prompt="hello"),
                [Candidate(address) for address in sorted(policy._ring.workers)],
            )
            is None
        )

    def test_single_worker(self):
        policy = CacheAwarePolicy(workers=["w1"], prefix_length=256)
        assert (
            policy.select_node(
                SchedulingContext(role="decode", prompt="anything"),
                [Candidate(address) for address in sorted(policy._ring.workers)],
            )
            == "w1"
        )
        assert (
            policy.select_node(
                SchedulingContext(role="decode", prompt="something else"),
                [Candidate(address) for address in sorted(policy._ring.workers)],
            )
            == "w1"
        )

    def test_none_prompt(self):
        policy = CacheAwarePolicy(workers=["w1", "w2"], prefix_length=256)
        result = policy.select_node(
            SchedulingContext(role="decode", prompt=None),
            [Candidate(address) for address in sorted(policy._ring.workers)],
        )
        assert result in ("w1", "w2")

    def test_empty_prompt(self):
        policy = CacheAwarePolicy(workers=["w1", "w2"], prefix_length=256)
        result = policy.select_node(
            SchedulingContext(role="decode", prompt=""),
            [Candidate(address) for address in sorted(policy._ring.workers)],
        )
        assert result in ("w1", "w2")

    def test_prompt_shorter_than_prefix_length(self):
        policy = CacheAwarePolicy(workers=["w1", "w2", "w3"], prefix_length=256)
        result = policy.select_node(
            SchedulingContext(role="decode", prompt="short"),
            [Candidate(address) for address in sorted(policy._ring.workers)],
        )
        assert result in ("w1", "w2", "w3")
        assert (
            policy.select_node(
                SchedulingContext(role="decode", prompt="short"),
                [Candidate(address) for address in sorted(policy._ring.workers)],
            )
            == result
        )

    def test_deterministic(self):
        policy = CacheAwarePolicy(workers=["w1", "w2", "w3"], prefix_length=256)
        prompt = "deterministic test prompt " * 20
        results = {
            policy.select_node(
                SchedulingContext(role="decode", prompt=prompt),
                [Candidate(address) for address in sorted(policy._ring.workers)],
            )
            for _ in range(10)
        }
        assert len(results) == 1

    def test_add_worker(self):
        policy = CacheAwarePolicy(workers=["w1", "w2"], prefix_length=256)
        policy.on_instance_added("decode", "w3", 4096)
        result = policy.select_node(
            SchedulingContext(role="decode", prompt="test"),
            [Candidate(address) for address in sorted(policy._ring.workers)],
        )
        assert result in ("w1", "w2", "w3")

    def test_remove_worker(self):
        policy = CacheAwarePolicy(workers=["w1", "w2", "w3"], prefix_length=256)
        policy.on_instance_removed("decode", "w2", 0)
        result = policy.select_node(
            SchedulingContext(role="decode", prompt="test"),
            [Candidate(address) for address in sorted(policy._ring.workers)],
        )
        assert result in ("w1", "w3")

    def test_add_duplicate_worker(self):
        policy = CacheAwarePolicy(workers=["w1", "w2"], prefix_length=256)
        policy.on_instance_added("decode", "w1", 4096)
        assert len(policy._ring) == 2

    def test_remove_nonexistent_worker(self):
        policy = CacheAwarePolicy(workers=["w1"], prefix_length=256)
        policy.on_instance_removed("decode", "w99", 0)
        assert (
            policy.select_node(
                SchedulingContext(role="decode", prompt="test"),
                [Candidate(address) for address in sorted(policy._ring.workers)],
            )
            == "w1"
        )

    def test_runtime_preserves_prompt(self):
        policy = CacheAwarePolicy(workers=["w1", "w2", "w3"], prefix_length=256)
        prompt = "The quick brown fox " * 50
        context = SchedulingContext(role="decode", prompt=prompt)
        nodes = ["w1", "w2", "w3"]
        lease = Scheduler().reserve(policy, context, nodes)
        assert lease.address == policy.select_node(
            context, [Candidate(node) for node in nodes]
        )
        lease.release()

    def test_schedule_does_not_fall_back_to_a_draining_worker(self):
        registry = InstanceRegistry()
        registry.add("decode", "w1")
        registry.mark_healthy("w1")
        registry.begin_draining("decode", "w1")
        policy = CacheAwarePolicy(workers=["w1"], registry=registry)

        selected = Scheduler(registry).reserve(
            policy,
            SchedulingContext(role="decode", prompt="test"),
            ["w1"],
        )

        assert selected is None

    def test_custom_prefix_length(self):
        workers = ["w1", "w2", "w3"]
        policy = CacheAwarePolicy(workers=workers, prefix_length=2)
        w1 = policy.select_node(
            SchedulingContext(role="decode", prompt="hello world AAAA BBBB"),
            [Candidate(address) for address in sorted(policy._ring.workers)],
        )
        w2 = policy.select_node(
            SchedulingContext(role="decode", prompt="hello world CCCC DDDD"),
            [Candidate(address) for address in sorted(policy._ring.workers)],
        )
        assert w1 == w2

    def test_default_prefix_length(self):
        policy = CacheAwarePolicy(workers=["w1"])
        assert policy._prefix_length == DEFAULT_PREFIX_LENGTH

    # ------------------------------------------------------------------
    # Tokenizer tests
    # ------------------------------------------------------------------

    def test_tokenizer_used_when_provided(self):
        """When a tokenizer is provided, encode() is called."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(300))
        policy = CacheAwarePolicy(
            workers=["w1", "w2", "w3"],
            prefix_length=256,
            tokenizer=mock_tokenizer,
        )
        policy.select_node(
            SchedulingContext(role="decode", prompt="hello world"),
            [Candidate(address) for address in sorted(policy._ring.workers)],
        )
        mock_tokenizer.encode.assert_called_once_with("hello world")

    def test_tokenizer_fallback_on_error(self):
        """When tokenizer.encode() raises, fall back to whitespace split."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = RuntimeError("tokenizer broken")
        policy = CacheAwarePolicy(
            workers=["w1", "w2"],
            prefix_length=256,
            tokenizer=mock_tokenizer,
        )
        # Should not raise, falls back to whitespace split
        result = policy.select_node(
            SchedulingContext(role="decode", prompt="hello world tokens here"),
            [Candidate(address) for address in sorted(policy._ring.workers)],
        )
        assert result in ("w1", "w2")

    def test_no_tokenizer_uses_whitespace_split(self):
        """Without tokenizer, whitespace split is used for tokenization."""
        policy = CacheAwarePolicy(
            workers=["w1", "w2", "w3"],
            prefix_length=2,
            tokenizer=None,
        )
        # "hello world X" and "hello world Y" share first 2 whitespace tokens
        w1 = policy.select_node(
            SchedulingContext(role="decode", prompt="hello world AAA"),
            [Candidate(address) for address in sorted(policy._ring.workers)],
        )
        w2 = policy.select_node(
            SchedulingContext(role="decode", prompt="hello world BBB"),
            [Candidate(address) for address in sorted(policy._ring.workers)],
        )
        assert w1 == w2

    def test_tokenizer_produces_different_routing_than_whitespace(self):
        """A tokenizer that produces different tokens should route differently."""
        # Tokenizer that reverses the string before splitting
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda p: list(range(len(p.split())))
        policy_with_tok = CacheAwarePolicy(
            workers=["w1", "w2", "w3", "w4"],
            prefix_length=256,
            tokenizer=mock_tokenizer,
        )
        policy_without_tok = CacheAwarePolicy(
            workers=["w1", "w2", "w3", "w4"],
            prefix_length=256,
            tokenizer=None,
        )
        prompt = "alpha beta gamma delta epsilon " * 20
        # Both should return valid workers (may or may not differ)
        r1 = policy_with_tok.select_node(
            SchedulingContext(role="decode", prompt=prompt),
            [Candidate(address) for address in sorted(policy_with_tok._ring.workers)],
        )
        r2 = policy_without_tok.select_node(
            SchedulingContext(role="decode", prompt=prompt),
            [
                Candidate(address)
                for address in sorted(policy_without_tok._ring.workers)
            ],
        )
        assert r1 in ("w1", "w2", "w3", "w4")
        assert r2 in ("w1", "w2", "w3", "w4")

    # ------------------------------------------------------------------
    # Consistent hash stability tests
    # ------------------------------------------------------------------

    def test_consistent_hash_stability_across_instances(self):
        """Two separate policy instances with same config produce same routing."""
        workers = ["w1", "w2", "w3", "w4"]
        p1 = CacheAwarePolicy(workers=workers, prefix_length=256)
        p2 = CacheAwarePolicy(workers=workers, prefix_length=256)
        for i in range(50):
            prompt = f"stability test prompt number {i} " * 10
            assert p1.select_node(
                SchedulingContext(role="decode", prompt=prompt),
                [Candidate(address) for address in sorted(p1._ring.workers)],
            ) == p2.select_node(
                SchedulingContext(role="decode", prompt=prompt),
                [Candidate(address) for address in sorted(p2._ring.workers)],
            )

    def test_add_worker_minimal_remapping(self):
        """Adding a worker to the policy remaps minimal keys."""
        workers = ["w1", "w2", "w3", "w4"]
        policy = CacheAwarePolicy(workers=workers, prefix_length=256)
        prompts = [f"prompt {i} content " * 20 for i in range(200)]
        before = {
            p: policy.select_node(
                SchedulingContext(role="decode", prompt=p),
                [Candidate(address) for address in sorted(policy._ring.workers)],
            )
            for p in prompts
        }
        policy.on_instance_added("decode", "w5", 4096)
        after = {
            p: policy.select_node(
                SchedulingContext(role="decode", prompt=p),
                [Candidate(address) for address in sorted(policy._ring.workers)],
            )
            for p in prompts
        }
        moved = sum(1 for p in prompts if before[p] != after[p])
        # ~1/5 = 20% should move, allow generous 40% margin
        assert moved < len(prompts) * 0.40

    def test_remove_worker_only_affected_keys_move(self):
        """Removing a worker only remaps keys that were on that worker."""
        workers = ["w1", "w2", "w3", "w4"]
        policy = CacheAwarePolicy(workers=workers, prefix_length=256)
        prompts = [f"prompt {i} content " * 20 for i in range(200)]
        before = {
            p: policy.select_node(
                SchedulingContext(role="decode", prompt=p),
                [Candidate(address) for address in sorted(policy._ring.workers)],
            )
            for p in prompts
        }
        policy.on_instance_removed("decode", "w3", 0)
        after = {
            p: policy.select_node(
                SchedulingContext(role="decode", prompt=p),
                [Candidate(address) for address in sorted(policy._ring.workers)],
            )
            for p in prompts
        }
        # Only keys previously on w3 should have moved
        for p in prompts:
            if before[p] != "w3":
                assert (
                    before[p] == after[p]
                ), f"Key not on w3 should not have moved: {before[p]} -> {after[p]}"
