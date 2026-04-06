# SPDX-License-Identifier: Apache-2.0
"""Tests for Power of Two Choices scheduling policy."""

from collections import Counter

from xpyd.scheduler.power_of_two import PowerOfTwoPolicy


class TestPowerOfTwoSelect:
    """Test the core selection logic."""

    def test_picks_least_loaded(self):
        policy = PowerOfTwoPolicy(workers=["w1", "w2", "w3"])
        policy.set_load("w1", 10)
        policy.set_load("w2", 2)
        policy.set_load("w3", 5)
        counts = Counter(policy.select() for _ in range(1000))
        assert counts["w2"] > counts["w1"]

    def test_random_pair_selection(self):
        policy = PowerOfTwoPolicy(workers=["w1", "w2", "w3", "w4"])
        pairs_seen = set()
        for _ in range(1000):
            policy.select()
            pairs_seen.add(tuple(sorted(policy.last_pair)))
        assert len(pairs_seen) > 1

    def test_no_workers_returns_none(self):
        policy = PowerOfTwoPolicy(workers=[])
        assert policy.select() is None

    def test_single_worker(self):
        policy = PowerOfTwoPolicy(workers=["w1"])
        assert policy.select() == "w1"
        assert policy.last_pair == ("w1",)

    def test_two_workers_picks_less_loaded(self):
        policy = PowerOfTwoPolicy(workers=["w1", "w2"])
        policy.set_load("w1", 100)
        policy.set_load("w2", 0)
        # First select should always pick w2 (0 < 100)
        result = policy.select()
        assert result == "w2"

    def test_equal_load_distributes(self):
        policy = PowerOfTwoPolicy(workers=["w1", "w2", "w3"])
        # All load 0 → should distribute (tie-breaking picks first in pair)
        counts = Counter(policy.select() for _ in range(1000))
        # All workers should get some traffic
        assert len(counts) > 1


class TestWorkerManagement:
    """Test add/remove worker operations."""

    def test_add_worker(self):
        policy = PowerOfTwoPolicy(workers=["w1"])
        policy.add_worker("w2")
        assert policy.get_load("w2") == 0
        # Should now be able to select from both
        results = {policy.select() for _ in range(100)}
        assert len(results) == 2

    def test_add_duplicate_worker(self):
        policy = PowerOfTwoPolicy(workers=["w1", "w2"])
        policy.add_worker("w1")
        # Should not duplicate
        with policy.lock:
            assert policy._workers.count("w1") == 1

    def test_remove_worker(self):
        policy = PowerOfTwoPolicy(workers=["w1", "w2", "w3"])
        policy.remove_worker("w2")
        results = {policy.select() for _ in range(100)}
        assert "w2" not in results

    def test_remove_nonexistent_worker(self):
        policy = PowerOfTwoPolicy(workers=["w1"])
        policy.remove_worker("w99")  # Should not raise


class TestLoadTracking:
    """Test load tracking and schedule_completion."""

    def test_set_and_get_load(self):
        policy = PowerOfTwoPolicy(workers=["w1"])
        policy.set_load("w1", 42)
        assert policy.get_load("w1") == 42

    def test_get_load_unknown_worker(self):
        policy = PowerOfTwoPolicy(workers=["w1"])
        assert policy.get_load("unknown") == 0

    def test_schedule_completion_decrements(self):
        policy = PowerOfTwoPolicy(workers=["w1"])
        policy.set_load("w1", 5)
        policy.schedule_completion(decode_instance="w1")
        assert policy.get_load("w1") == 4

    def test_schedule_completion_no_negative(self):
        policy = PowerOfTwoPolicy(workers=["w1"])
        policy.set_load("w1", 0)
        policy.schedule_completion(decode_instance="w1")
        assert policy.get_load("w1") == 0


class TestScheduleInterface:
    """Test the SchedulingPolicy.schedule() bridge."""

    def test_schedule_without_registry(self):
        policy = PowerOfTwoPolicy(workers=["w1", "w2"])
        import itertools

        cycler = itertools.cycle(["w1", "w2"])
        result = policy.schedule(cycler)
        assert result in ("w1", "w2")

    def test_schedule_no_workers(self):
        policy = PowerOfTwoPolicy(workers=[])
        import itertools

        cycler = itertools.cycle([])
        assert policy.schedule(cycler) is None


class TestLoadAutoIncrement:
    """Verify select() increments load so power-of-two is truly load-aware."""

    def test_select_increments_load(self):
        policy = PowerOfTwoPolicy(workers=["w1"])
        assert policy.get_load("w1") == 0
        policy.select()
        assert policy.get_load("w1") == 1
        policy.select()
        assert policy.get_load("w1") == 2

    def test_select_then_completion_balances(self):
        policy = PowerOfTwoPolicy(workers=["w1", "w2"])
        # Select once — one worker gets load 1
        selected = policy.select()
        assert policy.get_load(selected) == 1
        # Complete it
        policy.schedule_completion(decode_instance=selected)
        assert policy.get_load(selected) == 0
