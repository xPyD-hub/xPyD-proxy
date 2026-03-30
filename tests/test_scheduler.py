# SPDX-License-Identifier: Apache-2.0
"""Unit tests for scheduler module."""

from __future__ import annotations

import itertools
from unittest.mock import patch

import pytest
from scheduler import (
    LoadBalancedScheduler,
    RoundRobinSchedulingPolicy,
    SchedulingPolicy,
)


def test_base_class_is_abstract():
    with pytest.raises(TypeError):
        SchedulingPolicy()


class TestRoundRobin:
    def test_cycles_through_instances(self):
        policy = RoundRobinSchedulingPolicy()
        instances = ["a:1", "b:2", "c:3"]
        cycler = itertools.cycle(instances)
        results = [policy.schedule(cycler) for _ in range(6)]
        assert results == ["a:1", "b:2", "c:3", "a:1", "b:2", "c:3"]

    def test_ignores_request_len_and_max_tokens(self):
        policy = RoundRobinSchedulingPolicy()
        instances = ["x:1", "y:2"]
        cycler = itertools.cycle(instances)
        r1 = policy.schedule(cycler, is_prompt=True, request_len=9999, max_tokens=9999)
        r2 = policy.schedule(cycler, is_prompt=False, request_len=1, max_tokens=1)
        assert r1 == "x:1"
        assert r2 == "y:2"

    def test_schedule_completion_is_noop(self):
        policy = RoundRobinSchedulingPolicy()
        policy.schedule_completion(
            prefill_instance="a:1", decode_instance="b:2", req_len=100
        )


@patch(
    "scheduler.load_balanced.query_instance_model_len",
    return_value=[131072, 131072],
)
class TestLoadBalanced:
    def test_prefill_distributes(self, _mock):
        policy = LoadBalancedScheduler(["p1:1", "p2:2"], ["d1:1", "d2:2"])
        cyc = itertools.cycle(["p1:1", "p2:2"])
        r1 = policy.schedule(cyc, is_prompt=True, request_len=100, max_tokens=1)
        r2 = policy.schedule(cyc, is_prompt=True, request_len=100, max_tokens=1)
        assert {r1, r2} == {"p1:1", "p2:2"}

    def test_decode_distributes(self, _mock):
        policy = LoadBalancedScheduler(["p1:1", "p2:2"], ["d1:1", "d2:2"])
        cyc = itertools.cycle(["d1:1", "d2:2"])
        d1 = policy.schedule(cyc, is_prompt=False, request_len=50, max_tokens=10)
        d2 = policy.schedule(cyc, is_prompt=False, request_len=50, max_tokens=10)
        assert {d1, d2} == {"d1:1", "d2:2"}

    def test_returns_none_when_too_large(self, _mock):
        _mock.return_value = [100, 100]
        policy = LoadBalancedScheduler(["p1:1", "p2:2"], ["d1:1", "d2:2"])
        cyc = itertools.cycle(["p1:1", "p2:2"])
        assert (
            policy.schedule(cyc, is_prompt=True, request_len=90, max_tokens=20) is None
        )

    def test_completion_releases_prefill(self, _mock):
        policy = LoadBalancedScheduler(["p1:1", "p2:2"], ["d1:1", "d2:2"])
        cyc = itertools.cycle(["p1:1", "p2:2"])
        inst = policy.schedule(cyc, is_prompt=True, request_len=100, max_tokens=1)
        policy.schedule_completion(prefill_instance=inst, req_len=100)
        assert all(c == 0 for c in policy.prefill_bs_counter)

    def test_completion_releases_decode(self, _mock):
        policy = LoadBalancedScheduler(["p1:1", "p2:2"], ["d1:1", "d2:2"])
        cyc = itertools.cycle(["d1:1", "d2:2"])
        inst = policy.schedule(cyc, is_prompt=False, request_len=50, max_tokens=10)
        policy.schedule_completion(decode_instance=inst, req_len=50)
        assert all(c == 0 for c in policy.decode_bs_counter)

    def test_idle_resets_counters(self, _mock):
        policy = LoadBalancedScheduler(["p1:1", "p2:2"], ["d1:1", "d2:2"])
        cyc = itertools.cycle(["p1:1", "p2:2"])
        i1 = policy.schedule(cyc, is_prompt=True, request_len=100, max_tokens=1)
        i2 = policy.schedule(cyc, is_prompt=True, request_len=200, max_tokens=1)
        policy.schedule_completion(prefill_instance=i1, req_len=100)
        policy.schedule_completion(prefill_instance=i2, req_len=200)
        assert all(c == 0 for c in policy.prefill_utils_counter)
