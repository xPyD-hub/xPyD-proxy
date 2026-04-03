# SPDX-License-Identifier: Apache-2.0
"""Unit tests for dual-role completion fast path."""

from __future__ import annotations

from unittest.mock import MagicMock

from xpyd.registry import InstanceRegistry


class TestIsDualModel:
    """Test _is_dual_model detection."""

    def test_dual_model_detected(self):
        proxy = MagicMock()
        proxy.dual_instances = {"qwen-2": ["10.0.0.1:8000"]}
        from xpyd.proxy import Proxy

        assert Proxy._is_dual_model(proxy, "qwen-2") is True

    def test_pd_model_not_dual(self):
        proxy = MagicMock()
        proxy.dual_instances = {}
        from xpyd.proxy import Proxy

        assert Proxy._is_dual_model(proxy, "llama-3") is False

    def test_empty_dual_list_not_dual(self):
        proxy = MagicMock()
        proxy.dual_instances = {"qwen-2": []}
        from xpyd.proxy import Proxy

        assert Proxy._is_dual_model(proxy, "qwen-2") is False


class TestScheduleDual:
    """Test schedule_dual picks from registry."""

    def setup_method(self):
        self.reg = InstanceRegistry()
        self.reg.add("dual", "10.0.0.1:8000", model="qwen-2")
        self.reg.add("dual", "10.0.0.2:8000", model="qwen-2")
        self.reg.mark_healthy("10.0.0.1:8000")
        self.reg.mark_healthy("10.0.0.2:8000")

    def test_schedule_dual_returns_instance(self):
        from xpyd.scheduler import RoundRobinSchedulingPolicy

        policy = RoundRobinSchedulingPolicy(registry=self.reg)
        proxy = MagicMock()
        proxy.dual_instances = {"qwen-2": ["10.0.0.1:8000", "10.0.0.2:8000"]}
        proxy.registry = self.reg
        proxy.scheduling_policy = policy
        proxy.model_schedulers = {}
        proxy._dual_rr_counters = {}
        from xpyd.proxy import Proxy

        result = Proxy.schedule_dual(proxy, "qwen-2")
        assert result in ("10.0.0.1:8000", "10.0.0.2:8000")

    def test_schedule_dual_no_model(self):
        proxy = MagicMock()
        proxy.dual_instances = {}
        proxy.registry = self.reg
        proxy._dual_rr_counters = {}
        from xpyd.proxy import Proxy

        result = Proxy.schedule_dual(proxy, "nonexistent")
        assert result is None


class TestScheduleDualCompletion:
    """Test single load accounting for dual."""

    def test_single_accounting(self):
        registry = MagicMock()
        proxy = MagicMock()
        proxy.registry = registry
        from xpyd.proxy import Proxy

        Proxy.schedule_dual_completion(proxy, "10.0.0.1:8000", req_len=100)
        registry.decrement_active_requests.assert_called_once_with(
            "10.0.0.1:8000",
        )

    def test_no_registry(self):
        proxy = MagicMock()
        proxy.registry = None
        from xpyd.proxy import Proxy

        # Should not raise even without registry
        Proxy.schedule_dual_completion(proxy, "10.0.0.1:8000", req_len=100)


class TestPerModelSchedulerDispatch:
    """Test per-model scheduler fallback chain."""

    def setup_method(self):
        self.reg = InstanceRegistry()
        self.reg.add("dual", "10.0.0.1:8000", model="qwen-2")
        self.reg.add("dual", "10.0.0.2:8000", model="qwen-2")
        self.reg.mark_healthy("10.0.0.1:8000")
        self.reg.mark_healthy("10.0.0.2:8000")

    def test_explicit_loadbalanced_picks_lowest_load(self):
        """model_schedulers={'qwen-2': 'loadbalanced'} uses load-balanced."""
        from xpyd.proxy import Proxy
        from xpyd.scheduler import RoundRobinSchedulingPolicy

        # Global policy is RR, but model-level overrides to loadbalanced
        policy = RoundRobinSchedulingPolicy(registry=self.reg)
        proxy = MagicMock()
        proxy.dual_instances = {"qwen-2": ["10.0.0.1:8000", "10.0.0.2:8000"]}
        proxy.registry = self.reg
        proxy.scheduling_policy = policy
        proxy.model_schedulers = {"qwen-2": "loadbalanced"}
        proxy._dual_rr_counters = {}
        proxy._schedule_dual_load_balanced = (
            lambda available: Proxy._schedule_dual_load_balanced(proxy, available)
        )

        # Give instance 1 more load
        self.reg.increment_active_requests("10.0.0.1:8000")
        self.reg.increment_active_requests("10.0.0.1:8000")

        result = Proxy.schedule_dual(proxy, "qwen-2")
        # Should pick instance 2 (lower load)
        assert result == "10.0.0.2:8000"

    def test_explicit_roundrobin_cycles(self):
        """model_schedulers={'qwen-2': 'roundrobin'} uses round-robin."""
        from xpyd.proxy import Proxy
        from xpyd.scheduler import LoadBalancedScheduler

        # Global is loadbalanced, but model overrides to roundrobin
        policy = LoadBalancedScheduler(
            prefill_instances=[],
            decode_instances=[],
            registry=self.reg,
        )
        proxy = MagicMock()
        proxy.dual_instances = {"qwen-2": ["10.0.0.1:8000", "10.0.0.2:8000"]}
        proxy.registry = self.reg
        proxy.scheduling_policy = policy
        proxy.model_schedulers = {"qwen-2": "roundrobin"}
        proxy._dual_rr_counters = {}

        results = []
        for _ in range(4):
            r = Proxy.schedule_dual(proxy, "qwen-2")
            results.append(r)
        # Should alternate between the two instances
        assert results[0] != results[1]
        assert results[0] == results[2]

    def test_fallback_to_global_loadbalanced(self):
        """No model-level scheduler, global is LoadBalanced → load-balanced."""
        from xpyd.proxy import Proxy
        from xpyd.scheduler import LoadBalancedScheduler

        policy = LoadBalancedScheduler(
            prefill_instances=[],
            decode_instances=[],
            registry=self.reg,
        )
        proxy = MagicMock()
        proxy.dual_instances = {"qwen-2": ["10.0.0.1:8000", "10.0.0.2:8000"]}
        proxy.registry = self.reg
        proxy.scheduling_policy = policy
        proxy.model_schedulers = {}
        proxy._dual_rr_counters = {}
        proxy._schedule_dual_load_balanced = (
            lambda available: Proxy._schedule_dual_load_balanced(proxy, available)
        )

        self.reg.increment_active_requests("10.0.0.1:8000")
        result = Proxy.schedule_dual(proxy, "qwen-2")
        assert result == "10.0.0.2:8000"

    def test_fallback_global_rr_uses_rr(self):
        """No model-level, global is RR → uses round-robin (not load_balanced)."""
        from xpyd.proxy import Proxy
        from xpyd.scheduler import RoundRobinSchedulingPolicy

        policy = RoundRobinSchedulingPolicy(registry=self.reg)
        proxy = MagicMock()
        proxy.dual_instances = {"qwen-2": ["10.0.0.1:8000", "10.0.0.2:8000"]}
        proxy.registry = self.reg
        proxy.scheduling_policy = policy
        proxy.model_schedulers = {}
        proxy._dual_rr_counters = {}

        results = []
        for _ in range(4):
            r = Proxy.schedule_dual(proxy, "qwen-2")
            results.append(r)
        # Round-robin: alternates
        assert results[0] != results[1]
        assert results[0] == results[2]
