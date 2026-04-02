# SPDX-License-Identifier: Apache-2.0
"""Unit tests for multi-model routing support."""

import itertools
from unittest.mock import patch

from xpyd.registry import InstanceRegistry


class TestRegistryAddWithModel:
    """test_registry_add_with_model"""

    def test_add_with_model(self):
        reg = InstanceRegistry()
        reg.add("prefill", "10.0.0.1:8000", model="llama-3")
        info = reg.get_instance_info("10.0.0.1:8000")
        assert info.model == "llama-3"
        assert info.role == "prefill"

    def test_add_without_model_defaults_empty(self):
        reg = InstanceRegistry()
        reg.add("decode", "10.0.0.2:8000")
        info = reg.get_instance_info("10.0.0.2:8000")
        assert info.model == ""


class TestRegistryGetByRoleAndModel:
    """test_registry_get_by_role_and_model"""

    def setup_method(self):
        self.reg = InstanceRegistry()
        self.reg.add("decode", "10.0.0.1:9001", model="llama-3")
        self.reg.add("decode", "10.0.0.2:9002", model="llama-3")
        self.reg.add("decode", "10.0.0.3:9003", model="deepseek-r1")
        for addr in ["10.0.0.1:9001", "10.0.0.2:9002", "10.0.0.3:9003"]:
            self.reg.mark_healthy(addr)

    def test_filter_by_model(self):
        result = self.reg.get_available_instances("decode", model="llama-3")
        assert sorted(result) == ["10.0.0.1:9001", "10.0.0.2:9002"]

    def test_no_model_filter_returns_all(self):
        result = self.reg.get_available_instances("decode")
        assert len(result) == 3


class TestRegistryModelIsolation:
    """test_registry_model_isolation"""

    def test_model_a_not_in_model_b(self):
        reg = InstanceRegistry()
        reg.add("prefill", "10.0.0.1:8000", model="llama-3")
        reg.add("prefill", "10.0.0.2:8000", model="deepseek-r1")
        reg.mark_healthy("10.0.0.1:8000")
        reg.mark_healthy("10.0.0.2:8000")

        result_a = reg.get_available_instances("prefill", model="llama-3")
        result_b = reg.get_available_instances("prefill", model="deepseek-r1")
        assert result_a == ["10.0.0.1:8000"]
        assert result_b == ["10.0.0.2:8000"]


class TestRegistryUnknownModelEmpty:
    """test_registry_unknown_model_empty"""

    def test_unknown_model_returns_empty(self):
        reg = InstanceRegistry()
        reg.add("prefill", "10.0.0.1:8000", model="llama-3")
        reg.mark_healthy("10.0.0.1:8000")
        result = reg.get_available_instances("prefill", model="nonexistent")
        assert result == []


class TestRegistryGetRegisteredModels:
    """test_registry_get_registered_models"""

    def test_returns_unique_models(self):
        reg = InstanceRegistry()
        reg.add("prefill", "10.0.0.1:8000", model="llama-3")
        reg.add("prefill", "10.0.0.2:8000", model="llama-3")
        reg.add("decode", "10.0.0.3:8000", model="deepseek-r1")
        reg.add("decode", "10.0.0.4:8000", model="qwen-2")
        models = reg.get_registered_models()
        assert models == ["deepseek-r1", "llama-3", "qwen-2"]

    def test_empty_model_excluded(self):
        reg = InstanceRegistry()
        reg.add("prefill", "10.0.0.1:8000")
        reg.add("decode", "10.0.0.2:8000", model="llama-3")
        models = reg.get_registered_models()
        assert models == ["llama-3"]

    def test_no_instances_returns_empty(self):
        reg = InstanceRegistry()
        assert reg.get_registered_models() == []


class TestSchedulerRoutesByModel:
    """test_scheduler_routes_by_model (mock registry)"""

    def test_load_balanced_routes_by_model(self):
        from xpyd.scheduler.load_balanced import LoadBalancedScheduler

        reg = InstanceRegistry()
        # llama-3 instances
        reg.add("prefill", "10.0.0.1:8000", model="llama-3")
        reg.add("decode", "10.0.0.1:9000", model="llama-3")
        # deepseek instances
        reg.add("prefill", "10.0.0.2:8000", model="deepseek-r1")
        reg.add("decode", "10.0.0.2:9000", model="deepseek-r1")
        for addr in [
            "10.0.0.1:8000",
            "10.0.0.1:9000",
            "10.0.0.2:8000",
            "10.0.0.2:9000",
        ]:
            reg.mark_healthy(addr)

        all_prefill = ["10.0.0.1:8000", "10.0.0.2:8000"]
        all_decode = ["10.0.0.1:9000", "10.0.0.2:9000"]

        with patch(
            "xpyd.scheduler.load_balanced.query_instance_model_len",
            return_value=[131072] * 2,
        ):
            sched = LoadBalancedScheduler(
                all_prefill,
                all_decode,
                registry=reg,
            )

        cycler = itertools.cycle(all_prefill)
        result = sched.schedule(
            cycler,
            is_prompt=True,
            request_len=100,
            max_tokens=100,
            model="llama-3",
        )
        assert result == "10.0.0.1:8000"

        cycler_d = itertools.cycle(all_decode)
        result_d = sched.schedule(
            cycler_d,
            is_prompt=False,
            request_len=100,
            max_tokens=100,
            model="deepseek-r1",
        )
        assert result_d == "10.0.0.2:9000"


class TestSchedulerNoInstanceForModel:
    """test_scheduler_no_instance_for_model"""

    def test_returns_none_for_unknown_model(self):
        from xpyd.scheduler.load_balanced import LoadBalancedScheduler

        reg = InstanceRegistry()
        reg.add("prefill", "10.0.0.1:8000", model="llama-3")
        reg.add("decode", "10.0.0.1:9000", model="llama-3")
        reg.mark_healthy("10.0.0.1:8000")
        reg.mark_healthy("10.0.0.1:9000")

        with patch(
            "xpyd.scheduler.load_balanced.query_instance_model_len",
            return_value=[131072],
        ):
            sched = LoadBalancedScheduler(
                ["10.0.0.1:8000"],
                ["10.0.0.1:9000"],
                registry=reg,
            )

        cycler = itertools.cycle(["10.0.0.1:8000"])
        result = sched.schedule(
            cycler,
            is_prompt=True,
            request_len=100,
            max_tokens=100,
            model="nonexistent",
        )
        assert result is None


class TestSchedulerLoadBalanceWithinModel:
    """test_scheduler_load_balance_within_model

    With 2 decode instances for model A, requests should only go to
    model A instances (not model B), verifying model-based filtering.
    """

    def test_routes_only_to_correct_model_instances(self):
        from xpyd.scheduler.load_balanced import LoadBalancedScheduler

        reg = InstanceRegistry()
        reg.add("decode", "10.0.0.1:9001", model="llama-3")
        reg.add("decode", "10.0.0.2:9002", model="llama-3")
        reg.add("decode", "10.0.0.3:9003", model="deepseek-r1")
        for addr in ["10.0.0.1:9001", "10.0.0.2:9002", "10.0.0.3:9003"]:
            reg.mark_healthy(addr)

        all_decode = ["10.0.0.1:9001", "10.0.0.2:9002", "10.0.0.3:9003"]

        with patch(
            "xpyd.scheduler.load_balanced.query_instance_model_len",
            return_value=[131072] * 3,
        ):
            sched = LoadBalancedScheduler(
                [],
                all_decode,
                registry=reg,
            )

        # All requests with model="llama-3" must only go to llama-3 instances
        for _ in range(20):
            cycler = itertools.cycle(all_decode)
            result = sched.schedule(
                cycler,
                is_prompt=False,
                request_len=100,
                max_tokens=100,
                model="llama-3",
            )
            assert result in ("10.0.0.1:9001", "10.0.0.2:9002")
            # Release the slot
            if result:
                sched.schedule_completion(
                    decode_instance=result,
                    req_len=100,
                )
