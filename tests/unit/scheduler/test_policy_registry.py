# SPDX-License-Identifier: Apache-2.0
"""Tests for the policy registry (Task 10d)."""

import pytest
from scheduler.policy_registry import PolicyRegistry, default_registry
from scheduler.scheduler_base import SchedulingPolicy


class _DummyPolicy(SchedulingPolicy):
    """Minimal concrete policy for testing."""

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def schedule(self, cycler, is_prompt=None, request_len=None, max_tokens=None):
        return None


class TestPolicyRegistry:
    """Unit tests for :class:`PolicyRegistry`."""

    def test_register_and_select(self):
        """Registering a policy allows creating instances by name."""
        reg = PolicyRegistry()
        reg.register("dummy", _DummyPolicy)

        assert reg.has("dummy")
        instance = reg.create("dummy", foo="bar")
        assert isinstance(instance, _DummyPolicy)
        assert instance.kwargs == {"foo": "bar"}

    def test_unknown_policy_raises(self):
        """Creating an unregistered policy raises ValueError."""
        reg = PolicyRegistry()
        with pytest.raises(ValueError, match="Unknown scheduling policy"):
            reg.create("nonexistent")

    def test_builtin_policies_registered(self):
        """The default registry ships with roundrobin and loadbalanced."""
        assert default_registry.has("roundrobin")
        assert default_registry.has("loadbalanced")
        assert "roundrobin" in default_registry.list_policies()
        assert "loadbalanced" in default_registry.list_policies()

    def test_list_policies_sorted(self):
        """list_policies returns names in sorted order."""
        reg = PolicyRegistry()
        reg.register("zebra", _DummyPolicy)
        reg.register("alpha", _DummyPolicy)
        assert reg.list_policies() == ["alpha", "zebra"]

    def test_has_returns_false_for_missing(self):
        """has() returns False for unregistered names."""
        reg = PolicyRegistry()
        assert not reg.has("missing")

    def test_register_non_policy_raises_type_error(self):
        """Registering a class that is not a SchedulingPolicy raises TypeError."""
        reg = PolicyRegistry()
        with pytest.raises(TypeError, match="not a subclass of SchedulingPolicy"):
            reg.register("bad", dict)

    def test_register_non_class_raises_type_error(self):
        """Registering a non-class object raises TypeError."""
        reg = PolicyRegistry()
        with pytest.raises(TypeError, match="not a subclass of SchedulingPolicy"):
            reg.register("bad", "not_a_class")

    def test_duplicate_register_warns(self, caplog):
        """Re-registering the same name logs a warning."""
        reg = PolicyRegistry()
        reg.register("dup", _DummyPolicy)
        with caplog.at_level("WARNING"):
            reg.register("dup", _DummyPolicy)
        assert "Overwriting existing policy" in caplog.text

    def test_create_returns_correct_type(self):
        """create() returns an instance of the registered class."""
        rr = default_registry.create("roundrobin")
        assert isinstance(rr, SchedulingPolicy)

        lb = default_registry.create(
            "loadbalanced", prefill_instances=["p1"], decode_instances=["d1"]
        )
        assert isinstance(lb, SchedulingPolicy)
