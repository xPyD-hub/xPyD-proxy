# SPDX-License-Identifier: Apache-2.0
"""Policy registry for extensible scheduling strategies."""

from __future__ import annotations

from typing import Any

from .load_balanced import LoadBalancedScheduler
from .round_robin import RoundRobinSchedulingPolicy
from .scheduler_base import SchedulingPolicy


class PolicyRegistry:
    """Registry and factory for scheduling policies.

    Provides a central place to register, look up, and instantiate
    :class:`SchedulingPolicy` implementations by name.
    """

    def __init__(self) -> None:
        self._policies: dict[str, type[SchedulingPolicy]] = {}

    def register(self, name: str, policy_cls: type) -> None:
        """Register a scheduling policy class under *name*."""
        self._policies[name] = policy_cls

    def create(self, name: str, **kwargs: Any) -> SchedulingPolicy:
        """Create a new instance of the policy registered as *name*.

        Raises:
            ValueError: If *name* has not been registered.
        """
        if name not in self._policies:
            raise ValueError(
                f"Unknown scheduling policy: {name!r}. "
                f"Available: {self.list_policies()}"
            )
        return self._policies[name](**kwargs)

    def has(self, name: str) -> bool:
        """Return whether *name* is a registered policy."""
        return name in self._policies

    def list_policies(self) -> list[str]:
        """Return a sorted list of registered policy names."""
        return sorted(self._policies)


def _build_default_registry() -> PolicyRegistry:
    """Create and populate the default registry with built-in policies."""
    registry = PolicyRegistry()

    # Built-in policies
    registry.register("roundrobin", RoundRobinSchedulingPolicy)
    registry.register("loadbalanced", LoadBalancedScheduler)

    # Future policies — placeholders registered when their classes exist.
    # Uncomment each line once the corresponding module is implemented:
    # registry.register("consistent_hash", ConsistentHashPolicy)
    # registry.register("power_of_two", PowerOfTwoPolicy)
    # registry.register("cache_aware", CacheAwarePolicy)

    return registry


default_registry: PolicyRegistry = _build_default_registry()
"""Module-level default registry with all built-in policies pre-registered."""
