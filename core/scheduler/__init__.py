# SPDX-License-Identifier: Apache-2.0
"""Scheduler module for MicroPDProxy."""

try:
    from .cache_aware import CacheAwarePolicy
    from .consistent_hash import ConsistentHashPolicy
    from .load_balanced import LoadBalancedScheduler
    from .policy_registry import PolicyRegistry, default_registry
    from .power_of_two import PowerOfTwoPolicy
    from .round_robin import RoundRobinSchedulingPolicy
    from .scheduler_base import SchedulingPolicy
except ImportError:
    from cache_aware import CacheAwarePolicy
    from consistent_hash import ConsistentHashPolicy
    from load_balanced import LoadBalancedScheduler
    from policy_registry import PolicyRegistry, default_registry
    from power_of_two import PowerOfTwoPolicy
    from round_robin import RoundRobinSchedulingPolicy
    from scheduler_base import SchedulingPolicy

__all__ = [
    "SchedulingPolicy",
    "RoundRobinSchedulingPolicy",
    "LoadBalancedScheduler",
    "ConsistentHashPolicy",
    "PowerOfTwoPolicy",
    "CacheAwarePolicy",
    "PolicyRegistry",
    "default_registry",
]
