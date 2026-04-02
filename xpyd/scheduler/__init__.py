# SPDX-License-Identifier: Apache-2.0
"""Scheduler module for MicroPDProxy."""

try:
    from .scheduler_base import SchedulingPolicy
    from .round_robin import RoundRobinSchedulingPolicy
    from .load_balanced import LoadBalancedScheduler
    from .consistent_hash import ConsistentHashPolicy
    from .power_of_two import PowerOfTwoPolicy
    from .cache_aware import CacheAwarePolicy
    from .policy_registry import PolicyRegistry, default_registry
except ImportError:
    from xpyd.scheduler.scheduler_base import SchedulingPolicy
    from xpyd.scheduler.round_robin import RoundRobinSchedulingPolicy
    from xpyd.scheduler.load_balanced import LoadBalancedScheduler
    from xpyd.scheduler.consistent_hash import ConsistentHashPolicy
    from xpyd.scheduler.power_of_two import PowerOfTwoPolicy
    from xpyd.scheduler.cache_aware import CacheAwarePolicy
    from xpyd.scheduler.policy_registry import PolicyRegistry, default_registry

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
