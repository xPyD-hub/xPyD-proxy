# SPDX-License-Identifier: Apache-2.0
"""Scheduler module for xPyD proxy."""

from xpyd.scheduler.cache_aware import CacheAwarePolicy
from xpyd.scheduler.consistent_hash import ConsistentHashPolicy
from xpyd.scheduler.load_balanced import LoadBalancedScheduler
from xpyd.scheduler.policy_registry import PolicyRegistry, default_registry
from xpyd.scheduler.power_of_two import PowerOfTwoPolicy
from xpyd.scheduler.round_robin import RoundRobinSchedulingPolicy
from xpyd.scheduler.scheduler_base import SchedulingPolicy

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
