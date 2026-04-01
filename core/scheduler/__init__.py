# SPDX-License-Identifier: Apache-2.0
"""Scheduler module for MicroPDProxy."""

try:
    from .scheduler_base import SchedulingPolicy
    from .round_robin import RoundRobinSchedulingPolicy
    from .load_balanced import LoadBalancedScheduler
    from .policy_registry import PolicyRegistry, default_registry
except ImportError:
    from scheduler_base import SchedulingPolicy
    from round_robin import RoundRobinSchedulingPolicy
    from load_balanced import LoadBalancedScheduler
    from policy_registry import PolicyRegistry, default_registry

__all__ = [
    "SchedulingPolicy",
    "RoundRobinSchedulingPolicy",
    "LoadBalancedScheduler",
    "PolicyRegistry",
    "default_registry",
]
