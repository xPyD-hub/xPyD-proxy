# SPDX-License-Identifier: Apache-2.0
"""Scheduler module for MicroPDProxy."""

try:
    from .scheduler_base import SchedulingPolicy
    from .round_robin import RoundRobinSchedulingPolicy
    from .load_balanced import LoadBalancedScheduler
except ImportError:
    from scheduler_base import SchedulingPolicy
    from round_robin import RoundRobinSchedulingPolicy
    from load_balanced import LoadBalancedScheduler

__all__ = [
    "SchedulingPolicy",
    "RoundRobinSchedulingPolicy",
    "LoadBalancedScheduler",
]
