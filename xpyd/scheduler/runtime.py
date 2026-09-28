# SPDX-License-Identifier: Apache-2.0
"""Shared candidate filtering and request reservations for all topologies."""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Optional, Sequence

from xpyd.registry import InstanceRegistry
from xpyd.scheduler.scheduler_base import Candidate, SchedulingContext, SchedulingPolicy


class Reservation:
    """Own one scheduling decision and release it at most once."""

    def __init__(
        self,
        runtime: Scheduler,
        policy: SchedulingPolicy,
        context: SchedulingContext,
        address: str,
    ):
        self.address = address
        self.context = context
        self._runtime = runtime
        self._policy = policy
        self._released = False

    def release(self) -> None:
        with self._runtime.lock:
            if self._released:
                return
            self._policy.on_request_finished(self.context, self.address)
            if self._runtime.registry is not None:
                self._runtime.registry.decrement_active_requests(self.address)
            self._runtime._active[self.address] -= 1
            if self._runtime._active[self.address] == 0:
                del self._runtime._active[self.address]
            self._released = True


class Scheduler:
    """Serialize selection/accounting, leaving algorithm state to the policy."""

    def __init__(self, registry: Optional[InstanceRegistry] = None):
        self.registry = registry
        self.lock = threading.RLock()
        self._active: dict[str, int] = defaultdict(int)
        self._legacy: dict[tuple[str, str, int], deque[Reservation]] = {}

    def reserve(
        self,
        policy: SchedulingPolicy,
        context: SchedulingContext,
        addresses: Sequence[str],
    ) -> Optional[Reservation]:
        with self.lock:
            available = (
                set(self.registry.get_available_instances(context.role, context.model))
                if self.registry is not None
                else set(addresses)
            )
            candidates = [
                Candidate(
                    address,
                    (
                        self.registry.get_active_requests(address)
                        if self.registry is not None
                        else self._active.get(address, 0)
                    ),
                )
                for address in dict.fromkeys(addresses)
                if address in available
            ]
            if not candidates:
                return None
            address = policy.select_node(context, candidates)
            if address is None:
                return None
            if address not in {candidate.address for candidate in candidates}:
                raise ValueError(f"Scheduler selected ineligible instance {address!r}")
            if self.registry is not None:
                self.registry.increment_active_requests(address)
            self._active[address] += 1
            return Reservation(self, policy, context, address)

    def remember(self, reservation: Optional[Reservation]) -> Optional[str]:
        """Adapt the legacy address-returning Proxy scheduling API."""
        if reservation is None:
            return None
        key = (
            reservation.context.role,
            reservation.address,
            reservation.context.request_len,
        )
        with self.lock:
            self._legacy.setdefault(key, deque()).append(reservation)
        return reservation.address

    def finish(self, role: str, address: str, request_len: int) -> None:
        with self.lock:
            key = (role, address, request_len)
            pending = self._legacy.get(key)
            if not pending:
                raise ValueError(f"No scheduling reservation for {key!r}")
            pending[0].release()
            pending.popleft()
            if not pending:
                del self._legacy[key]
