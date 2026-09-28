# SPDX-License-Identifier: Apache-2.0
"""Abstract base class for scheduling policies."""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence

if TYPE_CHECKING:
    from xpyd.registry import InstanceRegistry


InstanceRole = Literal["prefill", "decode", "aggregated"]


@dataclass(frozen=True)
class SchedulingContext:
    role: InstanceRole
    model: str = ""
    request_len: int = 0
    max_tokens: int = 0
    header: Optional[str] = None
    session_id: Optional[str] = None
    user: Optional[str] = None
    client_ip: Optional[str] = None
    prompt: Optional[str] = None


@dataclass(frozen=True)
class Candidate:
    address: str
    active_requests: int = 0


class SchedulingPolicy(ABC):
    """Base class for all scheduling policies."""

    def __init__(
        self,
        registry: Optional["InstanceRegistry"] = None,
        workers: Optional[Sequence[str]] = None,
    ):
        self.lock = threading.Lock()
        self._registry: Optional["InstanceRegistry"] = registry

    @property
    def registry(self) -> Optional["InstanceRegistry"]:
        return self._registry

    @registry.setter
    def registry(self, value: Optional["InstanceRegistry"]) -> None:
        self._registry = value

    @abstractmethod
    def select_node(
        self, context: SchedulingContext, candidates: Sequence[Candidate]
    ) -> Optional[str]:
        """Select only from eligible candidates; reserve strategy-specific load."""
        raise NotImplementedError

    @classmethod
    def from_config(
        cls,
        *,
        prefill_instances: Sequence[str] = (),
        decode_instances: Sequence[str] = (),
        workers: Sequence[str] = (),
        registry: Optional["InstanceRegistry"] = None,
        tokenizer: Any = None,
        **options: Any,
    ) -> "SchedulingPolicy":
        """Construct a policy without requiring callers to know its constructor."""
        return cls(workers=list(workers), registry=registry, **options)

    def on_instance_added(  # noqa: B027 - optional lifecycle hook
        self, role: InstanceRole, address: str, max_model_len: int
    ) -> None:
        """Update policy state before the shared node list is extended."""

    def on_instance_removed(  # noqa: B027 - optional lifecycle hook
        self, role: InstanceRole, address: str, index: int
    ) -> None:
        """Update policy state before the shared node list is shortened."""

    def on_request_finished(  # noqa: B027 - optional lifecycle hook
        self, context: SchedulingContext, address: str
    ) -> None:
        """Release strategy-specific load, independently of success or failure."""
