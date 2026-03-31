# SPDX-License-Identifier: Apache-2.0
"""Centralized proxy configuration with validation.

Provides ``ProxyConfig`` as the single source of truth for all proxy
parameters.  Values are resolved in precedence order:

    CLI args  >  environment variables  >  defaults

``ProxyConfig.from_args()`` bridges the existing argparse layer.
"""

from __future__ import annotations

import ipaddress
import os
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class ProxyConfig(BaseModel):
    """Validated proxy configuration."""

    model_config = ConfigDict(extra="forbid")

    model: str
    prefill: List[str] = []
    decode: List[str]
    port: int = 8000
    generator_on_p_node: bool = False
    roundrobin: bool = False
    admin_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("port")
    @classmethod
    def _port_in_range(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"port must be between 1 and 65535, got {v}")
        return v

    @field_validator("prefill", "decode", mode="before")
    @classmethod
    def _coerce_none_to_list(cls, v):
        if v is None:
            return []
        return v

    @field_validator("prefill", "decode")
    @classmethod
    def _validate_instances(cls, instances: List[str]) -> List[str]:
        for inst in instances:
            parts = inst.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid instance format: {inst}")
            host, port_str = parts
            if host != "localhost":
                try:
                    ipaddress.ip_address(host)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid host in instance {inst}: {exc}"
                    ) from exc
            try:
                port = int(port_str)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid port in instance {inst}: {exc}"
                ) from exc
            if not (1 <= port <= 65535):
                raise ValueError(
                    f"Port out of range in instance {inst}"
                )
        return instances

    @model_validator(mode="after")
    def _require_decode(self) -> "ProxyConfig":
        if not self.decode:
            raise ValueError(
                "Please specify at least one decode node."
            )
        return self

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_args(cls, args) -> "ProxyConfig":
        """Build a ``ProxyConfig`` from an ``argparse.Namespace``.

        Environment variables ``ADMIN_API_KEY`` / ``OPENAI_API_KEY`` are
        used as fallbacks when the corresponding values are not supplied
        via CLI.
        """
        return cls(
            model=args.model,
            prefill=getattr(args, "prefill", None),
            decode=getattr(args, "decode", None),
            port=getattr(args, "port", 8000),
            generator_on_p_node=getattr(args, "generator_on_p_node", False),
            roundrobin=getattr(args, "roundrobin", False),
            admin_api_key=os.environ.get("ADMIN_API_KEY"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )
