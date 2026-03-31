# SPDX-License-Identifier: Apache-2.0
"""Centralized proxy configuration with validation.

Provides ``ProxyConfig`` as the single source of truth for all proxy
parameters.  Values are resolved in precedence order:

    CLI args  >  environment variables  >  YAML config  >  defaults

``ProxyConfig.from_args()`` bridges the existing argparse layer and
optionally merges a YAML config file when ``--config`` is provided.
"""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
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
    # YAML loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_yaml(path: str | Path) -> Dict[str, Any]:
        """Load and return a YAML config file as a dict.

        Raises ``FileNotFoundError`` if the file does not exist and
        ``ValueError`` for invalid YAML content.
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(config_path) as fh:
            try:
                data = yaml.safe_load(fh)
            except yaml.YAMLError as exc:
                raise ValueError(f"Malformed YAML in {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(
                f"YAML config must be a mapping, got {type(data).__name__}"
            )
        return data

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_args(cls, args) -> "ProxyConfig":
        """Build a ``ProxyConfig`` from an ``argparse.Namespace``,
        optionally merging a YAML config file.

        Precedence: CLI args > environment variables > YAML > defaults.
        """
        # Argparse defaults — used to detect whether user explicitly set a value.
        _arg_defaults: Dict[str, Any] = {
            "model": None,
            "prefill": None,
            "decode": None,
            "port": 8000,
            "generator_on_p_node": False,
            "roundrobin": False,
        }

        # 1. Load YAML base (if provided)
        yaml_data: Dict[str, Any] = {}
        config_path = getattr(args, "config", None)
        if config_path:
            yaml_data = cls.load_yaml(config_path)

        # 2. Build merged dict: YAML first, then CLI overrides
        merged: Dict[str, Any] = {}
        for field_name, default in _arg_defaults.items():
            cli_value = getattr(args, field_name, default)
            # CLI value is considered "set" if it differs from the
            # argparse default (i.e. the user explicitly passed it).
            if cli_value != default:
                merged[field_name] = cli_value
            elif field_name in yaml_data:
                merged[field_name] = yaml_data[field_name]
            # If neither CLI nor YAML provided a value, omit from merged
            # so Pydantic uses its own default (or raises for required fields).

        # Pop YAML-only keys that don't map directly to ProxyConfig fields
        scheduling = yaml_data.pop("scheduling", None)
        admin_api_key_yaml = yaml_data.pop("admin_api_key", None)
        openai_api_key_yaml = yaml_data.pop("openai_api_key", None)

        # Reject unknown YAML keys early
        known_fields = set(_arg_defaults.keys())
        unknown = set(yaml_data.keys()) - known_fields
        if unknown:
            raise ValueError(
                f"Unknown keys in YAML config: {sorted(unknown)}"
            )

        # scheduling → roundrobin mapping
        if scheduling is not None and "roundrobin" not in merged:
            merged["roundrobin"] = scheduling == "roundrobin"

        # 3. Environment variables override YAML for api keys
        admin_key = os.environ.get("ADMIN_API_KEY") or admin_api_key_yaml
        openai_key = os.environ.get("OPENAI_API_KEY") or openai_api_key_yaml
        if admin_key:
            merged.setdefault("admin_api_key", admin_key)
        if openai_key:
            merged.setdefault("openai_api_key", openai_key)

        return cls(**merged)
