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

from topology import expand_topology


class ProxyConfig(BaseModel):
    """Validated proxy configuration."""

    model_config = ConfigDict(extra="forbid")

    model: str
    prefill: List[str] = []
    decode: List[str]
    port: int = 8000
    log_level: str = "warning"
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

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, v: str) -> str:
        valid = {"debug", "info", "warning", "error"}
        if v not in valid:
            raise ValueError(
                f"log_level must be one of {sorted(valid)}, got {v!r}"
            )
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
    # Topology expansion helper
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_node_config(role: str, node_cfg: Any) -> List[str]:
        """Expand a YAML node config into a flat list of host:port strings.

        Accepts either:
        - A list of strings (backward compat): ``["10.0.0.1:8000"]``
        - A topology dict: ``{nodes: [...], tp_size: N, dp_size: M, ...}``
        """
        if isinstance(node_cfg, list):
            return node_cfg
        if isinstance(node_cfg, dict):
            required = {"nodes", "tp_size", "dp_size", "world_size_per_node"}
            missing = required - set(node_cfg.keys())
            if missing:
                raise ValueError(
                    f"{role} topology config missing keys: {sorted(missing)}"
                )
            unknown = set(node_cfg.keys()) - required
            if unknown:
                raise ValueError(
                    f"{role} topology config has unknown keys: {sorted(unknown)}"
                )
            return expand_topology(
                role=role,
                nodes=node_cfg["nodes"],
                tp_size=node_cfg["tp_size"],
                dp_size=node_cfg["dp_size"],
                world_size_per_node=node_cfg["world_size_per_node"],
            )
        raise ValueError(
            f"{role} must be a list of addresses or a topology dict, "
            f"got {type(node_cfg).__name__}"
        )

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
            "log_level": "warning",
        }

        # 1. Load YAML base (if provided)
        yaml_data: Dict[str, Any] = {}
        config_path = getattr(args, "config", None)
        if config_path:
            yaml_data = cls.load_yaml(config_path)

        # 1b. Expand topology-style prefill/decode configs into flat lists
        for role in ("prefill", "decode"):
            if role in yaml_data:
                yaml_data[role] = cls._expand_node_config(role, yaml_data[role])

        # 2. Pop YAML-only keys that don't map directly to ProxyConfig fields
        scheduling = yaml_data.pop("scheduling", None)
        admin_api_key_yaml = yaml_data.pop("admin_api_key", None)
        openai_api_key_yaml = yaml_data.pop("openai_api_key", None)

        # 3. Reject unknown YAML keys early
        known_fields = set(_arg_defaults.keys())
        unknown = set(yaml_data.keys()) - known_fields
        if unknown:
            raise ValueError(
                f"Unknown keys in YAML config: {sorted(unknown)}"
            )

        # 4. Build merged dict: YAML first, then CLI overrides.
        #    NOTE: argparse cannot distinguish "user explicitly passed the
        #    default" from "argparse filled in the default". So if a user
        #    passes e.g. `--port 8000` and the YAML has `port: 9000`, the
        #    YAML value wins. This is a known argparse limitation.
        merged: Dict[str, Any] = {}
        for field_name, default in _arg_defaults.items():
            cli_value = getattr(args, field_name, default)
            if cli_value != default:
                merged[field_name] = cli_value
            elif field_name in yaml_data:
                merged[field_name] = yaml_data[field_name]

        # 5. scheduling → roundrobin mapping
        _VALID_SCHEDULING = {"roundrobin", "loadbalanced"}
        if scheduling is not None:
            if scheduling not in _VALID_SCHEDULING:
                raise ValueError(
                    f"Invalid scheduling value {scheduling!r}; "
                    f"expected one of {sorted(_VALID_SCHEDULING)}"
                )
            if "roundrobin" not in merged:
                merged["roundrobin"] = scheduling == "roundrobin"

        # 6. Environment variables override YAML for api keys.
        #    Use `is not None` so an explicit empty-string env var
        #    still takes precedence over the YAML value.
        admin_env = os.environ.get("ADMIN_API_KEY")
        admin_key = admin_env if admin_env is not None else admin_api_key_yaml
        openai_env = os.environ.get("OPENAI_API_KEY")
        openai_key = openai_env if openai_env is not None else openai_api_key_yaml
        if admin_key:
            merged.setdefault("admin_api_key", admin_key)
        if openai_key:
            merged.setdefault("openai_api_key", openai_key)

        return cls(**merged)
