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
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from xpyd.resilience import ResilienceConfig
from xpyd.topology import expand_topology


class CircuitBreakerConfig(BaseModel):
    """Configuration for the per-instance circuit breaker."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_duration_seconds: int = 30
    window_duration_seconds: int = 60


class HealthCheckConfig(BaseModel):
    """Health check configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    interval_seconds: float = 10.0
    timeout_seconds: float = 3.0


class InstanceEntry(BaseModel):
    """Per-instance entry for multi-model configuration."""

    model_config = ConfigDict(extra="forbid")

    address: str
    role: str
    model: str = ""  # empty = auto-detect via discovery

    @field_validator("role")
    @classmethod
    def _valid_role(cls, v: str) -> str:
        if v not in ("prefill", "decode"):
            raise ValueError(f"role must be 'prefill' or 'decode', got {v!r}")
        return v

    @field_validator("address")
    @classmethod
    def _valid_address(cls, v: str) -> str:
        parts = v.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid instance format: {v}")
        host, port_str = parts
        if host != "localhost":
            try:
                ipaddress.ip_address(host)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid host in instance {v}: {exc}"
                ) from exc
        try:
            port = int(port_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid port in instance {v}: {exc}"
            ) from exc
        if not (1 <= port <= 65535):
            raise ValueError(f"Port out of range in instance {v}")
        return v


class ProxyConfig(BaseModel):
    """Validated proxy configuration."""

    model_config = ConfigDict(extra="forbid")

    model: str = ""
    prefill: List[str] = []
    decode: List[str] = []
    instances: Optional[List[InstanceEntry]] = None
    models: Optional[List[Dict[str, Any]]] = None
    port: int = 8000
    log_level: str = "warning"
    generator_on_p_node: bool = False
    roundrobin: bool = False
    scheduling: str = "loadbalanced"
    scheduling_config: Dict[str, Any] = Field(default_factory=dict)
    admin_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    wait_timeout_seconds: int = 600
    probe_interval_seconds: int = 10
    health_check: HealthCheckConfig = HealthCheckConfig()
    circuit_breaker: CircuitBreakerConfig = CircuitBreakerConfig()
    retry: ResilienceConfig = ResilienceConfig()

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
        # Multi-model config: instances or models field provides everything
        if self.instances is not None or self.models is not None:
            # Reject mixing multi-model with legacy prefill/decode lists
            if self.prefill or self.decode:
                raise ValueError(
                    "Cannot use 'instances' or 'models' together with "
                    "legacy 'prefill'/'decode' lists."
                )
            # Still require at least one decode entry
            if self.instances is not None:
                has_decode = any(
                    e.role == "decode" for e in self.instances
                )
                if not has_decode:
                    raise ValueError(
                        "Please specify at least one decode node "
                        "in the instances list."
                    )
            return self
        if not self.decode:
            raise ValueError(
                "Please specify at least one decode node."
            )
        if not self.model:
            raise ValueError(
                "Please specify a model name (required in single-model mode)."
            )
        return self

    @model_validator(mode="after")
    def _expand_models_to_instances(self) -> "ProxyConfig":
        """Expand the 'models' shorthand into the 'instances' list."""
        if self.models is not None:
            if self.instances is not None:
                raise ValueError(
                    "Cannot specify both 'models' and 'instances'."
                )
            expanded: List[InstanceEntry] = []
            for entry in self.models:
                name = entry.get("name", "")
                if not name:
                    raise ValueError(
                        "Each model in 'models' must have a non-empty 'name'."
                    )
                for addr in entry.get("prefill", []):
                    expanded.append(InstanceEntry(
                        address=addr, role="prefill", model=name,
                    ))
                for addr in entry.get("decode", []):
                    expanded.append(InstanceEntry(
                        address=addr, role="decode", model=name,
                    ))
            self.instances = expanded
            self.models = None  # consumed
            # Validate at least one decode entry after expansion
            if not any(e.role == "decode" for e in expanded):
                raise ValueError(
                    "Please specify at least one decode node "
                    "in the models config."
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
        """Build a ``ProxyConfig`` from an ``argparse.Namespace``.

        .. deprecated::
            This method is retained for backward compatibility with existing
            test infrastructure.  Production code should use
            :meth:`from_yaml` instead.

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
            "wait_timeout_seconds": 600,
            "probe_interval_seconds": 10,
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

        # 1c. Flatten nested 'startup' section into top-level keys
        startup = yaml_data.pop("startup", None)
        if startup is not None and not isinstance(startup, dict):
            raise ValueError(
                f"'startup' section must be a mapping, got {type(startup).__name__}"
            )
        if isinstance(startup, dict):
            for key in ("wait_timeout_seconds", "probe_interval_seconds"):
                if key in startup:
                    yaml_data[key] = startup[key]
            unknown_startup = set(startup.keys()) - {
                "wait_timeout_seconds",
                "probe_interval_seconds",
            }
            if unknown_startup:
                raise ValueError(
                    f"Unknown keys in startup config: {sorted(unknown_startup)}"
                )

        # 1d. Handle nested 'health_check' section
        health_check_raw = yaml_data.pop("health_check", None)
        if health_check_raw is not None:
            if not isinstance(health_check_raw, dict):
                raise ValueError(
                    f"'health_check' must be a mapping, "
                    f"got {type(health_check_raw).__name__}"
                )
            # Validate via pydantic model (rejects unknown keys)
            health_cfg = HealthCheckConfig(**health_check_raw)
            yaml_data["health_check"] = health_cfg

        # 1e. Handle nested 'retry' section
        retry_raw = yaml_data.pop("retry", None)
        if retry_raw is not None:
            if not isinstance(retry_raw, dict):
                raise ValueError(
                    f"'retry' must be a mapping, "
                    f"got {type(retry_raw).__name__}"
                )

        # 2. Pop YAML-only keys that don't map directly to ProxyConfig fields
        scheduling = yaml_data.pop("scheduling", None)
        admin_api_key_yaml = yaml_data.pop("admin_api_key", None)
        openai_api_key_yaml = yaml_data.pop("openai_api_key", None)
        circuit_breaker_yaml = yaml_data.pop("circuit_breaker", None)

        # 2b. Pop strategy-specific config sections
        _STRATEGY_NAMES = {
            "consistent_hash", "power_of_two", "cache_aware",
        }
        scheduling_config: Dict[str, Any] = {}
        for strategy_name in _STRATEGY_NAMES:
            strategy_section = yaml_data.pop(strategy_name, None)
            if strategy_section is not None:
                if not isinstance(strategy_section, dict):
                    raise ValueError(
                        f"'{strategy_name}' must be a mapping, "
                        f"got {type(strategy_section).__name__}"
                    )
                scheduling_config[strategy_name] = strategy_section

        # 3. Reject unknown YAML keys early
        known_fields = set(_arg_defaults.keys()) | {
            "health_check", "instances", "models",
            "scheduling", "scheduling_config",
            "circuit_breaker", "retry",
        }
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

        # 5. scheduling → roundrobin mapping + new policy support
        _VALID_SCHEDULING = {
            "roundrobin", "loadbalanced",
            "consistent_hash", "power_of_two", "cache_aware",
        }
        if scheduling is not None:
            if scheduling not in _VALID_SCHEDULING:
                raise ValueError(
                    f"Invalid scheduling value {scheduling!r}; "
                    f"expected one of {sorted(_VALID_SCHEDULING)}"
                )
            if "roundrobin" not in merged:
                merged["roundrobin"] = scheduling == "roundrobin"
            merged["scheduling"] = scheduling
        else:
            # Default: if roundrobin flag is set use it, otherwise
            # loadbalanced
            if merged.get("roundrobin"):
                merged["scheduling"] = "roundrobin"

        # Attach strategy-specific config
        if scheduling_config:
            merged["scheduling_config"] = scheduling_config

        # 6. Environment variables override YAML for api keys.
        #    Use `is not None` so an explicit empty-string env var
        #    still takes precedence over the YAML value.
        admin_env = os.environ.get("ADMIN_API_KEY")
        admin_key = admin_env if admin_env is not None else admin_api_key_yaml
        openai_env = os.environ.get("OPENAI_API_KEY")
        openai_key = openai_env if openai_env is not None else openai_api_key_yaml
        if admin_key is not None:
            merged.setdefault("admin_api_key", admin_key)
        if openai_key is not None:
            merged.setdefault("openai_api_key", openai_key)

        # Pass through health_check if present in yaml_data
        if "health_check" in yaml_data:
            merged.setdefault("health_check", yaml_data["health_check"])

        # 7. Circuit breaker config (YAML only, no CLI override)
        if circuit_breaker_yaml is not None:
            merged["circuit_breaker"] = CircuitBreakerConfig(
                **circuit_breaker_yaml
            )

        # 8. Retry config (YAML only, no CLI override)
        if retry_raw is not None:
            merged["retry"] = ResilienceConfig(**retry_raw)

        # 9. Multi-model config (YAML only, no CLI override)
        if "instances" in yaml_data:
            merged["instances"] = yaml_data["instances"]
        if "models" in yaml_data:
            merged["models"] = yaml_data["models"]

        return cls(**merged)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProxyConfig":
        """Build config from YAML file only."""
        yaml_data = cls.load_yaml(path)

        # Expand topology-style prefill/decode configs into flat lists
        for role in ("prefill", "decode"):
            if role in yaml_data:
                yaml_data[role] = cls._expand_node_config(role, yaml_data[role])

        # Flatten nested 'startup' section into top-level keys
        startup = yaml_data.pop("startup", None)
        if startup is not None and not isinstance(startup, dict):
            raise ValueError(
                f"'startup' section must be a mapping, got {type(startup).__name__}"
            )
        if isinstance(startup, dict):
            for key in ("wait_timeout_seconds", "probe_interval_seconds"):
                if key in startup:
                    yaml_data[key] = startup[key]
            unknown_startup = set(startup.keys()) - {
                "wait_timeout_seconds",
                "probe_interval_seconds",
            }
            if unknown_startup:
                raise ValueError(
                    f"Unknown keys in startup config: {sorted(unknown_startup)}"
                )

        # Handle nested 'health_check' section
        health_check_raw = yaml_data.pop("health_check", None)
        if health_check_raw is not None:
            if not isinstance(health_check_raw, dict):
                raise ValueError(
                    f"'health_check' must be a mapping, "
                    f"got {type(health_check_raw).__name__}"
                )
            yaml_data["health_check"] = HealthCheckConfig(**health_check_raw)

        # Handle nested 'retry' section
        retry_raw = yaml_data.pop("retry", None)
        if retry_raw is not None:
            if not isinstance(retry_raw, dict):
                raise ValueError(
                    f"'retry' must be a mapping, "
                    f"got {type(retry_raw).__name__}"
                )
            yaml_data["retry"] = ResilienceConfig(**retry_raw)

        # Handle nested 'circuit_breaker' section
        circuit_breaker_raw = yaml_data.pop("circuit_breaker", None)
        if circuit_breaker_raw is not None:
            if not isinstance(circuit_breaker_raw, dict):
                raise ValueError(
                    f"'circuit_breaker' must be a mapping, "
                    f"got {type(circuit_breaker_raw).__name__}"
                )
            yaml_data["circuit_breaker"] = CircuitBreakerConfig(
                **circuit_breaker_raw
            )

        # Handle scheduling and strategy-specific config sections
        _VALID_SCHEDULING = {
            "roundrobin", "loadbalanced",
            "consistent_hash", "power_of_two", "cache_aware",
        }
        scheduling = yaml_data.pop("scheduling", None)
        _STRATEGY_NAMES = {
            "consistent_hash", "power_of_two", "cache_aware",
        }
        scheduling_config: Dict[str, Any] = {}
        for strategy_name in _STRATEGY_NAMES:
            strategy_section = yaml_data.pop(strategy_name, None)
            if strategy_section is not None:
                if not isinstance(strategy_section, dict):
                    raise ValueError(
                        f"'{strategy_name}' must be a mapping, "
                        f"got {type(strategy_section).__name__}"
                    )
                scheduling_config[strategy_name] = strategy_section

        if scheduling is not None:
            if scheduling not in _VALID_SCHEDULING:
                raise ValueError(
                    f"Invalid scheduling value {scheduling!r}; "
                    f"expected one of {sorted(_VALID_SCHEDULING)}"
                )
            yaml_data["scheduling"] = scheduling
            yaml_data["roundrobin"] = scheduling == "roundrobin"

        if scheduling_config:
            yaml_data["scheduling_config"] = scheduling_config

        # Environment variables override YAML for api keys
        admin_api_key_yaml = yaml_data.pop("admin_api_key", None)
        openai_api_key_yaml = yaml_data.pop("openai_api_key", None)
        admin_env = os.environ.get("ADMIN_API_KEY")
        admin_key = admin_env if admin_env is not None else admin_api_key_yaml
        openai_env = os.environ.get("OPENAI_API_KEY")
        openai_key = openai_env if openai_env is not None else openai_api_key_yaml
        if admin_key is not None:
            yaml_data["admin_api_key"] = admin_key
        if openai_key is not None:
            yaml_data["openai_api_key"] = openai_key

        return cls(**yaml_data)
