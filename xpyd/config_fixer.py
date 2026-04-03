# SPDX-License-Identifier: Apache-2.0
"""Config fixer for xPyD YAML configurations.

Provides ``ConfigFixer`` which auto-corrects common configuration mistakes
and suggests fixes for ambiguous issues.
"""

from __future__ import annotations

import copy
import difflib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from xpyd.scheduler.policy_registry import default_registry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_ROLES = ("prefill", "decode", "dual")
_DEFAULT_PORT = 8000
_MIN_FUZZY_LEN = 3


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class FixResult:
    """A single auto-applied fix."""

    path: str
    old_value: str
    new_value: str
    description: str


@dataclass
class Suggestion:
    """A suggestion that requires user confirmation."""

    path: str
    message: str


@dataclass
class FixReport:
    """Aggregate report from a fixer run."""

    fixes: List[FixResult] = field(default_factory=list)
    suggestions: List[Suggestion] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fuzzy_match_role(value: str) -> Optional[str]:
    """Return the corrected role string, or ``None`` if no fix needed."""
    if value in _VALID_ROLES:
        return None
    # Case normalization first
    lowered = value.lower().strip()
    if lowered in _VALID_ROLES:
        return lowered
    # Edit-distance matching (only if long enough)
    if len(lowered) < _MIN_FUZZY_LEN:
        return None
    matches = difflib.get_close_matches(
        lowered, _VALID_ROLES, n=1, cutoff=0.6,
    )
    if matches:
        return matches[0]
    return None


def _fuzzy_match_scheduler(value: str) -> Optional[str]:
    """Return the corrected scheduler string, or ``None`` if no fix."""
    policies = default_registry.list_policies()
    if value in policies:
        return None
    lowered = value.lower().strip()
    if lowered in policies:
        return lowered
    if len(lowered) < _MIN_FUZZY_LEN:
        return None
    matches = difflib.get_close_matches(lowered, policies, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    return None


def _strip_address(addr: str) -> str:
    """Strip whitespace from an address string."""
    return addr.strip()


def _add_default_port(addr: str) -> Optional[str]:
    """Add default port if missing.  Returns new value or ``None``."""
    addr = addr.strip()
    if ":" in addr:
        return None
    # Must look like an IP or 'localhost'
    if addr == "localhost" or re.match(r"^\d{1,3}(\.\d{1,3}){3}$", addr):
        return f"{addr}:{_DEFAULT_PORT}"
    return None


# ---------------------------------------------------------------------------
# ConfigFixer
# ---------------------------------------------------------------------------


class ConfigFixer:
    """Auto-fix and suggest improvements for xPyD YAML configs.

    Parameters
    ----------
    data:
        Parsed YAML data (a ``dict``).
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = copy.deepcopy(data)
        self._report = FixReport()

    # -- public API --------------------------------------------------------

    def run(self) -> FixReport:
        """Run all fix and suggestion rules.  Returns a :class:`FixReport`."""
        self._fix_top_level_whitespace()
        self._fix_legacy_addresses("prefill")
        self._fix_legacy_addresses("decode")
        self._fix_scheduling()
        self._fix_instances()
        self._fix_models()
        # Suggest-only rules
        self._suggest_dual_pd_mix()
        self._suggest_address_conflict()
        self._suggest_unbalanced_pd()
        self._suggest_missing_decode()
        return self._report

    @property
    def fixed_data(self) -> Dict[str, Any]:
        """Return the (potentially modified) config data."""
        return self._data

    # -- auto-fix rules ----------------------------------------------------

    def _fix_top_level_whitespace(self) -> None:
        """Strip whitespace from model name."""
        model = self._data.get("model")
        if isinstance(model, str) and model != model.strip():
            old = model
            self._data["model"] = model.strip()
            self._report.fixes.append(FixResult(
                path="model",
                old_value=repr(old),
                new_value=repr(self._data["model"]),
                description="model name whitespace trimmed",
            ))

    def _fix_legacy_addresses(self, key: str) -> None:
        """Fix addresses in legacy prefill/decode lists."""
        addrs = self._data.get(key)
        if not isinstance(addrs, list):
            return
        for i, addr in enumerate(addrs):
            if not isinstance(addr, str):
                continue
            path = f"{key}[{i}]"
            self._fix_address(addrs, i, path)

    def _fix_address(self, lst: list, idx: int, path: str) -> None:
        """Fix a single address: strip whitespace, add default port."""
        addr = lst[idx]
        if not isinstance(addr, str):
            return
        stripped = _strip_address(addr)
        if stripped != addr:
            lst[idx] = stripped
            self._report.fixes.append(FixResult(
                path=path,
                old_value=repr(addr),
                new_value=repr(stripped),
                description="extra whitespace trimmed",
            ))
            addr = stripped

        port_fixed = _add_default_port(addr)
        if port_fixed is not None:
            lst[idx] = port_fixed
            self._report.fixes.append(FixResult(
                path=path,
                old_value=repr(addr),
                new_value=repr(port_fixed),
                description="default port added",
            ))

    def _fix_scheduling(self) -> None:
        """Fix top-level scheduling value."""
        sched = self._data.get("scheduling")
        if not isinstance(sched, str):
            return
        stripped = sched.strip()
        if stripped != sched:
            self._data["scheduling"] = stripped
            self._report.fixes.append(FixResult(
                path="scheduling",
                old_value=repr(sched),
                new_value=repr(stripped),
                description="extra whitespace trimmed",
            ))
            sched = stripped

        fixed = _fuzzy_match_scheduler(sched)
        if fixed is not None:
            self._data["scheduling"] = fixed
            self._report.fixes.append(FixResult(
                path="scheduling",
                old_value=repr(sched),
                new_value=repr(fixed),
                description="scheduler typo corrected",
            ))

    def _fix_instances(self) -> None:
        """Fix entries in the 'instances' list."""
        instances = self._data.get("instances")
        if not isinstance(instances, list):
            return
        for i, entry in enumerate(instances):
            if not isinstance(entry, dict):
                continue
            # Fix role
            role = entry.get("role")
            if isinstance(role, str):
                stripped = role.strip()
                if stripped != role:
                    entry["role"] = stripped
                    self._report.fixes.append(FixResult(
                        path=f"instances[{i}].role",
                        old_value=repr(role),
                        new_value=repr(stripped),
                        description="extra whitespace trimmed",
                    ))
                    role = stripped
                fixed_role = _fuzzy_match_role(role)
                if fixed_role is not None:
                    entry["role"] = fixed_role
                    self._report.fixes.append(FixResult(
                        path=f"instances[{i}].role",
                        old_value=repr(role),
                        new_value=repr(fixed_role),
                        description="role normalized",
                    ))
            # Fix address
            addr = entry.get("address")
            if isinstance(addr, str):
                self._fix_instance_address(entry, i)
            # Fix model whitespace
            model = entry.get("model")
            if isinstance(model, str) and model != model.strip():
                old = model
                entry["model"] = model.strip()
                self._report.fixes.append(FixResult(
                    path=f"instances[{i}].model",
                    old_value=repr(old),
                    new_value=repr(entry["model"]),
                    description="model name whitespace trimmed",
                ))

    def _fix_instance_address(self, entry: dict, idx: int) -> None:
        """Fix address within an instance entry."""
        addr = entry["address"]
        path = f"instances[{idx}].address"
        stripped = addr.strip()
        if stripped != addr:
            entry["address"] = stripped
            self._report.fixes.append(FixResult(
                path=path,
                old_value=repr(addr),
                new_value=repr(stripped),
                description="extra whitespace trimmed",
            ))
            addr = stripped
        port_fixed = _add_default_port(addr)
        if port_fixed is not None:
            entry["address"] = port_fixed
            self._report.fixes.append(FixResult(
                path=path,
                old_value=repr(addr),
                new_value=repr(port_fixed),
                description="default port added",
            ))

    def _fix_models(self) -> None:
        """Fix entries in the 'models' shorthand list."""
        models = self._data.get("models")
        if not isinstance(models, list):
            return
        for i, entry in enumerate(models):
            if not isinstance(entry, dict):
                continue
            # Fix model name whitespace
            name = entry.get("name")
            if isinstance(name, str) and name != name.strip():
                old = name
                entry["name"] = name.strip()
                self._report.fixes.append(FixResult(
                    path=f"models[{i}].name",
                    old_value=repr(old),
                    new_value=repr(entry["name"]),
                    description="model name whitespace trimmed",
                ))
            # Fix addresses in prefill/decode/dual lists
            for role_key in ("prefill", "decode", "dual"):
                addrs = entry.get(role_key)
                if not isinstance(addrs, list):
                    continue
                for j, addr in enumerate(addrs):
                    if not isinstance(addr, str):
                        continue
                    path = f"models[{i}].{role_key}[{j}]"
                    self._fix_models_address(addrs, j, path)
            # Fix per-model scheduler
            sched = entry.get("scheduler")
            if isinstance(sched, str):
                stripped = sched.strip()
                if stripped != sched:
                    entry["scheduler"] = stripped
                    self._report.fixes.append(FixResult(
                        path=f"models[{i}].scheduler",
                        old_value=repr(sched),
                        new_value=repr(stripped),
                        description="extra whitespace trimmed",
                    ))
                    sched = stripped
                fixed = _fuzzy_match_scheduler(sched)
                if fixed is not None:
                    entry["scheduler"] = fixed
                    self._report.fixes.append(FixResult(
                        path=f"models[{i}].scheduler",
                        old_value=repr(sched),
                        new_value=repr(fixed),
                        description="scheduler typo corrected",
                    ))

    def _fix_models_address(
        self, lst: list, idx: int, path: str,
    ) -> None:
        """Fix a single address in a models entry list."""
        addr = lst[idx]
        stripped = addr.strip()
        if stripped != addr:
            lst[idx] = stripped
            self._report.fixes.append(FixResult(
                path=path,
                old_value=repr(addr),
                new_value=repr(stripped),
                description="extra whitespace trimmed",
            ))
            addr = stripped
        port_fixed = _add_default_port(addr)
        if port_fixed is not None:
            lst[idx] = port_fixed
            self._report.fixes.append(FixResult(
                path=path,
                old_value=repr(addr),
                new_value=repr(port_fixed),
                description="default port added",
            ))

    # -- suggest-only rules ------------------------------------------------

    def _collect_model_roles(self) -> Dict[str, Dict[str, int]]:
        """Collect per-model role counts from instances or models."""
        model_roles: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int),
        )
        instances = self._data.get("instances")
        if isinstance(instances, list):
            for entry in instances:
                if not isinstance(entry, dict):
                    continue
                model = entry.get("model", "")
                role = entry.get("role", "")
                if role in _VALID_ROLES:
                    model_roles[model][role] += 1
        models = self._data.get("models")
        if isinstance(models, list):
            for entry in models:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name", "")
                for role_key in ("prefill", "decode", "dual"):
                    addrs = entry.get(role_key)
                    if isinstance(addrs, list):
                        model_roles[name][role_key] += len(addrs)
        return model_roles

    def _collect_addresses(self) -> Dict[str, List[str]]:
        """Collect address → list of model names mapping."""
        addr_models: Dict[str, List[str]] = defaultdict(list)
        instances = self._data.get("instances")
        if isinstance(instances, list):
            for entry in instances:
                if not isinstance(entry, dict):
                    continue
                addr = entry.get("address", "")
                model = entry.get("model", "")
                addr_models[addr].append(model)
        models = self._data.get("models")
        if isinstance(models, list):
            for entry in models:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name", "")
                for role_key in ("prefill", "decode", "dual"):
                    addrs = entry.get(role_key)
                    if isinstance(addrs, list):
                        for addr in addrs:
                            if isinstance(addr, str):
                                addr_models[addr].append(name)
        return addr_models

    def _suggest_dual_pd_mix(self) -> None:
        """Warn when a model mixes dual and P/D instances."""
        for model, roles in self._collect_model_roles().items():
            has_dual = roles.get("dual", 0) > 0
            has_pd = roles.get("prefill", 0) > 0 or roles.get("decode", 0) > 0
            if has_dual and has_pd:
                self._report.suggestions.append(Suggestion(
                    path=f"model '{model}'",
                    message=(
                        f"Model '{model}' has both dual and prefill/decode "
                        f"instances. Consider converting all to dual or all "
                        f"to prefill/decode."
                    ),
                ))

    def _suggest_address_conflict(self) -> None:
        """Warn when the same address serves multiple models."""
        for addr, models in self._collect_addresses().items():
            unique_models = set(m for m in models if m)
            if len(unique_models) > 1:
                self._report.suggestions.append(Suggestion(
                    path=f"address '{addr}'",
                    message=(
                        f"Address '{addr}' is used by multiple models: "
                        f"{sorted(unique_models)}. Possible conflict."
                    ),
                ))

    def _suggest_unbalanced_pd(self) -> None:
        """Warn when prefill/decode ratio is heavily unbalanced."""
        for model, roles in self._collect_model_roles().items():
            p = roles.get("prefill", 0)
            d = roles.get("decode", 0)
            if p > 0 and d > 0:
                ratio = max(p, d) / min(p, d)
                if ratio >= 4.0:
                    self._report.suggestions.append(Suggestion(
                        path=f"model '{model}'",
                        message=(
                            f"Model '{model}' has {p} prefill and {d} decode "
                            f"instances (ratio {ratio:.1f}:1). Consider "
                            f"rebalancing."
                        ),
                    ))

    def _suggest_missing_decode(self) -> None:
        """Warn when prefill is present but decode is missing."""
        for model, roles in self._collect_model_roles().items():
            has_dual = roles.get("dual", 0) > 0
            p = roles.get("prefill", 0)
            d = roles.get("decode", 0)
            if has_dual:
                continue
            if p > 0 and d == 0:
                self._report.suggestions.append(Suggestion(
                    path=f"model '{model}'",
                    message=(
                        f"Model '{model}' has {p} prefill but no decode "
                        f"instances. P/D mode requires both."
                    ),
                ))
            elif d > 0 and p == 0:
                self._report.suggestions.append(Suggestion(
                    path=f"model '{model}'",
                    message=(
                        f"Model '{model}' has {d} decode but no prefill "
                        f"instances. P/D mode requires both."
                    ),
                ))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_fix_config(
    config_path: str,
    *,
    write: bool = False,
    interactive: bool = False,
) -> int:
    """Run the fix-config command.  Returns exit code (0 = ok)."""
    import sys
    from datetime import datetime
    from pathlib import Path

    path = Path(config_path)
    if not path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    try:
        with open(path) as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        print(f"Error: malformed YAML: {exc}", file=sys.stderr)
        return 1

    if not isinstance(data, dict):
        print("Error: YAML config must be a mapping.", file=sys.stderr)
        return 1

    fixer = ConfigFixer(data)
    report = fixer.run()

    # Print fixes
    if report.fixes:
        print(f"\u2705 Fixed {len(report.fixes)} issue(s):")
        for fix in report.fixes:
            print(f"  - {fix.path}: {fix.old_value} \u2192 {fix.new_value}"
                  f" ({fix.description})")
    else:
        print("\u2705 No issues found.")

    # Handle suggestions
    if report.suggestions:
        print(f"\n\u26a0\ufe0f  {len(report.suggestions)} issue(s) need "
              f"your attention:")
        for sug in report.suggestions:
            print(f"  - {sug.path}: {sug.message}")
            if interactive:
                try:
                    answer = input("    Apply suggestion? [y/n] ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    answer = "n"
                if answer != "y":
                    print("    Skipped.")

    # Output
    fixed_yaml = yaml.dump(fixer.fixed_data, default_flow_style=False,
                           sort_keys=False)
    if write:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{config_path}.{timestamp}.bak"
        # Create backup
        import shutil
        shutil.copy2(config_path, backup_path)
        print(f"\nBackup saved to: {backup_path}")
        with open(path, "w") as fh:
            fh.write(fixed_yaml)
        print(f"Fixed config written to: {config_path}")
        print("\nNote: --write does not preserve YAML comments or formatting.")
    else:
        if report.fixes:
            print("\n--- Fixed config ---")
            print(fixed_yaml)

    return 0
