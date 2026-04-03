# SPDX-License-Identifier: Apache-2.0
"""Unit tests for dual-role instance configuration."""

from __future__ import annotations

import pytest

from xpyd.config import InstanceEntry, ProxyConfig


class TestInstanceEntryDualRole:
    """InstanceEntry accepts role='dual'."""

    def test_dual_role_valid(self):
        entry = InstanceEntry(address="10.0.0.1:8000", role="dual", model="qwen-2")
        assert entry.role == "dual"

    def test_invalid_role_rejected(self):
        with pytest.raises(ValueError, match="role must be"):
            InstanceEntry(address="10.0.0.1:8000", role="unknown")


class TestDualInstancesConfig:
    """Config parsing for dual instances."""

    def test_format2_dual_instances(self):
        cfg = ProxyConfig(
            instances=[
                {"address": "10.0.0.1:8000", "role": "dual", "model": "qwen-2"},
                {"address": "10.0.0.2:8000", "role": "dual", "model": "qwen-2"},
            ],
        )
        assert len(cfg.instances) == 2
        assert all(e.role == "dual" for e in cfg.instances)

    def test_format3_dual_shorthand(self):
        cfg = ProxyConfig(
            models=[
                {
                    "name": "qwen-2",
                    "dual": ["10.0.0.1:8000", "10.0.0.2:8000"],
                },
            ],
        )
        assert cfg.instances is not None
        assert len(cfg.instances) == 2
        assert all(e.role == "dual" for e in cfg.instances)
        assert cfg.models is None

    def test_format3_dual_with_scheduler(self):
        cfg = ProxyConfig(
            models=[
                {
                    "name": "qwen-2",
                    "dual": ["10.0.0.1:8000"],
                    "scheduler": "round_robin",
                },
            ],
        )
        assert cfg._model_schedulers == {"qwen-2": "round_robin"}

    def test_format3_mixed_models_dual_and_pd(self):
        """Different models can use dual vs P/D."""
        cfg = ProxyConfig(
            models=[
                {
                    "name": "llama-3",
                    "prefill": ["10.0.0.1:8000"],
                    "decode": ["10.0.0.2:8000"],
                },
                {
                    "name": "qwen-2",
                    "dual": ["10.0.0.3:8000", "10.0.0.4:8000"],
                },
            ],
        )
        assert cfg.instances is not None
        roles = {e.model: e.role for e in cfg.instances}
        assert roles["llama-3"] in ("prefill", "decode")
        assert roles["qwen-2"] == "dual"


class TestDualPDMutualExclusivity:
    """Same model cannot mix dual and P/D."""

    def test_format2_mixing_rejected(self):
        with pytest.raises(ValueError, match="mixes dual and prefill/decode"):
            ProxyConfig(
                instances=[
                    {"address": "10.0.0.1:8000", "role": "dual", "model": "qwen-2"},
                    {"address": "10.0.0.2:8000", "role": "decode", "model": "qwen-2"},
                ],
            )

    def test_format3_mixing_rejected(self):
        with pytest.raises(ValueError, match="cannot have both 'dual' and"):
            ProxyConfig(
                models=[
                    {
                        "name": "qwen-2",
                        "dual": ["10.0.0.1:8000"],
                        "prefill": ["10.0.0.2:8000"],
                    },
                ],
            )

    def test_format3_mixing_dual_decode_rejected(self):
        with pytest.raises(ValueError, match="cannot have both 'dual' and"):
            ProxyConfig(
                models=[
                    {
                        "name": "qwen-2",
                        "dual": ["10.0.0.1:8000"],
                        "decode": ["10.0.0.2:8000"],
                    },
                ],
            )


class TestRequireDecodeWithDual:
    """_require_decode accepts dual as alternative to decode."""

    def test_dual_only_valid(self):
        cfg = ProxyConfig(
            instances=[
                {"address": "10.0.0.1:8000", "role": "dual", "model": "qwen-2"},
            ],
        )
        assert len(cfg.instances) == 1

    def test_pd_without_decode_rejected(self):
        with pytest.raises(
            ValueError, match="requires at least one prefill and one decode"
        ):
            ProxyConfig(
                instances=[
                    {"address": "10.0.0.1:8000", "role": "prefill", "model": "llama-3"},
                ],
            )

    def test_legacy_format_unchanged(self):
        """Old prefill/decode format still works."""
        cfg = ProxyConfig(
            model="llama-3",
            decode=["10.0.0.1:8000"],
        )
        assert cfg.model == "llama-3"
        assert cfg.instances is None


class TestConfigEdgeCases:
    """Edge cases for dual config validation."""

    def test_pd_without_prefill_rejected(self):
        """Model with only decode instances (no prefill) is rejected."""
        with pytest.raises(ValueError, match="requires at least one prefill"):
            ProxyConfig(
                instances=[
                    {"address": "10.0.0.1:8000", "role": "decode", "model": "llama-3"},
                ],
            )

    def test_invalid_scheduler_name_accepted_by_config(self):
        """Config accepts unknown scheduler names (validation happens at runtime)."""
        cfg = ProxyConfig(
            models=[
                {
                    "name": "qwen-2",
                    "dual": ["10.0.0.1:8000"],
                    "scheduler": "nonexistent_strategy",
                },
            ],
        )
        assert cfg._model_schedulers == {"qwen-2": "nonexistent_strategy"}

    def test_scheduler_alias_round_robin(self):
        """'round_robin' (underscore) is accepted in config."""
        cfg = ProxyConfig(
            models=[
                {
                    "name": "qwen-2",
                    "dual": ["10.0.0.1:8000"],
                    "scheduler": "round_robin",
                },
            ],
        )
        assert cfg._model_schedulers == {"qwen-2": "round_robin"}
