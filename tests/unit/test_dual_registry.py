# SPDX-License-Identifier: Apache-2.0
"""Unit tests for dual-role registry isolation."""

from __future__ import annotations

from xpyd.registry import InstanceRegistry


class TestRegistryDualIsolation:
    """Dual instances must not appear in prefill/decode queries."""

    def setup_method(self):
        self.reg = InstanceRegistry()
        self.reg.add("prefill", "10.0.0.1:8000", model="llama-3")
        self.reg.add("decode", "10.0.0.2:8000", model="llama-3")
        self.reg.add("dual", "10.0.0.3:8000", model="qwen-2")
        self.reg.add("dual", "10.0.0.4:8000", model="qwen-2")
        for addr in [
            "10.0.0.1:8000",
            "10.0.0.2:8000",
            "10.0.0.3:8000",
            "10.0.0.4:8000",
        ]:
            self.reg.mark_healthy(addr)

    def test_dual_not_in_prefill(self):
        result = self.reg.get_available_instances("prefill")
        assert "10.0.0.3:8000" not in result
        assert "10.0.0.4:8000" not in result

    def test_dual_not_in_decode(self):
        result = self.reg.get_available_instances("decode")
        assert "10.0.0.3:8000" not in result
        assert "10.0.0.4:8000" not in result

    def test_get_dual_instances(self):
        result = self.reg.get_dual_instances()
        assert sorted(result) == ["10.0.0.3:8000", "10.0.0.4:8000"]

    def test_get_dual_instances_by_model(self):
        result = self.reg.get_dual_instances(model="qwen-2")
        assert sorted(result) == ["10.0.0.3:8000", "10.0.0.4:8000"]

    def test_get_dual_instances_wrong_model(self):
        result = self.reg.get_dual_instances(model="llama-3")
        assert result == []

    def test_dual_in_registered_models(self):
        models = self.reg.get_registered_models()
        assert "qwen-2" in models
        assert "llama-3" in models

    def test_unhealthy_dual_not_returned(self):
        self.reg.mark_unhealthy("10.0.0.3:8000")
        result = self.reg.get_dual_instances(model="qwen-2")
        assert result == ["10.0.0.4:8000"]


class TestRegistryDualAdd:
    """Registry accepts dual role."""

    def test_add_dual(self):
        reg = InstanceRegistry()
        reg.add("dual", "10.0.0.1:8000", model="qwen-2")
        info = reg.get_instance_info("10.0.0.1:8000")
        assert info.role == "dual"
        assert info.model == "qwen-2"

    def test_invalid_role_rejected(self):
        reg = InstanceRegistry()
        import pytest

        with pytest.raises(ValueError, match="Invalid role"):
            reg.add("unknown", "10.0.0.1:8000")
