"""Tests for core/config.py — ProxyConfig validation and construction."""

from __future__ import annotations

import argparse
import os
from unittest.mock import patch

import pytest

from xpyd.config import ProxyConfig


class TestProxyConfigValidation:
    """Validation rules."""

    def test_valid_minimal(self):
        cfg = ProxyConfig(model="m", decode=["10.0.0.1:8000"])
        assert cfg.model == "m"
        assert cfg.prefill == []
        assert cfg.decode == ["10.0.0.1:8000"]
        assert cfg.port == 8000

    def test_valid_full(self):
        cfg = ProxyConfig(
            model="/path/to/model",
            prefill=["10.0.0.1:8001"],
            decode=["10.0.0.2:8002", "10.0.0.3:8003"],
            port=9000,
            generator_on_p_node=True,
            roundrobin=True,
            admin_api_key="secret",
            openai_api_key="sk-test",
        )
        assert cfg.port == 9000
        assert cfg.generator_on_p_node is True
        assert cfg.roundrobin is True
        assert cfg.admin_api_key == "secret"

    def test_decode_required(self):
        with pytest.raises(ValueError, match="at least one decode"):
            ProxyConfig(model="m", decode=[])

    def test_decode_none_coerced(self):
        with pytest.raises(ValueError, match="at least one decode"):
            ProxyConfig(model="m", decode=None)

    def test_prefill_none_coerced(self):
        cfg = ProxyConfig(model="m", prefill=None, decode=["10.0.0.1:8000"])
        assert cfg.prefill == []

    def test_port_too_low(self):
        with pytest.raises(ValueError, match="port must be between"):
            ProxyConfig(model="m", decode=["10.0.0.1:8000"], port=0)

    def test_port_too_high(self):
        with pytest.raises(ValueError, match="port must be between"):
            ProxyConfig(model="m", decode=["10.0.0.1:8000"], port=70000)

    def test_invalid_instance_format_no_colon(self):
        with pytest.raises(ValueError, match="Invalid instance format"):
            ProxyConfig(model="m", decode=["badformat"])

    def test_invalid_instance_host(self):
        with pytest.raises(ValueError, match="Invalid host"):
            ProxyConfig(model="m", decode=["not-an-ip:8000"])

    def test_invalid_instance_port_non_numeric(self):
        with pytest.raises(ValueError, match="Invalid port"):
            ProxyConfig(model="m", decode=["10.0.0.1:abc"])

    def test_invalid_instance_port_out_of_range(self):
        with pytest.raises(ValueError, match="Port out of range"):
            ProxyConfig(model="m", decode=["10.0.0.1:99999"])

    def test_localhost_allowed(self):
        cfg = ProxyConfig(model="m", decode=["localhost:8000"])
        assert cfg.decode == ["localhost:8000"]

    def test_extra_fields_rejected(self):
        with pytest.raises(ValueError):
            ProxyConfig(model="m", decode=["10.0.0.1:8000"], unknown="x")

    def test_defaults(self):
        cfg = ProxyConfig(model="m", decode=["10.0.0.1:8000"])
        assert cfg.port == 8000
        assert cfg.generator_on_p_node is False
        assert cfg.roundrobin is False
        assert cfg.admin_api_key is None
        assert cfg.openai_api_key is None


class TestProxyConfigFromArgs:
    """from_args() bridge."""

    @staticmethod
    def _make_args(**overrides):
        defaults = {
            "model": "/path/model",
            "prefill": ["10.0.0.1:8001"],
            "decode": ["10.0.0.2:8002"],
            "port": 8000,
            "generator_on_p_node": False,
            "roundrobin": False,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_from_args_basic(self):
        args = self._make_args()
        cfg = ProxyConfig.from_args(args)
        assert cfg.model == "/path/model"
        assert cfg.prefill == ["10.0.0.1:8001"]
        assert cfg.decode == ["10.0.0.2:8002"]

    def test_from_args_env_vars(self):
        args = self._make_args()
        with patch.dict(os.environ, {"ADMIN_API_KEY": "ak", "OPENAI_API_KEY": "ok"}):
            cfg = ProxyConfig.from_args(args)
        assert cfg.admin_api_key == "ak"
        assert cfg.openai_api_key == "ok"

    def test_from_args_no_env_vars(self):
        args = self._make_args()
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("ADMIN_API_KEY", "OPENAI_API_KEY")
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = ProxyConfig.from_args(args)
        assert cfg.admin_api_key is None
        assert cfg.openai_api_key is None

    def test_from_args_prefill_none(self):
        args = self._make_args(prefill=None)
        cfg = ProxyConfig.from_args(args)
        assert cfg.prefill == []
