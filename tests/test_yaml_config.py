"""Tests for YAML config loading and CLI/YAML/env precedence."""

from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
from config import ProxyConfig


@pytest.fixture
def tmp_yaml(tmp_path):
    """Helper that writes YAML content to a temp file and returns the path."""

    def _write(content: str) -> Path:
        p = tmp_path / "config.yaml"
        p.write_text(textwrap.dedent(content))
        return p

    return _write


# ------------------------------------------------------------------
# YAML loading
# ------------------------------------------------------------------


class TestLoadYaml:
    def test_valid_yaml(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /path/to/model
            decode:
              - "10.0.0.1:8000"
            port: 9000
            """
        )
        data = ProxyConfig.load_yaml(p)
        assert data["model"] == "/path/to/model"
        assert data["port"] == 9000

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            ProxyConfig.load_yaml("/nonexistent/path.yaml")

    def test_malformed_yaml(self, tmp_yaml):
        p = tmp_yaml("{ bad yaml :::")
        with pytest.raises(ValueError, match="Malformed YAML"):
            ProxyConfig.load_yaml(p)

    def test_non_dict_yaml(self, tmp_yaml):
        p = tmp_yaml("- just\n- a\n- list\n")
        with pytest.raises(ValueError, match="must be a mapping"):
            ProxyConfig.load_yaml(p)

    def test_unknown_keys_rejected(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /path/model
            decode:
              - "10.0.0.1:8000"
            unknown_field: oops
            """
        )
        with pytest.raises(ValueError):
            ProxyConfig(**ProxyConfig.load_yaml(p))


# ------------------------------------------------------------------
# Precedence: CLI > env > YAML > defaults
# ------------------------------------------------------------------


def _make_args(**overrides):
    """Build a minimal argparse Namespace with defaults."""
    defaults = {
        "config": None,
        "model": None,
        "prefill": None,
        "decode": None,
        "port": 8000,
        "generator_on_p_node": False,
        "roundrobin": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestPrecedence:
    def test_yaml_only(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /yaml/model
            decode:
              - "10.0.0.1:8000"
            port: 9000
            """
        )
        args = _make_args(config=str(p))
        cfg = ProxyConfig.from_args(args)
        assert cfg.model == "/yaml/model"
        assert cfg.port == 9000

    def test_cli_overrides_yaml(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /yaml/model
            decode:
              - "10.0.0.1:8000"
            port: 9000
            """
        )
        args = _make_args(config=str(p), port=7777, model="/cli/model")
        cfg = ProxyConfig.from_args(args)
        assert cfg.model == "/cli/model"
        assert cfg.port == 7777

    def test_cli_decode_overrides_yaml(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /yaml/model
            decode:
              - "10.0.0.1:8000"
            """
        )
        args = _make_args(config=str(p), decode=["10.0.0.2:9000"])
        cfg = ProxyConfig.from_args(args)
        assert cfg.decode == ["10.0.0.2:9000"]

    def test_env_overrides_yaml_for_api_keys(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /yaml/model
            decode:
              - "10.0.0.1:8000"
            admin_api_key: yaml-key
            """
        )
        args = _make_args(config=str(p))
        with patch.dict(os.environ, {"ADMIN_API_KEY": "env-key"}):
            cfg = ProxyConfig.from_args(args)
        # env var takes precedence over yaml via `or` logic
        assert cfg.admin_api_key == "env-key"

    def test_yaml_api_key_fallback(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /yaml/model
            decode:
              - "10.0.0.1:8000"
            admin_api_key: yaml-key
            """
        )
        args = _make_args(config=str(p))
        env = {k: v for k, v in os.environ.items() if k != "ADMIN_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            cfg = ProxyConfig.from_args(args)
        assert cfg.admin_api_key == "yaml-key"

    def test_no_config_uses_cli_only(self):
        args = _make_args(model="/cli/model", decode=["10.0.0.1:8000"], port=5555)
        cfg = ProxyConfig.from_args(args)
        assert cfg.model == "/cli/model"
        assert cfg.port == 5555

    def test_scheduling_roundrobin_from_yaml(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /yaml/model
            decode:
              - "10.0.0.1:8000"
            scheduling: roundrobin
            """
        )
        args = _make_args(config=str(p))
        cfg = ProxyConfig.from_args(args)
        assert cfg.roundrobin is True

    def test_scheduling_loadbalanced_from_yaml(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /yaml/model
            decode:
              - "10.0.0.1:8000"
            scheduling: loadbalanced
            """
        )
        args = _make_args(config=str(p))
        cfg = ProxyConfig.from_args(args)
        assert cfg.roundrobin is False

    def test_cli_roundrobin_overrides_yaml(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /yaml/model
            decode:
              - "10.0.0.1:8000"
            scheduling: loadbalanced
            """
        )
        args = _make_args(config=str(p), roundrobin=True)
        cfg = ProxyConfig.from_args(args)
        assert cfg.roundrobin is True

    def test_prefill_from_yaml(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /yaml/model
            prefill:
              - "10.0.0.1:8001"
              - "10.0.0.2:8001"
            decode:
              - "10.0.0.3:8002"
            """
        )
        args = _make_args(config=str(p))
        cfg = ProxyConfig.from_args(args)
        assert cfg.prefill == ["10.0.0.1:8001", "10.0.0.2:8001"]

    def test_generator_on_p_node_from_yaml(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /yaml/model
            decode:
              - "10.0.0.1:8000"
            generator_on_p_node: true
            """
        )
        args = _make_args(config=str(p))
        cfg = ProxyConfig.from_args(args)
        assert cfg.generator_on_p_node is True


# ------------------------------------------------------------------
# Missing model
# ------------------------------------------------------------------


class TestMissingModel:
    def test_no_model_anywhere_raises(self):
        args = _make_args(decode=["10.0.0.1:8000"])
        with pytest.raises(ValueError):
            ProxyConfig.from_args(args)

    def test_model_in_yaml_only(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /yaml/model
            decode:
              - "10.0.0.1:8000"
            """
        )
        args = _make_args(config=str(p))
        cfg = ProxyConfig.from_args(args)
        assert cfg.model == "/yaml/model"

    def test_unknown_keys_rejected_via_from_args(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /path/model
            decode:
              - "10.0.0.1:8000"
            unknown_field: oops
            """
        )
        args = _make_args(config=str(p))
        with pytest.raises(ValueError, match="Unknown keys"):
            ProxyConfig.from_args(args)
