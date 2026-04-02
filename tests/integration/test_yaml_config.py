"""Tests for YAML config loading, topology expansion, and precedence."""

from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from xpyd.config import ProxyConfig


@pytest.fixture
def tmp_yaml(tmp_path):
    """Helper that writes YAML content to a temp file and returns the path."""

    def _write(content: str) -> Path:
        p = tmp_path / "config.yaml"
        p.write_text(textwrap.dedent(content))
        return p

    return _write


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
        "log_level": "warning",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ------------------------------------------------------------------
# YAML loading basics
# ------------------------------------------------------------------


class TestLoadYaml:
    def test_valid_yaml(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /path/to/model
            decode:
              nodes:
                - "10.0.0.1:8200"
              tp_size: 1
              dp_size: 1
              world_size_per_node: 1
            """
        )
        data = ProxyConfig.load_yaml(p)
        assert data["model"] == "/path/to/model"

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
        args = _make_args(config=str(p))
        with pytest.raises(ValueError, match="Unknown keys"):
            ProxyConfig.from_args(args)


# ------------------------------------------------------------------
# Topology-style YAML
# ------------------------------------------------------------------


class TestTopologyYaml:
    def test_full_topology_expansion(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /path/model
            prefill:
              nodes:
                - "10.0.0.1:8100"
              tp_size: 8
              dp_size: 1
              world_size_per_node: 8
            decode:
              nodes:
                - "10.0.0.3:8200"
                - "10.0.0.4:8200"
              tp_size: 1
              dp_size: 16
              world_size_per_node: 8
            """
        )
        args = _make_args(config=str(p))
        cfg = ProxyConfig.from_args(args)
        assert cfg.prefill == ["10.0.0.1:8100"]
        assert len(cfg.decode) == 16
        assert cfg.decode[0] == "10.0.0.3:8200"
        assert cfg.decode[8] == "10.0.0.4:8200"

    def test_flat_list_backward_compat(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /path/model
            decode:
              - "10.0.0.1:8000"
              - "10.0.0.2:8000"
            """
        )
        args = _make_args(config=str(p))
        cfg = ProxyConfig.from_args(args)
        assert cfg.decode == ["10.0.0.1:8000", "10.0.0.2:8000"]

    def test_topology_missing_keys(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /path/model
            decode:
              nodes:
                - "10.0.0.1:8200"
              tp_size: 1
            """
        )
        args = _make_args(config=str(p))
        with pytest.raises(ValueError, match="missing keys"):
            ProxyConfig.from_args(args)

    def test_topology_invalid_constraint(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /path/model
            decode:
              nodes:
                - "10.0.0.1:8200"
              tp_size: 8
              dp_size: 2
              world_size_per_node: 8
            """
        )
        args = _make_args(config=str(p))
        with pytest.raises(ValueError, match="topology invalid"):
            ProxyConfig.from_args(args)

    def test_topology_unknown_keys_rejected(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /path/model
            decode:
              nodes:
                - "10.0.0.1:8200"
              tp_size: 1
              dp_size: 1
              world_size_per_node: 1
              extra_key: bad
            """
        )
        args = _make_args(config=str(p))
        with pytest.raises(ValueError, match="unknown keys"):
            ProxyConfig.from_args(args)


# ------------------------------------------------------------------
# Precedence: CLI > env > YAML > defaults
# ------------------------------------------------------------------


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

    def test_cli_decode_overrides_yaml_topology(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /yaml/model
            decode:
              nodes:
                - "10.0.0.1:8200"
              tp_size: 1
              dp_size: 1
              world_size_per_node: 1
            """
        )
        args = _make_args(config=str(p), decode=["10.0.0.99:9999"])
        cfg = ProxyConfig.from_args(args)
        assert cfg.decode == ["10.0.0.99:9999"]

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


# ------------------------------------------------------------------
# Scheduling
# ------------------------------------------------------------------


class TestScheduling:
    def test_roundrobin_from_yaml(self, tmp_yaml):
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

    def test_loadbalanced_from_yaml(self, tmp_yaml):
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

    def test_invalid_scheduling_rejected(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /yaml/model
            decode:
              - "10.0.0.1:8000"
            scheduling: typo
            """
        )
        args = _make_args(config=str(p))
        with pytest.raises(ValueError, match="Invalid scheduling value"):
            ProxyConfig.from_args(args)


# ------------------------------------------------------------------
# log_level
# ------------------------------------------------------------------


class TestLogLevel:
    def test_log_level_from_yaml(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /path/model
            decode:
              - "10.0.0.1:8000"
            log_level: debug
            """
        )
        args = _make_args(config=str(p))
        cfg = ProxyConfig.from_args(args)
        assert cfg.log_level == "debug"

    def test_invalid_log_level(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /path/model
            decode:
              - "10.0.0.1:8000"
            log_level: verbose
            """
        )
        args = _make_args(config=str(p))
        with pytest.raises(ValueError, match="log_level"):
            ProxyConfig.from_args(args)

    def test_default_log_level(self):
        args = _make_args(model="/m", decode=["10.0.0.1:8000"])
        cfg = ProxyConfig.from_args(args)
        assert cfg.log_level == "warning"


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
