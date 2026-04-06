"""Tests for topology expansion and topology-style YAML config."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pytest

from xpyd.config import ProxyConfig
from xpyd.topology import expand_topology, validate_topology

# ------------------------------------------------------------------
# topology.py unit tests
# ------------------------------------------------------------------


class TestValidateTopology:
    def test_valid(self):
        validate_topology("prefill", ["10.0.0.1:8100"], 8, 1, 8)

    def test_tp_not_power_of_two(self):
        with pytest.raises(ValueError, match="power of two"):
            validate_topology("prefill", ["10.0.0.1:8100"], 3, 1, 3)

    def test_dp_not_power_of_two(self):
        with pytest.raises(ValueError, match="power of two"):
            validate_topology("decode", ["10.0.0.1:8200"], 1, 3, 3)

    def test_product_mismatch(self):
        with pytest.raises(ValueError, match="topology invalid"):
            validate_topology("prefill", ["10.0.0.1:8100"], 8, 2, 8)

    def test_negative_tp(self):
        with pytest.raises(ValueError, match="positive integer"):
            validate_topology("prefill", ["10.0.0.1:8100"], -1, 1, 1)


class TestExpandTopology:
    def test_single_node_single_instance(self):
        result = expand_topology("prefill", ["10.0.0.1:8100"], 8, 1, 8)
        assert result == ["10.0.0.1:8100"]

    def test_single_node_multiple_instances(self):
        result = expand_topology("decode", ["10.0.0.1:8200"], 1, 8, 8)
        expected = [f"10.0.0.1:{8200 + i}" for i in range(8)]
        assert result == expected

    def test_two_nodes_dp16(self):
        result = expand_topology(
            "decode",
            ["10.0.0.1:8200", "10.0.0.2:8200"],
            1,
            16,
            8,
        )
        assert len(result) == 16
        # First 8 on node 0
        for i in range(8):
            assert result[i] == f"10.0.0.1:{8200 + i}"
        # Next 8 on node 1
        for i in range(8):
            assert result[8 + i] == f"10.0.0.2:{8200 + i}"

    def test_cross_node_tp(self):
        # 2 nodes, TP16, DP1, world=8 → 1 instance spanning 2 nodes
        result = expand_topology(
            "prefill",
            ["10.0.0.1:8100", "10.0.0.2:8100"],
            16,
            1,
            8,
        )
        assert result == ["10.0.0.1:8100"]

    def test_invalid_node_format(self):
        with pytest.raises(ValueError, match="invalid node format"):
            expand_topology("prefill", ["badformat"], 1, 1, 1)


# ------------------------------------------------------------------
# YAML topology integration tests
# ------------------------------------------------------------------


@pytest.fixture
def tmp_yaml(tmp_path):
    def _write(content: str) -> Path:
        p = tmp_path / "config.yaml"
        p.write_text(textwrap.dedent(content))
        return p

    return _write


def _make_args(**overrides):
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


class TestYamlTopologyExpansion:
    def test_topology_style_yaml(self, tmp_yaml):
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

    def test_flat_list_still_works(self, tmp_yaml):
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

    def test_cli_overrides_yaml_topology(self, tmp_yaml):
        p = tmp_yaml(
            """\
            model: /path/model
            decode:
              nodes:
                - "10.0.0.1:8200"
              tp_size: 1
              dp_size: 1
              world_size_per_node: 1
            """
        )
        args = _make_args(
            config=str(p),
            decode=["10.0.0.99:9999"],
        )
        cfg = ProxyConfig.from_args(args)
        assert cfg.decode == ["10.0.0.99:9999"]

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
