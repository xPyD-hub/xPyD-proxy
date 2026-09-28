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
        assert cfg.host == "0.0.0.0"
        assert cfg.first_token_source == "decode"

    def test_nixl_disaggregated_mode(self):
        cfg = ProxyConfig(
            model="m",
            prefill=["10.0.0.1:8001"],
            decode=["10.0.0.2:8002"],
            disaggregated_mode="nixl",
        )
        assert cfg.disaggregated_mode == "nixl"

    def test_valid_full(self):
        cfg = ProxyConfig(
            model="/path/to/model",
            prefill=["10.0.0.1:8001"],
            decode=["10.0.0.2:8002", "10.0.0.3:8003"],
            port=9000,
            first_token_source="prefill",
            roundrobin=True,
            admin_api_key="secret",
            openai_api_key="sk-test",
        )
        assert cfg.port == 9000
        assert cfg.first_token_source == "prefill"
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

    @pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost", "0.0.0.0"])
    def test_listen_host_from_yaml_and_legacy_args(self, tmp_path, host):
        config = tmp_path / "proxy.yaml"
        config.write_text(f'host: "{host}"\nmodel: m\ndecode: ["127.0.0.1:8000"]\n')
        assert ProxyConfig.from_yaml(config).host == host
        assert ProxyConfig.from_args(argparse.Namespace(config=config)).host == host

    @pytest.mark.parametrize("host", ["", "http://localhost", "127.0.0.1:8000"])
    def test_invalid_listen_host(self, host):
        with pytest.raises(ValueError, match="host must be an IP address or localhost"):
            ProxyConfig(model="m", decode=["127.0.0.1:8000"], host=host)

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

    @pytest.mark.parametrize(
        "legacy_field",
        ["generator_on_p_node", "kv_transfer_backend"],
    )
    def test_legacy_fields_rejected(self, legacy_field):
        with pytest.raises(ValueError, match=legacy_field):
            ProxyConfig(
                model="m",
                decode=["10.0.0.1:8000"],
                **{legacy_field: False},
            )

    def test_defaults(self):
        cfg = ProxyConfig(model="m", decode=["10.0.0.1:8000"])
        assert cfg.port == 8000
        assert cfg.first_token_source == "decode"
        assert cfg.roundrobin is False
        assert cfg.admin_api_key is None
        assert cfg.openai_api_key is None
        assert cfg.tokenizer_path is None
        assert cfg.scheduling == "loadbalanced"


class TestProxyConfigFromArgs:
    """from_args() bridge."""

    @staticmethod
    def _make_args(**overrides):
        defaults = {
            "model": "/path/model",
            "prefill": ["10.0.0.1:8001"],
            "decode": ["10.0.0.2:8002"],
            "port": 8000,
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


class TestProxyConfigFromYaml:
    """from_yaml() classmethod."""

    def test_from_yaml_minimal(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text("model: my-model\ndecode:\n  - '10.0.0.1:8000'\n")
        cfg = ProxyConfig.from_yaml(str(p))
        assert cfg.model == "my-model"
        assert cfg.decode == ["10.0.0.1:8000"]
        assert cfg.port == 8000

    def test_from_yaml_tokenizer_path(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text(
            "tokenizer_path: /models/tokenizers\n"
            "instances:\n"
            "  - address: '10.0.0.1:8000'\n"
            "    role: aggregated\n"
            "    model: org/model\n"
        )
        cfg = ProxyConfig.from_yaml(str(p))
        assert cfg.tokenizer_path == "/models/tokenizers"
        assert cfg.scheduling == "loadbalanced"

    def test_from_yaml_full(self, tmp_path):
        import textwrap

        p = tmp_path / "cfg.yaml"
        p.write_text(textwrap.dedent("""\
            model: /path/model
            prefill:
              - "10.0.0.1:8001"
            decode:
              - "10.0.0.2:8002"
            port: 9000
            log_level: debug
            scheduling: roundrobin
            startup:
              wait_timeout_seconds: 120
              probe_interval_seconds: 5
              heartbeat_interval_seconds: 15
            health_check:
              enabled: true
              interval_seconds: 5.0
              timeout_seconds: 2.0
        """))
        cfg = ProxyConfig.from_yaml(str(p))
        assert cfg.port == 9000
        assert cfg.log_level == "debug"
        assert cfg.roundrobin is True
        assert cfg.scheduling == "roundrobin"
        assert cfg.wait_timeout_seconds == 120
        assert cfg.heartbeat_interval_seconds == 15
        assert cfg.health_check.enabled is True

    def test_from_yaml_env_override(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text(
            "model: m\ndecode:\n  - '10.0.0.1:8000'\nadmin_api_key: yaml-key\n"
        )
        with patch.dict(os.environ, {"ADMIN_API_KEY": "env-key"}):
            cfg = ProxyConfig.from_yaml(str(p))
        assert cfg.admin_api_key == "env-key"

    @pytest.mark.parametrize(
        "legacy_field",
        ["generator_on_p_node", "kv_transfer_backend"],
    )
    def test_from_yaml_legacy_fields_rejected(self, tmp_path, legacy_field):
        p = tmp_path / "cfg.yaml"
        p.write_text(f"model: m\ndecode:\n  - '10.0.0.1:8000'\n{legacy_field}: false\n")
        with pytest.raises(ValueError, match=legacy_field):
            ProxyConfig.from_yaml(p)

    @pytest.mark.parametrize("disaggregated_mode", ["direct", "nixl"])
    @pytest.mark.parametrize("source", ["prefill", "decode"])
    def test_first_token_source_supported_for_disaggregated_modes(
        self, disaggregated_mode, source
    ):
        cfg = ProxyConfig(
            model="m",
            prefill=["10.0.0.1:8001"],
            decode=["10.0.0.2:8002"],
            disaggregated_mode=disaggregated_mode,
            first_token_source=source,
        )
        assert cfg.first_token_source == source

    def test_zmq_p_first_receiver_alignment(self):
        cfg = ProxyConfig(
            model="m",
            prefill=["10.0.0.1:8001"],
            decode=["10.0.0.2:8002"],
            disaggregated_mode="zmq",
            first_token_source="prefill",
            zmq={
                "receivers": {
                    "10.0.0.2:8002": {
                        "host": "10.0.0.2",
                        "init_ports": [7300],
                        "alloc_ports": [7400],
                        "skip_last_n_tokens": 1,
                        "discard_partial_chunks": False,
                    },
                },
            },
        )
        receiver = cfg.zmq.receivers["10.0.0.2:8002"]
        assert receiver.skip_last_n_tokens == 1
        assert receiver.discard_partial_chunks is False

    @pytest.mark.parametrize(
        ("skip_last_n_tokens", "discard_partial_chunks"),
        [
            (0, False),
            (1, True),
        ],
    )
    def test_zmq_p_first_receiver_alignment_mismatch_rejected(
        self, skip_last_n_tokens, discard_partial_chunks
    ):
        with pytest.raises(
            ValueError,
            match="skip_last_n_tokens|discard_partial_chunks",
        ):
            ProxyConfig(
                model="m",
                prefill=["10.0.0.1:8001"],
                decode=["10.0.0.2:8002"],
                disaggregated_mode="zmq",
                first_token_source="prefill",
                zmq={
                    "receivers": {
                        "10.0.0.2:8002": {
                            "host": "10.0.0.2",
                            "init_ports": [7300],
                            "alloc_ports": [7400],
                            "skip_last_n_tokens": skip_last_n_tokens,
                            "discard_partial_chunks": discard_partial_chunks,
                        },
                    },
                },
            )

    def test_zmq_d_first_receiver_alignment_is_not_required(self):
        cfg = ProxyConfig(
            model="m",
            prefill=["10.0.0.1:8001"],
            decode=["10.0.0.2:8002"],
            disaggregated_mode="zmq",
            first_token_source="decode",
            zmq={
                "receivers": {
                    "10.0.0.2:8002": {
                        "host": "10.0.0.2",
                        "init_ports": [7300],
                        "alloc_ports": [7400],
                        "skip_last_n_tokens": 0,
                        "discard_partial_chunks": True,
                    },
                },
            },
        )
        assert cfg.first_token_source == "decode"

    def test_from_yaml_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ProxyConfig.from_yaml("/nonexistent/path.yaml")


class TestMultiModelConfig:
    """Config parsing tests for multi-model format."""

    def test_instances_config_valid(self):
        cfg = ProxyConfig(
            instances=[
                {"address": "10.0.0.1:8000", "role": "prefill", "model": "llama-3"},
                {"address": "10.0.0.2:8000", "role": "decode", "model": "llama-3"},
            ],
        )
        assert len(cfg.instances) == 2
        assert cfg.instances[0].model == "llama-3"

    def test_models_shorthand_expands(self):
        cfg = ProxyConfig(
            models=[
                {
                    "name": "llama-3",
                    "prefill": ["10.0.0.1:8000"],
                    "decode": ["10.0.0.2:8000"],
                },
            ],
        )
        assert cfg.instances is not None
        assert len(cfg.instances) == 2
        assert cfg.models is None  # consumed

    def test_models_shorthand_empty_name_rejected(self):
        with pytest.raises(ValueError, match="non-empty 'name'"):
            ProxyConfig(
                models=[
                    {
                        "prefill": ["10.0.0.1:8000"],
                        "decode": ["10.0.0.2:8000"],
                    },
                ],
            )

    def test_models_shorthand_no_decode_rejected(self):
        with pytest.raises(
            ValueError, match="requires at least one prefill and one decode"
        ):
            ProxyConfig(
                models=[
                    {
                        "name": "llama-3",
                        "prefill": ["10.0.0.1:8000"],
                    },
                ],
            )

    def test_instances_no_decode_rejected(self):
        with pytest.raises(
            ValueError, match="requires at least one prefill and one decode"
        ):
            ProxyConfig(
                instances=[
                    {"address": "10.0.0.1:8000", "role": "prefill", "model": "llama-3"},
                ],
            )

    def test_both_models_and_instances_rejected(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            ProxyConfig(
                models=[
                    {
                        "name": "a",
                        "prefill": ["10.0.0.1:8000"],
                        "decode": ["10.0.0.2:8000"],
                    }
                ],
                instances=[
                    {"address": "10.0.0.3:8000", "role": "prefill", "model": "b"},
                    {"address": "10.0.0.4:8000", "role": "decode", "model": "b"},
                ],
            )

    def test_old_single_model_still_works(self):
        cfg = ProxyConfig(
            model="llama-3",
            decode=["10.0.0.1:8000"],
        )
        assert cfg.model == "llama-3"
        assert cfg.instances is None

    def test_from_args_multi_model_yaml(self, tmp_path):
        p = tmp_path / "multi.yaml"
        p.write_text(
            "instances:\n"
            "  - address: '10.0.0.1:8000'\n"
            "    role: prefill\n"
            "    model: llama-3\n"
            "  - address: '10.0.0.2:8000'\n"
            "    role: decode\n"
            "    model: llama-3\n"
        )
        args = argparse.Namespace(
            config=str(p),
            model=None,
            prefill=None,
            decode=None,
            port=8000,
            roundrobin=False,
            log_level="warning",
        )
        cfg = ProxyConfig.from_args(args)
        assert cfg.instances is not None
        assert len(cfg.instances) == 2
