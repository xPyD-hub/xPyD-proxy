"""Tests for the xpyd CLI subcommand parser and main() flow."""

from __future__ import annotations

import os
import textwrap
from unittest.mock import patch

import pytest

from xpyd.config import ProxyConfig
from xpyd.proxy import _build_parser, _resolve_config_path


class TestSubcommandParser:
    """Verify the new subcommand-based parser."""

    def test_proxy_subcommand_parses(self):
        parser = _build_parser()
        args = parser.parse_args(["proxy", "--config", "test.yaml"])
        assert args.command == "proxy"
        assert args.config == "test.yaml"

    def test_no_subcommand_shows_help(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_version_flag(self, capsys):
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_init_config_generates_file(self, tmp_path):
        from xpyd.init_config import generate_config_template

        out = tmp_path / "xpyd.yaml"
        generate_config_template(str(out))
        assert out.exists()
        content = out.read_text()
        assert "model:" in content
        assert "decode:" in content

    def test_init_config_custom_path(self, tmp_path):
        from xpyd.init_config import generate_config_template

        out = tmp_path / "sub" / "custom.yaml"
        generate_config_template(str(out))
        assert out.exists()

    def test_init_config_default_path(self):
        parser = _build_parser()
        args = parser.parse_args(["proxy", "--init-config"])
        assert args.init_config == "./xpyd.yaml"

    def test_init_config_explicit_path(self):
        parser = _build_parser()
        args = parser.parse_args(["proxy", "--init-config", "/tmp/out.yaml"])
        assert args.init_config == "/tmp/out.yaml"

    def test_validate_config_valid(self, tmp_path):
        p = tmp_path / "valid.yaml"
        p.write_text(
            textwrap.dedent(
                """\
            model: /path/model
            decode:
              - "10.0.0.1:8000"
        """
            )
        )
        config = ProxyConfig.from_yaml(str(p))
        assert config.model == "/path/model"

    def test_validate_config_invalid(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("not_a_field: oops\n")
        with pytest.raises(Exception):
            ProxyConfig.from_yaml(str(p))

    def test_port_override(self):
        parser = _build_parser()
        args = parser.parse_args(["proxy", "-c", "x.yaml", "--port", "9000"])
        assert args.port == 9000

    def test_log_level_override(self):
        parser = _build_parser()
        args = parser.parse_args(["proxy", "-c", "x.yaml", "--log-level", "debug"])
        assert args.log_level == "debug"

    def test_no_config_file_error_message(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        env = {k: v for k, v in os.environ.items() if k != "XPYD_CONFIG"}
        parser = _build_parser()
        args = parser.parse_args(["proxy"])
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                _resolve_config_path(args)
            assert exc_info.value.code == 1

    def test_old_args_rejected(self):
        parser = _build_parser()
        for flag in (
            "--model",
            "-m",
            "--prefill",
            "-p",
            "--decode",
            "-d",
            "--roundrobin",
            "--generator_on_p_node",
        ):
            with pytest.raises(SystemExit):
                parser.parse_args(["proxy", flag, "value"])


class TestConfigResolution:
    def test_cli_config_wins(self):
        parser = _build_parser()
        args = parser.parse_args(["proxy", "--config", "cli.yaml"])
        assert _resolve_config_path(args) == "cli.yaml"

    def test_env_var_fallback(self):
        parser = _build_parser()
        args = parser.parse_args(["proxy"])
        with patch.dict(os.environ, {"XPYD_CONFIG": "env.yaml"}):
            assert _resolve_config_path(args) == "env.yaml"

    def test_default_file_fallback(self, tmp_path, monkeypatch):
        (tmp_path / "xpyd.yaml").write_text("model: test\n")
        monkeypatch.chdir(tmp_path)
        parser = _build_parser()
        args = parser.parse_args(["proxy"])
        env = {k: v for k, v in os.environ.items() if k != "XPYD_CONFIG"}
        with patch.dict(os.environ, env, clear=True):
            result = _resolve_config_path(args)
        assert result is not None
        assert result.endswith("xpyd.yaml")
