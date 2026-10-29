"""Tests for the xpyd CLI subcommand parser and main() flow."""

from __future__ import annotations

import os
import textwrap
from unittest.mock import patch

import pytest

from xpyd.cli import (
    _apply_cli_overrides,
    _build_parser,
    _normalize_cli_args,
    _print_config_summary,
    _resolve_config_path,
)
from xpyd.config import ProxyConfig
from xpyd.proxy import Proxy, ProxyServer, main


class TestSubcommandParser:
    """Verify the new subcommand-based parser."""

    def test_prompt_displays_all_choices(self):
        from xpyd.init_config import _prompt

        with patch("builtins.input", return_value="") as input_mock:
            assert (
                _prompt(
                    "Deployment topology",
                    default="disaggregated",
                    choices=("aggregated", "disaggregated"),
                )
                == "disaggregated"
            )

        input_mock.assert_called_once_with(
            "Deployment topology (aggregated/disaggregated) [disaggregated]: "
        )

    def test_prompt_underlines_default_in_tty(self):
        from xpyd.init_config import _prompt

        with (
            patch("xpyd.init_config.sys.stdout.isatty", return_value=True),
            patch.dict(os.environ, {}, clear=True),
            patch("builtins.input", return_value="") as input_mock,
        ):
            assert _prompt("Mode", default="same") == "same"

        input_mock.assert_called_once_with("Mode [\033[4msame\033[0m]: ")

    def test_prompt_choices_are_case_insensitive(self):
        from xpyd.init_config import _prompt

        with patch("builtins.input", return_value="AGGREGATED"):
            assert (
                _prompt(
                    "Topology",
                    choices=("aggregated", "disaggregated"),
                )
                == "aggregated"
            )

    def test_proxy_subcommand_parses(self):
        parser = _build_parser()
        args = parser.parse_args(["proxy", "--config", "test.yaml"])
        assert args.command == "proxy"
        assert args.config == "test.yaml"

    def test_proxy_help_marks_subcommand_optional(self, capsys):
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["proxy", "--help"])
        assert exc_info.value.code == 0
        assert "usage: xpyd [proxy]" in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("argv", "expected"),
        [
            ([], ["proxy"]),
            (["-c", "config.yaml"], ["proxy", "-c", "config.yaml"]),
            (["--init-config"], ["proxy", "--init-config"]),
            (["proxy", "-c", "config.yaml"], ["proxy", "-c", "config.yaml"]),
            (["fix-config", "config.yaml"], ["fix-config", "config.yaml"]),
            (
                ["proxy", "fix-config", "config.yaml"],
                ["fix-config", "config.yaml"],
            ),
            (["--version"], ["--version"]),
            (["proxy", "--version"], ["--version"]),
        ],
    )
    def test_proxy_is_default_subcommand(self, argv, expected):
        assert _normalize_cli_args(argv) == expected

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
        assert 'role: "aggregated"' in content
        assert "decode:" not in content
        config = ProxyConfig.from_yaml(out)
        assert config.model == "deepseek-ai/DeepSeek-V4-Flash"
        assert [
            (instance.role, instance.address, instance.model)
            for instance in config.instances
        ] == [
            (
                "aggregated",
                "10.0.0.1:8100",
                "deepseek-ai/DeepSeek-V4-Flash",
            )
        ]
        assert config.health_check.enabled is True

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

    def test_init_config_force(self):
        parser = _build_parser()
        args = parser.parse_args(["proxy", "--init-config", "/tmp/out.yaml", "--force"])
        assert args.force is True

    def test_init_config_interactive(self, tmp_path):
        from xpyd.init_config import generate_interactive_config

        out = tmp_path / "interactive.yaml"
        answers = iter(
            [
                "disaggregated",
                "my-org/my-model",
                "nixl",
                "1",
                "2",
                "same",
                "8000",
                "10.0.0.1",
                "per-instance",
                "10.0.0.2:8000, 10.0.0.3:8000",
                "/models/tokenizers",
                "9000",
                "info",
                "roundrobin",
                "prefill",
                "y",
            ]
        )
        with patch("builtins.input", side_effect=answers):
            generate_interactive_config(str(out))

        config = ProxyConfig.from_yaml(out)
        assert [(item.role, item.address, item.model) for item in config.instances] == [
            ("prefill", "10.0.0.1:8000", "my-org/my-model"),
            ("decode", "10.0.0.2:8000", "my-org/my-model"),
            ("decode", "10.0.0.3:8000", "my-org/my-model"),
        ]
        assert config.tokenizer_path == "/models/tokenizers"
        assert config.port == 9000
        assert config.log_level == "info"
        assert config.scheduling == "roundrobin"
        assert config.disaggregated_mode == "nixl"
        assert config.first_token_source == "prefill"
        assert config.health_check.enabled is True

    def test_init_config_interactive_zmq(self, tmp_path):
        from xpyd.init_config import generate_interactive_config

        out = tmp_path / "zmq.yaml"
        answers = iter(
            [
                "disaggregated",
                "my-model",
                "zmq",
                "1",
                "2",
                "same",
                "8100",
                "10.0.0.1",
                "per-instance",
                "10.0.0.2:8200,10.0.0.3:8201",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
        with patch("builtins.input", side_effect=answers):
            generate_interactive_config(str(out))

        config = ProxyConfig.from_yaml(out)
        assert config.disaggregated_mode == "zmq"
        assert config.zmq is not None
        assert sorted(config.zmq.receivers) == [
            "10.0.0.2:8200",
            "10.0.0.3:8201",
        ]
        assert config.zmq.receivers["10.0.0.2:8200"].init_ports == [7300]
        assert config.zmq.receivers["10.0.0.3:8201"].alloc_ports == [7401]

    def test_init_config_interactive_aggregated(self, tmp_path):
        from xpyd.init_config import generate_interactive_config

        out = tmp_path / "aggregated.yaml"
        answers = iter(
            [
                "aggregated",
                "my-org/my-model",
                "2",
                "same",
                "8000",
                "10.0.0.1-10.0.0.2",
                "",
                "",
                "",
                "",
                "",
            ]
        )
        with patch("builtins.input", side_effect=answers):
            generate_interactive_config(str(out))

        config = ProxyConfig.from_yaml(out)
        assert [(item.role, item.address, item.model) for item in config.instances] == [
            ("aggregated", "10.0.0.1:8000", "my-org/my-model"),
            ("aggregated", "10.0.0.2:8000", "my-org/my-model"),
        ]
        assert config.first_token_source == "decode"
        assert config.health_check.enabled is True

    def test_init_config_expands_address_formats(self):
        from xpyd.init_config import _expand_address_input

        addresses = _expand_address_input(
            "192.168.0.1, 192.168.0.2:9000 " "192.168.0.3-192.168.0.4",
            default_port=8100,
        )

        assert addresses == [
            "192.168.0.1:8100",
            "192.168.0.2:9000",
            "192.168.0.3:8100",
            "192.168.0.4:8100",
        ]

    def test_init_config_expands_per_instance_ports(self):
        from xpyd.init_config import _expand_address_input

        addresses = _expand_address_input(
            "192.168.0.1:8100-8102 " "192.168.0.2-192.168.0.4:8200-8202",
            default_port=8000,
            require_explicit_port=True,
        )

        assert addresses == [
            "192.168.0.1:8100",
            "192.168.0.1:8101",
            "192.168.0.1:8102",
            "192.168.0.2:8200",
            "192.168.0.3:8201",
            "192.168.0.4:8202",
        ]

    def test_init_config_reprompts_for_address_count_mismatch(self, capsys):
        from xpyd.init_config import _prompt_role_instances

        with patch(
            "builtins.input",
            side_effect=[
                "same",
                "8200",
                "192.168.0.1-192.168.0.2",
                "192.168.0.1-192.168.0.3",
            ],
        ):
            instances = _prompt_role_instances(
                "decode",
                3,
                base_port=8200,
                model="my-model",
            )

        assert [item["address"] for item in instances] == [
            "192.168.0.1:8200",
            "192.168.0.2:8200",
            "192.168.0.3:8200",
        ]
        assert "Expected 3 decode addresses" in capsys.readouterr().out

    def test_init_config_instance_count_must_be_positive(self):
        from xpyd.init_config import _prompt_instance_count

        with patch("builtins.input", side_effect=["0", "many", "2"]):
            assert _prompt_instance_count("decode") == 2

    def test_config_summary_reports_instance_topology(self, capsys):
        config = ProxyConfig(
            instances=[
                {
                    "address": "10.0.0.1:8000",
                    "role": "aggregated",
                    "model": "my-model",
                },
                {
                    "address": "10.0.0.2:8000",
                    "role": "aggregated",
                    "model": "my-model",
                },
            ]
        )

        _print_config_summary(config)

        output = capsys.readouterr().out
        assert "topology: aggregated" in output
        assert "models: my-model" in output
        assert "aggregated: 2 instances" in output
        assert "prefill:" not in output
        assert "decode:" not in output

    def test_proxy_server_does_not_contact_backends_during_construction(self):
        config = ProxyConfig(
            model="test-model",
            prefill=["127.0.0.1:18100"],
            decode=["127.0.0.1:18200"],
            health_check={"enabled": True},
        )

        with patch.object(Proxy, "ensure_tokenizer") as ensure_tokenizer:
            server = ProxyServer(config)

        ensure_tokenizer.assert_not_called()
        assert server._all_prefill == ["127.0.0.1:18100"]
        assert server._all_decode == ["127.0.0.1:18200"]

    @pytest.mark.parametrize(
        ("response", "expected"),
        [("y\n", True), ("YES\n", True), ("n\n", False), ("\n", False)],
    )
    def test_init_config_mode_selection(self, response, expected):
        from xpyd.init_config import _use_interactive_mode

        with (
            patch("xpyd.init_config.select.select", return_value=([object()], [], [])),
            patch("xpyd.init_config.sys.stdin") as stdin,
        ):
            stdin.readline.return_value = response
            assert _use_interactive_mode() is expected

    def test_init_config_mode_selection_times_out(self):
        from xpyd.init_config import _use_interactive_mode

        with patch(
            "xpyd.init_config.select.select",
            return_value=([], [], []),
        ) as select_mock:
            assert _use_interactive_mode() is False
        select_mock.assert_called_once()
        assert select_mock.call_args.args[3] == 5.0

    def test_init_config_has_no_interactive_flag(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["proxy", "--init-config", "--interactive"])

    def test_validate_config_valid(self, tmp_path):
        p = tmp_path / "valid.yaml"
        p.write_text(textwrap.dedent("""\
            model: /path/model
            decode:
              - "10.0.0.1:8000"
        """))
        config = ProxyConfig.from_yaml(str(p))
        assert config.model == "/path/model"

    def test_validate_config_invalid(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("not_a_field: oops\n")
        with pytest.raises(Exception, match="validation error"):
            ProxyConfig.from_yaml(str(p))

    def test_port_override(self):
        parser = _build_parser()
        args = parser.parse_args(["proxy", "-c", "x.yaml", "--port", "9000"])
        assert args.port == 9000

    @pytest.mark.parametrize("level", ["debug", "info", "warning", "error"])
    def test_log_level_override(self, level):
        parser = _build_parser()
        args = parser.parse_args(["proxy", "-c", "x.yaml", "--log-level", level])
        assert args.log_level == level

    @pytest.mark.parametrize("port", ["0", "-1", "65536", "not-a-number"])
    def test_invalid_port_rejected(self, port):
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["proxy", "--port", port])
        assert exc_info.value.code == 2

    def test_invalid_log_level_rejected(self):
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["proxy", "--log-level", "trace"])
        assert exc_info.value.code == 2

    def test_force_without_init_config_rejected(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["xpyd", "--force"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2
        assert "--force can only be used" in capsys.readouterr().err

    def test_disaggregated_mode_override(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["proxy", "-c", "x.yaml", "--disaggregated-mode", "zmq"]
        )
        assert args.disaggregated_mode == "zmq"

    def test_first_token_source_override(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["proxy", "-c", "x.yaml", "--first-token-source", "prefill"]
        )
        assert args.first_token_source == "prefill"

    def test_no_config_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        env = {k: v for k, v in os.environ.items() if k != "XPYD_CONFIG"}
        parser = _build_parser()
        args = parser.parse_args(["proxy"])
        with patch.dict(os.environ, env, clear=True):
            assert _resolve_config_path(args) is None

    def test_no_args_initializes_missing_default_config(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", ["xpyd"])
        env = {k: v for k, v in os.environ.items() if k != "XPYD_CONFIG"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch("xpyd.init_config.generate_config") as generate,
        ):
            main()

        generate.assert_called_once_with("./xpyd.yaml")
        assert "starting config initialization" in capsys.readouterr().out

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
            "--kv-transfer-backend",
        ):
            with pytest.raises(SystemExit):
                parser.parse_args(["proxy", flag, "value"])


class TestConfigResolution:
    def test_cli_config_wins_over_environment_and_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "xpyd.yaml").write_text("model: default\n")
        parser = _build_parser()
        args = parser.parse_args(["proxy", "--config", "cli.yaml"])
        with patch.dict(os.environ, {"XPYD_CONFIG": "env.yaml"}):
            assert _resolve_config_path(args) == "cli.yaml"

    def test_cli_config_wins(self):
        parser = _build_parser()
        args = parser.parse_args(["proxy", "--config", "cli.yaml"])
        assert _resolve_config_path(args) == "cli.yaml"

    def test_env_var_fallback(self):
        parser = _build_parser()
        args = parser.parse_args(["proxy"])
        with patch.dict(os.environ, {"XPYD_CONFIG": "env.yaml"}):
            assert _resolve_config_path(args) == "env.yaml"

    def test_default_file_fallback(self, tmp_path, monkeypatch, capsys):
        (tmp_path / "xpyd.yaml").write_text("model: test\n")
        monkeypatch.chdir(tmp_path)
        parser = _build_parser()
        args = parser.parse_args(["proxy"])
        env = {k: v for k, v in os.environ.items() if k != "XPYD_CONFIG"}
        with patch.dict(os.environ, env, clear=True):
            result = _resolve_config_path(args)
        assert result == "./xpyd.yaml"
        assert (
            "No config specified; found ./xpyd.yaml and using it."
            in capsys.readouterr().out
        )


class TestCliOverrides:
    @pytest.mark.parametrize(
        ("argument", "value", "field", "expected"),
        [
            ("--port", "8000", "port", 8000),
            ("--log-level", "error", "log_level", "error"),
            (
                "--disaggregated-mode",
                "nixl",
                "disaggregated_mode",
                "nixl",
            ),
            (
                "--first-token-source",
                "prefill",
                "first_token_source",
                "prefill",
            ),
        ],
    )
    def test_explicit_cli_value_overrides_yaml_default(
        self, argument, value, field, expected
    ):
        config = ProxyConfig(
            model="m",
            prefill=["127.0.0.1:8001"],
            decode=["127.0.0.1:8002"],
            port=9000,
            log_level="debug",
        )
        args = _build_parser().parse_args(["proxy", argument, value])

        updated = _apply_cli_overrides(config, args)

        assert getattr(updated, field) == expected

    def test_overrides_preserve_per_model_schedulers(self):
        config = ProxyConfig(
            models=[
                {
                    "name": "m",
                    "aggregated": ["127.0.0.1:8001"],
                    "scheduler": "roundrobin",
                }
            ]
        )
        args = _build_parser().parse_args(["proxy", "--port", "9000"])

        updated = _apply_cli_overrides(config, args)

        assert updated.port == 9000
        assert updated._model_schedulers == {"m": "roundrobin"}

    def test_invalid_combination_is_revalidated(self):
        config = ProxyConfig(
            model="m",
            prefill=["127.0.0.1:8001"],
            decode=["127.0.0.1:8002"],
        )
        args = _build_parser().parse_args(["proxy", "--disaggregated-mode", "zmq"])

        with pytest.raises(ValueError, match="requires a zmq configuration"):
            _apply_cli_overrides(config, args)


class TestCliMain:
    def test_cli_values_override_yaml_before_server_start(self, tmp_path, monkeypatch):
        config = tmp_path / "xpyd.yaml"
        config.write_text(
            "model: m\n"
            "prefill:\n"
            "  - 127.0.0.1:8001\n"
            "decode:\n"
            "  - 127.0.0.1:8002\n"
            "port: 9000\n"
            "log_level: debug\n"
            "disaggregated_mode: direct\n"
            "first_token_source: decode\n"
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "xpyd",
                "--config",
                str(config),
                "--port",
                "8000",
                "--log-level",
                "error",
                "--disaggregated-mode",
                "nixl",
                "--first-token-source",
                "prefill",
            ],
        )

        with patch("xpyd.proxy.ProxyServer") as server_class:
            main()

        effective = server_class.call_args.kwargs["config"]
        assert effective.port == 8000
        assert effective.log_level == "error"
        assert effective.disaggregated_mode == "nixl"
        assert effective.first_token_source == "prefill"
        server_class.return_value.run_server.assert_called_once_with()

    def test_init_config_refuses_to_overwrite(self, tmp_path, monkeypatch, capsys):
        config = tmp_path / "xpyd.yaml"
        config.write_text("keep: me\n")
        monkeypatch.setattr(
            "sys.argv",
            ["xpyd", "--init-config", str(config)],
        )

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 2
        assert config.read_text() == "keep: me\n"
        assert "--force to overwrite" in capsys.readouterr().err

    def test_init_config_force_overwrites(self, tmp_path, monkeypatch):
        config = tmp_path / "xpyd.yaml"
        config.write_text("old: content\n")
        monkeypatch.setattr(
            "sys.argv",
            ["xpyd", "--init-config", str(config), "--force"],
        )

        with patch("xpyd.init_config._use_interactive_mode", return_value=False):
            main()

        assert "Generated by" not in config.read_text()
        assert "instances:" in config.read_text()

    def test_init_config_write_failure_is_concise(self, tmp_path, monkeypatch, capsys):
        output = tmp_path / "xpyd.yaml"
        monkeypatch.setattr(
            "sys.argv",
            ["xpyd", "--init-config", str(output)],
        )

        with (
            patch(
                "xpyd.init_config.generate_config",
                side_effect=PermissionError("permission denied"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        assert "failed to write config: permission denied" in capsys.readouterr().err

    def test_missing_startup_config_is_concise(self, tmp_path, monkeypatch, capsys):
        missing = tmp_path / "missing.yaml"
        monkeypatch.setattr("sys.argv", ["xpyd", "--config", str(missing)])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        error = capsys.readouterr().err
        assert "failed to load config" in error
        assert str(missing) in error

    def test_invalid_cli_override_combination_is_concise(
        self, tmp_path, monkeypatch, capsys
    ):
        config = tmp_path / "xpyd.yaml"
        config.write_text(
            "model: m\n"
            "prefill:\n"
            "  - 127.0.0.1:8001\n"
            "decode:\n"
            "  - 127.0.0.1:8002\n"
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "xpyd",
                "--config",
                str(config),
                "--disaggregated-mode",
                "zmq",
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        assert "requires a zmq configuration" in capsys.readouterr().err
