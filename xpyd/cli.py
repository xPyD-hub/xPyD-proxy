# SPDX-License-Identifier: Apache-2.0
"""CLI parsing and configuration commands, independent of the proxy runtime."""

import argparse
import os
import sys
from pathlib import Path

from xpyd.config import ProxyConfig

_VERSION = "1.8.0"


def _valid_port(value: str) -> int:
    """Parse a TCP port for argparse with a concise user-facing error."""
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _build_parser():
    """Build the subcommand argument parser for the proxy CLI."""
    parser = argparse.ArgumentParser(
        prog="xpyd",
        description="xPyD — lightweight disaggregated proxy server",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"%(prog)s {_VERSION}",
    )

    subparsers = parser.add_subparsers(dest="command")

    proxy_parser = subparsers.add_parser(
        "proxy",
        prog="xpyd [proxy]",
        help="Start the proxy server (default command)",
    )
    proxy_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    proxy_parser.add_argument(
        "--validate-config",
        type=str,
        default=None,
        metavar="FILE",
        help="Validate YAML config and exit (no server start)",
    )
    proxy_parser.add_argument(
        "--init-config",
        nargs="?",
        const="./xpyd.yaml",
        default=None,
        metavar="PATH",
        help="Generate xpyd.yaml, optionally through an interactive wizard "
        "(default path: ./xpyd.yaml)",
    )
    proxy_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing --init-config destination",
    )
    proxy_parser.add_argument(
        "--port",
        type=_valid_port,
        default=None,
        help="Override the port from config",
    )
    proxy_parser.add_argument(
        "--log-level",
        choices=("debug", "info", "warning", "error"),
        default=None,
        dest="log_level",
        help="Override log level: debug|info|warning|error",
    )
    proxy_parser.add_argument(
        "--disaggregated-mode",
        choices=("direct", "nixl", "zmq"),
        default=None,
        help="disaggregated transfer mode (default: config value or direct)",
    )
    proxy_parser.add_argument(
        "--first-token-source",
        choices=("prefill", "decode"),
        default=None,
        help="Backend that provides the first client-visible token",
    )

    # fix-config subcommand
    fix_parser = subparsers.add_parser(
        "fix-config",
        help="Auto-fix common config mistakes",
    )
    fix_parser.add_argument(
        "config_path",
        type=str,
        help="Path to YAML configuration file to fix",
    )
    fix_parser.add_argument(
        "--write",
        action="store_true",
        default=False,
        help="Write fixes back to the file (creates timestamped .bak backup). "
        "Note: does not preserve YAML comments or formatting.",
    )
    fix_parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Interactively acknowledge ambiguous suggestions one by one",
    )

    return parser


def _normalize_cli_args(argv: list[str]) -> list[str]:
    """Treat omitted subcommands as the default ``proxy`` command."""
    if argv and argv[0] == "proxy":
        if len(argv) > 1 and argv[1] in {"fix-config", "--version", "-V"}:
            return argv[1:]
        return argv
    if argv and argv[0] in {"fix-config", "--version", "-V"}:
        return argv
    return ["proxy", *argv]


def _resolve_config_path(args):
    """Resolve the config file path: --config > XPYD_CONFIG env > ./xpyd.yaml.

    Returns the path string, or ``None`` when no config can be found.
    """
    if args.config:
        return args.config
    env_config = os.environ.get("XPYD_CONFIG")
    if env_config:
        return env_config
    default_path = "./xpyd.yaml"
    if os.path.exists(default_path):
        print(
            "No config specified; found ./xpyd.yaml and using it.",
            flush=True,
        )
        return default_path
    return None


def _print_config_summary(config: ProxyConfig) -> None:
    """Print topology-aware details for a validated configuration."""
    if config.instances is None:
        topology = "disaggregated" if config.prefill else "decode-only"
        print(f"  topology: {topology}")
        if config.prefill:
            print(f"  transfer: {config.disaggregated_mode}")
        print(f"  model: {config.model}")
        print(f"  prefill: {len(config.prefill)} instances")
        print(f"  decode: {len(config.decode)} instances")
    else:
        roles = {instance.role for instance in config.instances}
        topology = "aggregated" if roles == {"aggregated"} else "disaggregated"
        print(f"  topology: {topology}")
        if topology == "disaggregated":
            print(f"  transfer: {config.disaggregated_mode}")
        models = sorted(
            {instance.model for instance in config.instances if instance.model}
        )
        print(f"  models: {', '.join(models) if models else 'auto-detected'}")
        for role in ("aggregated", "prefill", "decode"):
            count = sum(instance.role == role for instance in config.instances)
            if count:
                print(f"  {role}: {count} instances")
    print(f"  port: {config.port}")
    print(f"  log_level: {config.log_level}")


def _apply_cli_overrides(config: ProxyConfig, args) -> ProxyConfig:
    """Apply explicitly supplied CLI values and revalidate the result."""
    overrides = {
        field: value
        for field in (
            "port",
            "log_level",
            "disaggregated_mode",
            "first_token_source",
        )
        if (value := getattr(args, field, None)) is not None
    }
    if not overrides:
        return config

    model_schedulers = config._model_schedulers.copy()
    updated = ProxyConfig.model_validate({**config.model_dump(), **overrides})
    updated._model_schedulers = model_schedulers
    return updated


def load_config(argv: list[str] | None = None) -> ProxyConfig | None:
    """Run configuration commands or return the configuration to start a proxy."""
    from xpyd.init_config import generate_config

    parser = _build_parser()
    args = parser.parse_args(
        _normalize_cli_args(sys.argv[1:] if argv is None else argv)
    )

    if args.command == "fix-config":
        from xpyd.config_fixer import run_fix_config

        sys.exit(
            run_fix_config(
                args.config_path,
                write=args.write,
                interactive=args.interactive,
            )
        )
    elif args.command == "proxy":
        if args.force and args.init_config is None:
            parser.error("--force can only be used with --init-config")

        # --init-config: generate template and exit
        if args.init_config is not None:
            output_path = Path(args.init_config)
            if output_path.exists() and not args.force:
                parser.error(
                    f"config already exists: {output_path}; use --force to overwrite"
                )
            try:
                generate_config(args.init_config)
            except OSError as exc:
                parser.exit(1, f"xpyd: failed to write config: {exc}\n")
            return None

        # --validate-config: validate and exit
        if args.validate_config:
            config_path = args.validate_config
            try:
                config = ProxyConfig.from_yaml(config_path)
                print(f"Config is valid: {config_path}")
                _print_config_summary(config)
                sys.exit(0)
            except Exception as exc:
                print(f"Config validation failed: {exc}", file=sys.stderr)
                sys.exit(1)

        # Resolve config path with precedence
        config_path = _resolve_config_path(args)
        if config_path is None:
            config_path = "./xpyd.yaml"
            print(
                "No config specified and ./xpyd.yaml was not found; "
                "starting config initialization."
            )
            generate_config(config_path)
            return None
        try:
            config = ProxyConfig.from_yaml(config_path)
            return _apply_cli_overrides(config, args)
        except (OSError, ValueError) as exc:
            parser.exit(1, f"xpyd: failed to load config: {exc}\n")
    else:
        parser.print_help()
    return None
