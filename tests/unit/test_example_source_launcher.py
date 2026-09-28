"""The example launcher must not use an unrelated installed xpyd command."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "examples/lib/run_proxy.sh"


@pytest.mark.parametrize("arguments", [["--version"], ["proxy", "--help"]])
def test_source_launcher_ignores_installed_command(tmp_path, arguments):
    old_command = tmp_path / "xpyd"
    old_command.write_text("#!/bin/sh\necho WRONG_XPYD >&2\nexit 99\n")
    old_command.chmod(0o755)
    result = subprocess.run(
        ["bash", str(LAUNCHER), *arguments],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "PYTHON": sys.executable,
            "PYTHONPATH": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "WRONG_XPYD" not in result.stderr
    assert "xpyd" in result.stdout.lower()


def test_source_launcher_preserves_failure_status(tmp_path):
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--validate-config", str(tmp_path / "missing.yaml")],
        env={**os.environ, "PYTHON": sys.executable},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0


@pytest.mark.parametrize("arguments", [["--version"], ["proxy", "--help"]])
def test_source_launcher_ignores_other_checkout(tmp_path, arguments):
    package = tmp_path / "xpyd"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "proxy.py").write_text(
        "def main():\n    print('WRONG_CHECKOUT')\n    raise SystemExit(99)\n"
    )
    (tmp_path / "sitecustomize.py").write_text(
        "import atexit\n"
        "import sys\n"
        "@atexit.register\n"
        "def report_source():\n"
        "    module = sys.modules.get('xpyd.proxy')\n"
        "    if module is not None:\n"
        "        print('XPYD_SOURCE=' + module.__file__)\n"
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER), *arguments],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHON": sys.executable,
            "PYTHONPATH": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "WRONG_CHECKOUT" not in result.stdout
    assert f"XPYD_SOURCE={ROOT / 'xpyd/proxy.py'}" in result.stdout.splitlines()


def test_source_launcher_preserves_relative_config_path(tmp_path):
    config = tmp_path / "local config.yaml"
    config.write_text("model: demo\ndecode:\n  - 127.0.0.1:8200\n")
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--validate-config", config.name],
        cwd=tmp_path,
        env={**os.environ, "PYTHON": sys.executable},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
