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
