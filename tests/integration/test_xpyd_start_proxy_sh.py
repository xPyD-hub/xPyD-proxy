"""Tests for core/xpyd_start_proxy.sh parameterization and validation."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "core" / "xpyd_start_proxy.sh"

_MINIMAL_TOPO = [
    "-pn",
    "1",
    "-pt",
    "8",
    "-pd",
    "1",
    "-pw",
    "8",
    "-dn",
    "1",
    "-dt",
    "8",
    "-dd",
    "1",
    "-dw",
    "8",
]


def run_script(*args: str, env_overrides: dict | None = None):
    env = {
        **os.environ,
        "model_path": "dummy-model",
        "XPYD_DRY_RUN": "1",
    }
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=REPO_ROOT / "core",
        env=env,
        capture_output=True,
        text=True,
    )


def extract_running_line(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.startswith("Running: "):
            return line
    raise AssertionError(f"Running line not found in stdout:\n{stdout}")


def test_valid_topology_simple_same_node_instances():
    result = run_script(
        "-pn",
        "2",
        "-pt",
        "4",
        "-pd",
        "4",
        "-pw",
        "8",
        "-dn",
        "2",
        "-dt",
        "2",
        "-dd",
        "8",
        "-dw",
        "8",
    )
    assert result.returncode == 0, result.stderr
    cmd = extract_running_line(result.stdout)
    expected_prefill = (
        "--prefill "
        "10.239.129.9:8100 10.239.129.9:8101 "
        "10.239.129.67:8100 10.239.129.67:8101"
    )
    assert expected_prefill in cmd
    expected_decode = (
        "--decode "
        "10.239.129.81:8200 10.239.129.81:8201 "
        "10.239.129.81:8202 10.239.129.81:8203 "
        "10.239.129.165:8200 10.239.129.165:8201 "
        "10.239.129.165:8202 10.239.129.165:8203"
    )
    assert expected_decode in cmd


def test_valid_topology_cross_node_instance_exposes_main_node_only():
    result = run_script(
        "-pn",
        "2",
        "-pt",
        "16",
        "-pd",
        "1",
        "-pw",
        "8",
        "-dn",
        "4",
        "-dt",
        "8",
        "-dd",
        "4",
        "-dw",
        "8",
    )
    assert result.returncode == 0, result.stderr
    cmd = extract_running_line(result.stdout)
    assert "--prefill 10.239.129.9:8100" in cmd
    prefill_section = cmd.split("--prefill", 1)[1].split("--decode", 1)[0]
    assert "10.239.129.67:8100" not in prefill_section
    expected_decode = (
        "--decode "
        "10.239.129.81:8200 10.239.129.165:8200 "
        "10.239.129.67:8200 10.239.129.21:8200"
    )
    assert expected_decode in cmd


def test_custom_base_ports():
    result = run_script(
        "-pn",
        "1",
        "-pt",
        "8",
        "-pd",
        "1",
        "-pw",
        "8",
        "-dn",
        "1",
        "-dt",
        "8",
        "-dd",
        "1",
        "-dw",
        "8",
        "--prefill-base-port",
        "9100",
        "--decode-base-port",
        "9200",
    )
    assert result.returncode == 0, result.stderr
    cmd = extract_running_line(result.stdout)
    assert "10.239.129.9:9100" in cmd
    assert "10.239.129.81:9200" in cmd


def test_reject_non_power_of_two_tp():
    result = run_script(
        "-pn",
        "1",
        "-pt",
        "3",
        "-pd",
        "8",
        "-pw",
        "8",
        "-dn",
        "1",
        "-dt",
        "8",
        "-dd",
        "1",
        "-dw",
        "8",
    )
    assert result.returncode != 0
    assert "prefill tp_size must be a power of two" in result.stderr


def test_reject_non_power_of_two_dp():
    result = run_script(
        "-pn",
        "1",
        "-pt",
        "8",
        "-pd",
        "3",
        "-pw",
        "8",
        "-dn",
        "1",
        "-dt",
        "8",
        "-dd",
        "1",
        "-dw",
        "8",
    )
    assert result.returncode != 0
    assert "prefill dp_size must be a power of two" in result.stderr


def test_reject_invalid_topology_product_constraint():
    result = run_script(
        "-pn",
        "2",
        "-pt",
        "8",
        "-pd",
        "1",
        "-pw",
        "8",
        "-dn",
        "1",
        "-dt",
        "8",
        "-dd",
        "1",
        "-dw",
        "8",
    )
    assert result.returncode != 0
    assert "prefill topology invalid" in result.stderr


def test_reject_nodes_exceeding_ip_list():
    result = run_script(
        "-pn",
        "8",
        "-pt",
        "8",
        "-pd",
        "8",
        "-pw",
        "8",
        "-dn",
        "1",
        "-dt",
        "8",
        "-dd",
        "1",
        "-dw",
        "8",
    )
    assert result.returncode != 0
    assert "prefill nodes exceeds available IP list length" in result.stderr


def test_reject_non_integer_argument():
    result = run_script(
        "-pn",
        "two",
        "-pt",
        "8",
        "-pd",
        "1",
        "-pw",
        "8",
        "-dn",
        "1",
        "-dt",
        "8",
        "-dd",
        "1",
        "-dw",
        "8",
    )
    assert result.returncode != 0
    assert "prefill nodes must be a positive integer" in result.stderr


def test_reject_zero_or_negative_argument():
    result = run_script(
        "-pn",
        "0",
        "-pt",
        "8",
        "-pd",
        "1",
        "-pw",
        "8",
        "-dn",
        "1",
        "-dt",
        "8",
        "-dd",
        "1",
        "-dw",
        "8",
    )
    assert result.returncode != 0
    assert "prefill nodes must be a positive integer" in result.stderr


def test_model_cli_arg_overrides_env_var():
    """--model CLI arg should override model_path env var."""
    result = run_script(
        *_MINIMAL_TOPO,
        "--model",
        "/cli/model/path",
        env_overrides={"model_path": "/env/model/path"},
    )
    assert result.returncode == 0, result.stderr
    cmd = extract_running_line(result.stdout)
    assert "--model /cli/model/path" in cmd
    assert "/env/model/path" not in cmd


def test_model_env_var_fallback():
    """When --model is not given, script should use model_path env var."""
    result = run_script(
        *_MINIMAL_TOPO,
        env_overrides={"model_path": "/env/fallback/model"},
    )
    assert result.returncode == 0, result.stderr
    cmd = extract_running_line(result.stdout)
    assert "--model /env/fallback/model" in cmd


def test_missing_model_errors():
    """When neither --model nor model_path env var is set, script should fail."""
    result = run_script(
        *_MINIMAL_TOPO,
        env_overrides={"model_path": ""},
    )
    assert result.returncode != 0
    assert "model path is not set" in result.stderr
