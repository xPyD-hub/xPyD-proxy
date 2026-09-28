"""CPU wheel reuse must match the requested build and Python ABI."""

import runpy
from pathlib import Path
from unittest.mock import patch

import pytest

CACHE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2]
        / "examples/disaggregated/opt-125m-cpu-nixl/wheel_cache.py"
    )
)
SELECT = CACHE["cached_wheel"]


def test_empty_cache(tmp_path):
    assert SELECT(tmp_path, "v1.3.0") == ""


@pytest.mark.parametrize(
    "filename",
    [
        "nixl_cu12-1.2.0-py3-none-any.whl",
        "nixl-1.3.0-py3-none-any.whl",
        "nixl_cu12-1.3.0-cp39-cp39-win_amd64.whl",
    ],
)
def test_wrong_wheel_fails(tmp_path, filename):
    (tmp_path / filename).touch()
    with pytest.raises(ValueError, match="Incompatible"):
        SELECT(tmp_path, "v1.3.0")


def test_matching_wheel(tmp_path):
    wheel = tmp_path / "nixl_cu12-1.3.0-py3-none-any.whl"
    wheel.touch()
    assert SELECT(tmp_path, "v1.3.0") == str(wheel)


def test_ambiguous_cache_fails(tmp_path):
    for build in ("1", "2"):
        (tmp_path / f"nixl_cu12-1.3.0-{build}-py3-none-any.whl").touch()
    with pytest.raises(ValueError, match="Ambiguous"):
        SELECT(tmp_path, "v1.3.0")


def test_fingerprint_tracks_versions_and_abi():
    fingerprint = CACHE["fingerprint"]
    baseline = fingerprint("v1.3.0", "0.25.0")
    assert baseline == fingerprint("1.3.0", "0.25.0")
    assert baseline != fingerprint("v1.2.0", "0.25.0")
    assert baseline != fingerprint("v1.3.0", "0.26.0")
    with patch("sysconfig.get_config_var", return_value="different-abi"):
        assert baseline != fingerprint("v1.3.0", "0.25.0")


def test_workflows_share_nixl_cache_key():
    import yaml

    root = Path(__file__).resolve().parents[2]
    keys = []
    for name in ("cpu-example.yml", "scheduler-matrix.yml"):
        workflow = yaml.safe_load((root / ".github/workflows" / name).read_text())
        for job in workflow["jobs"].values():
            for step in job["steps"]:
                settings = step.get("with", {})
                if settings.get("path") == "~/.cache/xpyd-nixl-wheels":
                    keys.append(settings["key"])
    assert len(keys) == 3
    assert len(set(keys)) == 1
