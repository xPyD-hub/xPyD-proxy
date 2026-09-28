"""Keep static and generated GPU example health settings consistent."""

import runpy
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from xpyd.config import HealthCheckConfig

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
CONFIGS = sorted(
    path
    for path in EXAMPLES.rglob("xpyd*.yaml")
    if "llama" in str(path.relative_to(EXAMPLES)).lower()
)
GENERATORS = sorted(
    EXAMPLES.glob("disaggregated/Llama_8P8D_*/generate_proxy_configs.py")
)


def assert_enabled(config):
    health = HealthCheckConfig(**config["health_check"])
    assert health.enabled
    assert health.interval_seconds == 2
    assert health.timeout_seconds == 2


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: str(path.relative_to(ROOT)))
def test_static_gpu_health(path):
    assert_enabled(yaml.safe_load(path.read_text()))


@pytest.mark.parametrize("path", GENERATORS, ids=lambda path: path.parent.name)
def test_generated_gpu_health(path, tmp_path):
    generated = []

    def capture(output, content):
        generated.append(yaml.safe_load(content))

    with patch.object(Path, "mkdir"), patch.object(Path, "write_text", capture):
        runpy.run_path(str(path))

    assert len(generated) == 10
    for config in generated:
        assert_enabled(config)
