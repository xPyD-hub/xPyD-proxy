import runpy
import socket
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
H = runpy.run_path(str(ROOT / "examples/accelerator_lifecycle/run.py"))


@pytest.mark.parametrize("topology", H["LAYOUTS"])
def test_topology_status_and_model_isolation(topology):
    nodes = H["instances"](topology)
    status = {
        f"{role}_instances": [
            {**node, "status": "healthy"} for node in nodes if node["role"] == role
        ]
        for role in ("prefill", "decode", "aggregated")
    }
    online = {node["address"] for node in nodes}
    assert H["check_status"](status, nodes, online)
    online.remove(nodes[0]["address"])
    assert not H["check_status"](status, nodes, online)
    status[f"{nodes[0]['role']}_instances"][0]["status"] = "unhealthy"
    assert H["check_status"](status, nodes, online)
    status[f"{nodes[0]['role']}_instances"][0]["address"] = "wrong-backend"
    with pytest.raises(AssertionError):
        H["check_status"](status, nodes, online)


@pytest.mark.parametrize(
    "topology,expected",
    [
        ("aggregated", "mode=aggregated | aggregated=0/2 online"),
        ("direct", "mode=disaggregated | P=0/2 online | D=0/2 online"),
        ("mixed", "mode=mixed | P=0/1 online | D=0/1 online | aggregated=0/2 online"),
    ],
)
def test_heartbeat_roles(topology, expected):
    assert H["heartbeat"](topology, H["instances"](topology), set()) == expected


def test_template_and_backend_names():
    from xpyd.config import ProxyConfig

    path = ROOT / "examples/accelerator_lifecycle/xpyd.yaml"
    config = ProxyConfig.from_yaml(path)
    assert config.health_check.enabled
    for node in H["instances"]("mixed"):
        command = H["backend_command"]("/model", node)
        assert command[command.index("--served-model-name") + 1] == node["model"]


@pytest.mark.parametrize(
    "device,output,expected",
    [
        ("cuda", "42\n", 42),
        (
            "xpu",
            '{"memory":{"used_mib":{"tile_0":{"current":41},"tile_1":{"current":3}}}}',
            44,
        ),
    ],
)
def test_memory_telemetry(monkeypatch, device, output, expected):
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: output)
    assert H["memory_mib"](device, 0) == expected


def test_unknown_memory_schema_fails(monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "{}")
    with pytest.raises(KeyError):
        H["memory_mib"]("xpu", 0)


def test_listening_port_is_not_free():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]
        assert not H["port_free"](port)
    assert H["port_free"](port)
