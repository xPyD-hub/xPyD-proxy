import ctypes
import os
import runpy
import signal
import socket
import subprocess
import sys
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


@pytest.fixture
def owned_group():
    if sys.platform != "linux":
        pytest.skip("Process-group cleanup requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    groups = []

    def spawn(behavior):
        child_code = (
            "import signal,sys,time\n"
            + (
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                if behavior == "ignore"
                else "signal.signal(signal.SIGTERM, "
                "lambda *_: (time.sleep(0.2), sys.exit(0)))\n"
            )
            + "print('ready', flush=True)\ntime.sleep(60)\n"
        )
        parent_code = (
            "import subprocess,sys,time\n"
            "child=subprocess.Popen([sys.executable,'-c',sys.argv[1]],"
            "stdout=subprocess.PIPE,text=True)\n"
            "child.stdout.readline()\n"
            "print(child.pid,flush=True)\ntime.sleep(60)\n"
        )
        parent = subprocess.Popen(
            [sys.executable, "-c", parent_code, child_code],
            start_new_session=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        child = int(parent.stdout.readline())
        groups.append((parent, child))
        return parent

    try:
        yield spawn
    finally:
        for parent, child in groups:
            try:
                os.killpg(parent.pid, signal.SIGKILL)
            except ProcessLookupError:
                # The fixture group may already have exited.
                pass
            parent.wait(timeout=5)
            os.waitpid(child, 0)
            parent.stdout.close()
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


def test_stop_waits_for_workers_after_leader_exit(owned_group):
    parent = owned_group("delay")
    H["stop"](parent, timeout=2, kill_timeout=2)
    assert parent.poll() is not None
    assert not H["group_running"](parent.pid)


@pytest.mark.parametrize("leader_exited", [False, True])
def test_stop_kills_unresponsive_workers_and_reports_failure(
    owned_group, leader_exited
):
    parent = owned_group("ignore")
    if leader_exited:
        parent.terminate()
        parent.wait(timeout=5)
    with pytest.raises(RuntimeError, match="required forced termination"):
        H["stop"](parent, timeout=0.2, kill_timeout=2)
    assert not H["group_running"](parent.pid)


@pytest.mark.parametrize("failure", ["stop", "release", None])
def test_tracking_retained_until_all_resources_released(monkeypatch, failure):
    process, other = object(), object()
    processes = {"backend": process, "other": other}
    namespace = H["stop_and_release"].__globals__

    def stop(value):
        assert value is process
        if failure == "stop":
            raise RuntimeError("stop failed")

    def wait(label, predicate, remaining):
        assert remaining == {"other": other}
        assert predicate()
        if failure == "release":
            raise RuntimeError("release failed")

    monkeypatch.setitem(namespace, "stop", stop)
    monkeypatch.setitem(namespace, "wait_for", wait)
    if failure:
        with pytest.raises(RuntimeError, match=failure + " failed"):
            H["stop_and_release"](processes, "backend", lambda: True)
        assert processes["backend"] is process
    else:
        H["stop_and_release"](processes, "backend", lambda: True)
        assert processes == {"other": other}
